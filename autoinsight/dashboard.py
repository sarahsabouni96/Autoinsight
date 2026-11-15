import os

import requests
import plotly.express as px
from dash import Dash, Input, Output, State, dcc, html, dash_table, no_update

from .analyst import answer

API = os.getenv("API_URL", "http://127.0.0.1:8000")
app = Dash(__name__, title="AutoInsight")
server = app.server


def fetch(path):
    response = requests.get(f"{API}/{path}", timeout=8)
    response.raise_for_status()
    return response.json()


def field(label, control):
    return html.Div([html.Label(label, htmlFor=control.id), control], className="field")


def layout():
    try:
        meta = fetch("metadata")
    except requests.RequestException:
        return html.Main([html.H1("AutoInsight"), html.P("Prediction service unavailable. Refresh to retry.")])
    makes = list(meta["models_by_make"])
    ranges = meta["ranges"]

    def number(label, key, default, step):
        low, high = ranges[label]
        return field(label, dcc.Input(id=key, type="number", min=low, max=high, step=step,
                                      value=min(high, max(low, default))))
    importance = sorted(meta["importance"].items(), key=lambda pair: pair[1])
    fig = px.bar(x=[v for _, v in importance], y=[k for k, _ in importance], orientation="h",
                 labels={"x": "Increase in MAE after shuffling", "y": ""})
    fig.update_layout(margin=dict(t=10, b=40, l=10, r=20), height=270)
    return html.Main([
        html.Header([html.H1("AutoInsight"), html.P("Used car price estimator")]),
        html.Section([
            field("Manufacturer", dcc.Dropdown(makes, makes[0], id="make", clearable=False)),
            field("Model", dcc.Dropdown(id="model", clearable=False)),
            field("Fuel type", dcc.Dropdown(meta["fuels"], meta["fuels"][0], id="fuel", clearable=False)),
            number("Year of manufacture", "year", 2016, 1), number("Mileage", "mileage", 60000, 1),
            number("Engine size", "engine", 2, .1),
        ], className="inputs"),
        html.Button("Estimate price", id="submit"), html.Div(id="result", role="status"),
        html.H2("Prediction history"), html.Div(id="history"),
        html.H2("Model performance"),
        html.P(f"Held-out MAE {meta['test']['MAE']:,.0f} · RMSE {meta['test']['RMSE']:,.0f} · "
               f"R² {meta['test']['R2']:.3f} · median baseline MAE {meta['baseline']['MAE']:,.0f}"),
        html.H3("Historical median price by manufacturer"),
        dcc.Graph(figure=px.bar(x=list(meta["make_medians"]), y=list(meta["make_medians"].values()),
                                labels={"x": "Manufacturer", "y": "Median price in fitting data"})),
        html.H2("Ask about the model"),
        field("Question", dcc.Input(id="question", type="text", placeholder="Which features matter most?")),
        html.Button("Ask analyst", id="ask"), html.P(id="answer", role="status"),
        html.H3("Feature importance"), dcc.Graph(figure=fig),
        dcc.Store(id="catalog", data=meta["models_by_make"]),
        dcc.Store(id="refresh", data=0)
    ])


app.layout = layout


@app.callback(Output("model", "options"), Output("model", "value"), Input("make", "value"), State("catalog", "data"))
def models(make, catalog):
    choices = catalog.get(make, [])
    return choices, choices[0] if choices else None


@app.callback(Output("result", "children"), Output("refresh", "data"), Input("submit", "n_clicks"),
              State("make", "value"), State("model", "value"), State("fuel", "value"),
              State("year", "value"), State("mileage", "value"), State("engine", "value"),
              prevent_initial_call=True)
def estimate(clicks, make, model, fuel, year, mileage, engine):
    car = dict(manufacturer=make, model=model, fuel_type=fuel, year=year, mileage=mileage, engine_size=engine)
    try:
        response = requests.post(f"{API}/predict", json=car, timeout=8)
        if response.status_code == 422:
            detail = response.json()["detail"]
            return str(detail) if isinstance(detail, str) else "Check the six input fields.", no_update
        response.raise_for_status()
        result = response.json()
        return html.Div([html.H2(f"Estimated price {result['estimate']:,.0f}"),
                         html.P(f"90% prediction interval {result['lower']:,.0f} – {result['upper']:,.0f} "
                                f"in dataset price units · segment {result['segment']}")]), clicks
    except requests.RequestException:
        return "Could not confirm the estimate. Check prediction history before retrying.", no_update


@app.callback(Output("history", "children"), Input("refresh", "data"))
def recent(_):
    try:
        rows = fetch("history")
    except requests.RequestException:
        return "History unavailable. Refresh to retry."
    if not rows:
        return "Your first estimate will appear here."
    records = [{"Time (UTC)": row["created"][:19].replace("T", " "), "Manufacturer": row["car"]["manufacturer"],
                "Model": row["car"]["model"], "Estimate": row["result"]["estimate"]} for row in rows]
    return dash_table.DataTable(data=records, columns=[{"name": k, "id": k} for k in records[0]], page_size=5)


@app.callback(Output("answer", "children"), Input("ask", "n_clicks"), State("question", "value"),
              prevent_initial_call=True)
def ask(_, question):
    try:
        return answer((question or "")[:500], fetch("metadata"))
    except requests.RequestException:
        return "Model statistics unavailable."


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050)
