"""Generate a self-contained exploratory report using fitting rows only."""
from pathlib import Path

import plotly.express as px


from .data import LOG, TARGET, YEAR, partitions, read_sales


def main():
    frame, _ = read_sales()
    fit, _, _ = partitions(frame)
    columns = LOG + [YEAR, TARGET]
    summary = fit[columns].describe().T
    width = summary["75%"] - summary["25%"]
    summary["Outside 1.5 IQR"] = ((fit[columns] < summary["25%"] - 1.5 * width)
                                  | (fit[columns] > summary["75%"] + 1.5 * width)).sum()
    sample = fit.sample(min(len(fit), 2000), random_state=42)
    parts = ["<!doctype html><meta charset='utf-8'><title>AutoInsight exploration</title>",
             "<h1>Fitting data exploration</h1>",
             f"<p>{len(fit):,} fitting records. Scatterplots show a seeded sample of {len(sample):,}.</p>",
             summary.to_html()]
    for index, col in enumerate(LOG):
        fig = px.scatter(sample, x=col, y=TARGET, color="Manufacturer", opacity=.6)
        parts.append(fig.to_html(full_html=False, include_plotlyjs=(index == 0)))
    path = Path("artifacts/exploration.html")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
