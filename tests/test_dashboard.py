from fastapi.testclient import TestClient

from autoinsight import dashboard
from autoinsight.api import create_app
from test_system import fitted


def test_dashboard_roundtrip(fitted, tmp_path, monkeypatch):
    with TestClient(create_app(fitted, f"sqlite:///{tmp_path / 'ui.db'}")) as api:
        monkeypatch.setattr(dashboard, "fetch", lambda path: api.get("/" + path).json())
        monkeypatch.setattr(dashboard.requests, "post", lambda url, json, timeout: api.post("/predict", json=json))
        assert dashboard.server.test_client().get("/_dash-layout").status_code == 200
        assert dashboard.models("Ford", {"Ford": ["Fiesta"]}) == (["Fiesta"], "Fiesta")
        result, changed = dashboard.estimate(1, "Ford", "Fiesta", "Petrol", 2015, 50000, 1.6)
        assert changed == 1 and "Estimated price" in str(result)
        assert dashboard.recent(1).data[0]["Model"] == "Fiesta"
        assert "MAE" in dashboard.ask(1, "How accurate is the model?")
