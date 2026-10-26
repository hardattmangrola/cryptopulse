"""
Unit tests for Phase 6 Monitoring (Prometheus & Grafana) configurations.
Validates Prometheus scrape jobs, Grafana datasource provisioning,
and the CryptoPulse Grafana dashboard JSON schema.
"""

import json
import pytest
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROMETHEUS_CONFIG = BASE_DIR / "monitoring" / "prometheus" / "prometheus.yml"
GRAFANA_DATASOURCE = BASE_DIR / "monitoring" / "grafana" / "provisioning" / "datasources" / "datasource.yml"
GRAFANA_PROVIDERS = BASE_DIR / "monitoring" / "grafana" / "provisioning" / "dashboards" / "dashboards.yml"
DASHBOARD_JSON = BASE_DIR / "monitoring" / "grafana" / "dashboards" / "cryptopulse_dashboard.json"


def test_prometheus_configuration_exists_and_valid():
    """Verify prometheus.yml exists, is valid YAML, and scrapes all microservices."""
    assert PROMETHEUS_CONFIG.exists()
    with open(PROMETHEUS_CONFIG, "r") as f:
        data = yaml.safe_load(f)

    assert "scrape_configs" in data
    jobs = [job["job_name"] for job in data["scrape_configs"]]
    assert "ingestion-service" in jobs
    assert "inference-service" in jobs
    assert "training-service" in jobs


def test_grafana_datasource_provisioning():
    """Verify Grafana datasource is configured to query Prometheus proxy."""
    assert GRAFANA_DATASOURCE.exists()
    with open(GRAFANA_DATASOURCE, "r") as f:
        data = yaml.safe_load(f)

    assert "datasources" in data
    ds = data["datasources"][0]
    assert ds["name"] == "Prometheus"
    assert ds["type"] == "prometheus"
    assert "prometheus:9090" in ds["url"]


def test_grafana_dashboard_provisioning():
    """Verify Grafana dashboard provider config maps to dashboards folder."""
    assert GRAFANA_PROVIDERS.exists()
    with open(GRAFANA_PROVIDERS, "r") as f:
        data = yaml.safe_load(f)

    assert "providers" in data
    assert data["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"


def test_grafana_dashboard_panels_and_metrics():
    """Verify cryptopulse_dashboard.json contains panels querying real metrics."""
    assert DASHBOARD_JSON.exists()
    with open(DASHBOARD_JSON, "r") as f:
        dash = json.load(f)

    assert dash["uid"] == "cryptopulse-overview"
    assert "panels" in dash
    assert len(dash["panels"]) >= 6

    # Extract all Prometheus target expressions
    expressions = []
    for panel in dash["panels"]:
        for target in panel.get("targets", []):
            if "expr" in target:
                expressions.append(target["expr"])

    # Ensure expressions reference actual application metric names
    all_expr_str = " ".join(expressions)
    assert "cryptopulse_active_model_version" in all_expr_str
    assert "cryptopulse_candles_ingested_total" in all_expr_str
    assert "cryptopulse_predictions_total" in all_expr_str
    assert "cryptopulse_binance_fetch_total" in all_expr_str
