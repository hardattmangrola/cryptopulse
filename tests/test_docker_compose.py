"""
Unit tests for Phase 5 Docker Compose and Containerization configurations.
Validates YAML structure, services, health checks, ports, volumes, and Dockerfiles.
"""

import pytest
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
COMPOSE_FILE = BASE_DIR / "docker-compose.yml"


def test_docker_compose_file_exists():
    """Verify docker-compose.yml exists in the project root."""
    assert COMPOSE_FILE.exists()


def test_docker_compose_services_and_ports():
    """Verify all required microservices and persistence services are defined with correct ports."""
    with open(COMPOSE_FILE, "r") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    required_services = ["postgres", "mlflow", "ingestion", "inference", "training", "monitoring", "prometheus", "grafana"]

    for s in required_services:
        assert s in services, f"Missing required service '{s}' in docker-compose.yml"

    # Port checks
    assert "5432:5432" in services["postgres"]["ports"]
    assert "5000:5000" in services["mlflow"]["ports"]
    assert "8001:8001" in services["ingestion"]["ports"]
    assert "8002:8002" in services["inference"]["ports"]
    assert "8003:8003" in services["training"]["ports"]
    assert "8004:8004" in services["monitoring"]["ports"]
    assert "9090:9090" in services["prometheus"]["ports"]
    assert "3000:3000" in services["grafana"]["ports"]


def test_docker_compose_healthcheck_and_dependencies():
    """Verify postgres health check and service dependencies."""
    with open(COMPOSE_FILE, "r") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data["services"]

    # Postgres healthcheck
    assert "healthcheck" in services["postgres"]
    assert "pg_isready" in str(services["postgres"]["healthcheck"]["test"])

    # Dependencies on healthy postgres
    assert services["ingestion"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["inference"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert services["training"]["depends_on"]["postgres"]["condition"] == "service_healthy"


def test_docker_compose_volumes_and_networks():
    """Verify persistent volumes and bridge network."""
    with open(COMPOSE_FILE, "r") as f:
        compose_data = yaml.safe_load(f)

    volumes = compose_data.get("volumes", {})
    assert "postgres_data" in volumes
    assert "mlflow_data" in volumes

    networks = compose_data.get("networks", {})
    assert "cryptopulse_net" in networks


def test_dockerfiles_exist():
    """Verify Dockerfiles exist for each application microservice."""
    for service_name in ["ingestion", "inference", "training", "monitoring"]:
        df = BASE_DIR / "services" / service_name / "Dockerfile"
        assert df.exists(), f"Missing Dockerfile for {service_name}"
        content = df.read_text()
        assert "FROM python:3.11-slim" in content
        assert "COPY shared/ /app/shared/" in content
