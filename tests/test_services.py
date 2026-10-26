"""
CryptoPulse - Core Services Smoke Test
Validates that all microservices (ingestion, inference, training, monitoring) are present and importable.
"""

def test_all_services_present():
    """Verify all 4 microservices are present and initialized."""
    from services.ingestion.app import app as ingestion_app
    from services.inference.app import app as inference_app
    from services.training.app import app as training_app
    from services.monitoring.app import app as monitoring_app

    assert ingestion_app is not None
    assert inference_app is not None
    assert training_app is not None
    assert monitoring_app is not None
