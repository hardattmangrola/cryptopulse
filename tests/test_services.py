"""
CryptoPulse - Core Services Smoke Test
Validates that all microservices (ingestion, inference, training, monitoring) are present and importable.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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
