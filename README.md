# CryptoPulse: Real-Time Cryptocurrency Price Movement Classification with Continuous MLOps

**CryptoPulse** is an end-to-end, microservice-based Continuous MLOps platform that predicts Bitcoin (BTC/USDT) short-term price movement direction (`UP`, `STABLE`, or `DOWN`) **1 minute ahead** with **real-time ground-truth evaluation displayed on a live Web UI within 60 seconds**.

Designed specifically for academic, university-level demonstration and practical learning, the system covers the entire machine learning and MLOps lifecycle:
1. **Historical Model Training & MLflow Experiment Tracking**: XGBoost classification model logged and versioned in MLflow (`mlflow.xgboost.log_model`).
2. **Live Market Data Ingestion**: Non-blocking polling of 1-minute completed candles from Binance's public REST API (`GET /api/v3/klines`).
3. **Real-Time Inference REST API & Interactive Web UI**: High-throughput inference with live Binance ticker, animated probability distribution bars, and live countdown badges.
4. **Delayed 1-Minute Ground Truth Evaluation**: Evaluates predicted direction against actual price movements ($t + 60\text{s}$), displaying `CORRECT` / `INCORRECT` status live on the Web UI.
5. **Continuous Data Drift Detection**: Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) test comparing production feature distributions against baseline reference quantiles.
6. **Automated Retraining Controller**: Triggers retraining upon detected feature drift ($\text{PSI} \ge 0.25$) or performance degradation (Macro F1 $< 0.30$) with cooldown safety.
7. **End-to-End Observability**: Standard Prometheus metrics scrape endpoints (`/metrics`) and Grafana monitoring dashboards.

---

## Architecture Overview

![architecture](public\architecture.png)

---

## Microservices Breakdown

| Microservice | Port | Description | Key Endpoints |
|:---|:---:|:---|:---|
| **Web UI & Inference** | `8002` | Live prediction interface and 1-minute movement inference engine. Evaluates ground truth inline. | `GET /`<br>`POST /predict`<br>`GET /predictions/recent`<br>`POST /reload-model`<br>`GET /metrics` |
| **Ingestion** | `8001` | Polls Binance public klines every 60s, validates finalized candles, and persists to PostgreSQL. | `GET /health`<br>`POST /poll`<br>`GET /recent-candles`<br>`GET /metrics` |
| **Training** | `8003` | Trains XGBoost classifier on historical data with chronological split, quality gates, and MLflow logging. | `POST /train`<br>`GET /health`<br>`GET /metrics` |
| **Continuous Monitoring** | `8004` | Runs background loops for 1-minute delayed evaluation, PSI/KS drift detection, and retraining controller. | `POST /evaluate`<br>`POST /drift/check`<br>`POST /controller/retrain`<br>`GET /status`<br>`GET /metrics` |
| **MLflow Tracking** | `5000` | Model artifact repository and experiment tracking. | Web UI at `http://localhost:5000` |
| **Prometheus** | `9090` | Timeseries metric scraper across all microservices. | Web UI at `http://localhost:9090` |
| **Grafana** | `3000` | Pre-provisioned operational and ML performance dashboards. | Web UI at `http://localhost:3000` |

---

## 1-Minute Prediction Horizon & Live Ground Truth Evaluation

### Target Class Definitions
For a candle at timestamp $t$ with close price $P_t$, the model predicts price movement over the next **1 minute** ($t + 60\text{s}$) with close price $P_{t+60\text{s}}$:
$$\text{Return}_{1\text{m}} = \frac{P_{t+60\text{s}} - P_t}{P_t}$$

- **`UP`**: $\text{Return}_{1\text{m}} > +0.05\%$
- **`DOWN`**: $\text{Return}_{1\text{m}} < -0.05\%$
- **`STABLE`**: $-0.05\% \le \text{Return}_{1\text{m}} \le +0.05\%$

### Real-Time Evaluation Lifecycle
1. The user clicks **⚡ Run Real-Time Prediction (Next 1m)** on `http://localhost:8002/`.
2. The model immediately classifies movement (`UP`, `STABLE`, or `DOWN`) and stores the prediction in PostgreSQL with status **`PENDING`**.
3. In the Web UI table, a pulsing amber badge displays:
   ```
   ⏳ Evaluating (58s left)
   ```
4. The Web UI automatically polls `/predictions/recent` every 3 seconds.
5. As soon as 60 seconds have elapsed, the actual candle price $P_{t+60\text{s}}$ is retrieved, ground truth is determined, and the UI status immediately updates live:
   - `✓ CORRECT` (Green badge) with actual return percentage (e.g. `Actual: UP (+0.082%)`).
   - `✗ INCORRECT` (Red badge) with actual return percentage (e.g. `Actual: DOWN (-0.061%)`).
6. The KPI summary strip calculates and updates the live **1-Minute Accuracy %** and **Total Evaluated** metrics dynamically.

---

## Continuous Maintenance (Phase 7)

### 1. Data Drift Detection (PSI & KS-Test)
- Baseline reference deciles ($q_{0.10}, \dots, q_{0.90}$) for all 20 technical indicators are extracted from training data into `models/reference_distribution.json`.
- Production observations are continuously binned:
  $$\text{PSI} = \sum_{i=1}^{10} (Actual_i - Expected_i) \times \ln\left(\frac{Actual_i}{Expected_i}\right)$$
- Drift thresholds:
  - $\text{PSI} < 0.10$: `NO_DRIFT`
  - $0.10 \le \text{PSI} < 0.25$: `MODERATE_DRIFT`
  - $\text{PSI} \ge 0.25$: `SIGNIFICANT_DRIFT` (triggers automated retraining).
- Metrics exported to Prometheus gauge `cryptopulse_feature_drift_psi{feature="..."}`.

### 2. Retraining Controller with Enforced Cooldown
- Monitors for:
  - Any feature with $\text{PSI} \ge 0.25$.
  - Rolling Macro F1 $< 0.30$.
- **Cooldown window**: Enforces a minimum of 10 minutes between training runs to prevent thrashing.
- **Quality Gate**: Candidate model must achieve Macro F1 $\ge 0.30$. If approved (`PROMOTED`), signals inference service via `POST /reload-model` for zero-downtime hot-reload.

---

## Quick Start & Verification

### 1. Run Complete Test Suite
```bash
pytest tests -v
```
*(All 54 tests pass across features, database, containerization, microservices, and continuous maintenance).*

### 2. Train Model and Log to MLflow
```bash
python services/training/train.py --rows 100000
```
- Trains on recent 1-minute historical Bitcoin data.
- Persists model artifact to `models/baseline_model.joblib`.
- Logs parameters, metrics, and models to MLflow (`mlflow.xgboost.log_model`).

### 3. Launch Complete Docker Compose Stack
```bash
docker compose up --build -d
```
Access all services locally:
- **CryptoPulse Web UI**: [http://localhost:8002](http://localhost:8002)
- **Monitoring & Drift API**: [http://localhost:8004/docs](http://localhost:8004/docs)
- **Training API**: [http://localhost:8003/docs](http://localhost:8003/docs)
- **Ingestion API**: [http://localhost:8001/docs](http://localhost:8001/docs)
- **MLflow Tracking**: [http://localhost:5000](http://localhost:5000)
- **Grafana Dashboards**: [http://localhost:3000](http://localhost:3000) (admin / admin)
- **Prometheus Explorer**: [http://localhost:9090](http://localhost:9090)
