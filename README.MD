## Telco Churn – End-to-End ML Project
### Purpose

Build and ship a full machine-learning solution for predicting customer churn in a telecom setting—from data prep and modeling to an API + web UI.

### Problem solved & benefits

- Faster decisions: Predicts which customers are likely to churn so teams can act before they leave.
- Operationalized ML: Model is accessible via a REST API and a simple UI; anyone can test it without notebooks.
- Traceable experiments: MLflow tracks runs, metrics, and artifacts for reproducibility and auditing.

### What I built

- Data & Modeling: Feature engineering + XGBoost classifier; experiments logged to MLflow.
- Model tracking: Runs, metrics, and the serialized model logged under a named MLflow experiment.
- Inference service: FastAPI app exposing /predict (POST) and a root health check /.
- Web UI: Gradio interface mounted at /ui for quick, shareable manual testing.
- Containerization: Docker image with uvicorn entrypoint (src.app.main:app) listening on port 8000.

### Roadblocks & how we solved them

Module import error in container (ModuleNotFoundError: serving)

- Cause: Python path in the image didn’t include src/.
- Fixes: Set PYTHONPATH=/app/src in the Dockerfile; corrected uvicorn app path to src.app.main:app.

Unhealthy targets behind ALB

- Cause: App didn’t respond at the health-check path; listener/target port mismatches.
- Fixes: Added GET / health endpoint; confirmed ALB listener on 80 forwards to TG on 8000; TG health check path set to /.

ALB DNS timing out

- Cause: Security group rules not aligned with traffic flow.
- Fixes: ALB SG allows inbound 80 from 0.0.0.0/0; task SG allows inbound 8000 from the ALB SG; outbound open.

Local testing vs. prod paths

- Cause: MLflow artifact URIs differ locally vs. in container.
- Fixes: For local dev, load via direct ./mlruns/.../artifacts/model; in prod, container loads the packaged model path used at build time.