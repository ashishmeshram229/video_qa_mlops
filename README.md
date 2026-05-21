# Real-Time Video Object Detection for Quality Assurance

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![Airflow](https://img.shields.io/badge/Apache_Airflow-2.8.1-017CEE.svg)](https://airflow.apache.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2.svg)](https://mlflow.org/)

## Project Overview
This project provides an end-to-end, fully automated MLOps pipeline for industrial quality inspection. Manual product quality checking on an assembly line is time-consuming, error-prone, and unscalable. This system automates the process using a state-of-the-art YOLOv8 deep learning model to instantly detect manufacturing defects (specifically focusing on the MVTec Anomaly Detection dataset).

The platform features a microservices architecture containerized with Docker, encompassing data engineering (Airflow), model registry (MLflow), an inference engine (FastAPI), a factory-floor user interface (Streamlit), and real-time observability (Prometheus & Grafana).

---

## Key Features
* **Automated Data Pipeline (ETL):** Apache Airflow DAGs handle data extraction, transformation, and validation, automatically triggering Data Version Control (DVC) for dataset versioning.
* **Dynamic Model Serving:** The FastAPI backend dynamically fetches the `best.pt` model weights tagged as "Staging" from the MLflow Model Registry at startup.
* **Batch In-Memory Processing:** The Streamlit UI allows factory workers to upload `.zip` archives of images. The backend extracts and processes these images entirely in-memory to prevent container disk bloat.
* **Automated Bounding Boxes:** The UI dynamically paints red bounding boxes and confidence scores over detected defects using Python's `PIL` library.
* **Real-Time Observability:** Prometheus scrapes the FastAPI `/metrics` endpoint every 5 seconds, visualized by custom Grafana dashboards tracking API hits, YOLO inference latency, and system load.

---

## Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | Apache Airflow | Master ETL, Data Quality, and DVC Pipeline (`data_pipeline_dag.py`). |
| **Data Versioning** | DVC | Tracks modifications to the large MVTec image dataset. |
| **Model Registry** | MLflow | Tracks model training experiments and stores the YOLO model artifacts. |
| **Inference API** | FastAPI | RESTful backend serving the YOLO model and exposing `/predict` and `/metrics`. |
| **Frontend UI** | Streamlit | Interface for factory workers to upload images and view segmentation results. |
| **Monitoring** | Prometheus & Grafana | Tracks API traffic, latency, and CPU load proxies in real-time. |
| **Infrastructure** | Docker Compose | Networks and spins up the entire multi-container architecture. |

---

## Repository Structure

```text
├── dags/
│   └── data_pipeline_dag.py         # Airflow ETL and DVC triggering
├── data/
│   ├── raw/bottle.tar.xz            # Raw MVTec compressed dataset
│   └── processed/                   # Transformed images and YOLO labels
├── deployments/
│   ├── docker/
│   │   ├── Dockerfile.api           # FastAPI backend Dockerfile
│   │   └── Dockerfile.ui            # Streamlit frontend Dockerfile
│   └── monitoring/
│       └── prometheus.yml           # Prometheus scraping configuration
├── src/
│   ├── api/
│   │   └── main.py                  # FastAPI inference engine & metrics
│   ├── pipeline/
│   │   └── data_engineering.py      # Data extraction and transformation logic
│   └── app.py                       # Streamlit User Interface
├── tests/
│   └── test_integration.py          # Pytest suite for API and MLflow connection
├── docker-compose.yml               # Complete infrastructure definition
└── run_demo.py                      # Load testing script for Grafana dashboards
Quick Start (Cold Start Runbook)
Follow these exact steps to wake up the infrastructure from scratch.

Phase 1: Environment Setup
Ensure your local environment has the required directories with the correct permissions for Airflow.

Bash
# Create required mount directories
mkdir -p dags logs data src plugins

# Set Airflow permissions
echo -e "AIRFLOW_UID=$(id -u)" > .env
Phase 2: Start the MLflow Vault
Because the FastAPI backend is hardcoded to fetch the AI model from MLflow on boot, the registry must be running first.

Bash
# Start MLflow in the background (leave this terminal open)
mlflow ui --host 0.0.0.0 --port 5000
Phase 3: Boot Up the Docker Factory
In a new terminal, initialize and start the core microservices.

Bash
# Initialize Airflow Database
docker compose up airflow-init

# Start the complete stack
docker compose up -d
Wait 30-60 seconds for all containers to initialize and report as healthy.

Service Access Ports
Once the stack is running, access the interfaces via your browser:

MLflow (Model Registry): http://localhost:5000

Airflow (Data Engineering): http://localhost:8081 (Login: admin / admin)

FastAPI (Swagger Docs): http://localhost:8000/docs

Streamlit (User Interface): http://localhost:8501

Grafana (System Monitoring): http://localhost:3000 (Login: admin / admin)

Live Demonstration & Monitoring
To fully test the system's capacity and view real-time metrics in Grafana:

Access the Streamlit UI and upload a .zip file of factory line images.

The UI will extract the ZIP in-memory, filter out hidden OS files (e.g., __MACOSX), and concurrently send the batch to the FastAPI backend.

Open the Grafana Dashboard ("MLOps Load & Performance Monitor").

Run the Load Test: To simulate heavy factory traffic (50+ concurrent cameras), run the provided simulation script:

Bash
pip install prometheus_client requests pillow
python run_demo.py
Watch the Grafana dashboard instantly populate with:

Total API Call Hits

Real-Time Traffic Spike & Alert Zones

YOLO Inference Latency (CPU Load Indicator)

Testing
The project includes an integration test suite that proves the system's structural integrity by booting the FastAPI app, connecting to MLflow, and loading the heavy YOLO PyTorch weights into memory before verifying the endpoints.

Bash
# Run the integration test suite
pytest
Built for MLOps and Industrial Automation.
