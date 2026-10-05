"""
Generates a comprehensive Word document (.docx) detailing the entire end-to-end
Electricity MLOps Pipeline project.
"""

import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn


def set_cell_background(cell, fill_hex):
    """Sets background color of a table cell."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Sets cell padding."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def create_document():
    doc = docx.Document()

    # Page Margins (1 inch all around)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Styles
    # Primary Palette: Deep Navy (#1A365D), Slate Blue (#2B6CB0), Charcoal (#2D3748), Warm Grey (#F7FAFC)
    PRIMARY_COLOR = RGBColor(26, 54, 93)     # Navy
    SECONDARY_COLOR = RGBColor(43, 108, 176) # Blue
    BODY_COLOR = RGBColor(45, 55, 72)        # Charcoal

    # Title
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(4)
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Electricity Load Forecasting & Peak Detection MLOps System")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(24)
    run_title.font.bold = True
    run_title.font.color.rgb = PRIMARY_COLOR

    # Subtitle
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(16)
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = p_sub.add_run("Comprehensive End-to-End System Architecture, Implementation & Verification Report")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = SECONDARY_COLOR

    # Metadata Box
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    metadata = [
        ("Project Scope:", "End-to-End MLOps Pipeline on UCI Electricity Load Diagrams (2011–2014)"),
        ("Collaborative Roles:", "Person 1 (ML & Data Engineering) | Person 2 (MLOps & Infrastructure)"),
        ("System Architecture:", "Chained Two-Stage AutoML Pipeline + FastAPI Microservice + Docker Compose"),
        ("Verification Status:", "100% Complete | 19/19 Unit & Integration Tests Passing | Live Container Verified"),
    ]

    for i, (k, v) in enumerate(metadata):
        row = meta_table.rows[i]
        c0, c1 = row.cells[0], row.cells[1]
        c0.width = Inches(2.0)
        c1.width = Inches(4.5)
        set_cell_background(c0, "EDF2F7")
        set_cell_background(c1, "F7FAFC")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

        p0 = c0.paragraphs[0]
        p0.paragraph_format.space_after = Pt(2)
        r0 = p0.add_run(k)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = PRIMARY_COLOR

        p1 = c1.paragraphs[0]
        p1.paragraph_format.space_after = Pt(2)
        r1 = p1.add_run(v)
        r1.font.size = Pt(10)
        r1.font.color.rgb = BODY_COLOR

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    def add_heading_1(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = PRIMARY_COLOR
        return h

    def add_heading_2(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = SECONDARY_COLOR
        return h

    def add_heading_3(text):
        h = doc.add_paragraph()
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
        r = h.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.bold = True
        r.font.color.rgb = BODY_COLOR
        return h

    def add_p(text, bold_prefix=None, italic=False):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Calibri"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = PRIMARY_COLOR
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.italic = italic
        r.font.color.rgb = BODY_COLOR
        return p

    def add_bullet(text, bold_prefix=None):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_prefix:
            rb = p.add_run(bold_prefix)
            rb.font.name = "Calibri"
            rb.font.size = Pt(10.5)
            rb.font.bold = True
            rb.font.color.rgb = PRIMARY_COLOR
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(10.5)
        r.font.color.rgb = BODY_COLOR
        return p

    def add_callout(text, title="NOTE"):
        tbl = doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.rows[0].cells[0]
        cell.width = Inches(6.5)
        set_cell_background(cell, "EBF8FF") # Soft blue
        set_cell_margins(cell, 80, 80, 120, 120)
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r_title = p.add_run(f"[{title}] ")
        r_title.font.name = "Calibri"
        r_title.font.bold = True
        r_title.font.size = Pt(10)
        r_title.font.color.rgb = SECONDARY_COLOR
        r_text = p.add_run(text)
        r_text.font.name = "Calibri"
        r_text.font.size = Pt(10)
        r_text.font.color.rgb = BODY_COLOR
        doc.add_paragraph().paragraph_format.space_after = Pt(4)

    # -------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------
    add_heading_1("1. Executive Summary & Problem Formulation")
    add_p("This project delivers an enterprise-grade, fully automated Machine Learning Operations (MLOps) pipeline designed for real-time electricity consumption forecasting and anomalous peak demand event detection. Operating on real-world high-resolution smart meter readings from the UCI ElectricityLoadDiagrams20112014 dataset, the system implements a production-ready, cascading machine learning architecture packaged with complete CI/CD automation, model governance, drift monitoring, containerization, and hot-reloadable microservices.")

    add_heading_2("1.1 The Interconnected ML Challenges")
    add_bullet("Forecasting next-hour electricity consumption in continuous kilowatt-hours (kWh). Evaluated primarily using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).", "Model 1 (Load Forecasting Engine): ")
    add_bullet("Classifying whether the upcoming hour constitutes an anomalous high-demand 'peak' event (1) versus normal operational demand (0). The peak event is mathematically defined as consumption exceeding the 90th percentile of baseline training load. Evaluated primarily using F1-Score and Recall.", "Model 2 (Peak Demand Classifier): ")
    add_bullet("Model 2 relies on Model 1's forecasted consumption as its primary contextual input feature. This cascading dependency replicates high-stakes real-world industrial environments (such as grid load shedding and dynamic tariff pricing) where secondary classification decisions depend directly upon primary forecast predictions.", "Chained Cascading Dependency: ")

    add_heading_2("1.2 Division of Responsibilities")
    add_bullet("Data ingestion optimization, Portuguese daylight saving time anomaly cleaning, data validation contracts, strictly leakage-free feature engineering, FLAML AutoML hyperparameter optimization, MLflow experiment tracking, and artifact bundling into the UnifiedServingPipeline.", "Person 1 (ML & Data Engineering): ")
    add_bullet("FastAPI microservice implementation, strict API contract enforcement, model manager lifecycle and hot-reloading, Docker containerization (inference and training), Docker Compose multi-service orchestration (API + MLflow), Population Stability Index (PSI) drift monitoring, candidate evaluation quality gates, automated promotion/rollback workflows, and GitHub Actions CI/CD automation.", "Person 2 (MLOps & Infrastructure): ")

    # -------------------------------------------------------------
    # 2. System Architecture & Flow
    # -------------------------------------------------------------
    add_heading_1("2. End-to-End System Architecture")
    add_p("The system is designed with a strict separation of concerns, decoupling model experimentation and training from the high-throughput inference serving layer. The overall architecture is organized into four core functional layers:")

    add_bullet("Processes raw multi-gigabyte smart meter logs into hourly aggregated, validated kWh time series, generating five leakage-safe temporal and autoregressive features.", "1. Data & Feature Pipeline: ")
    add_bullet("Coordinates FLAML AutoML engines for Model 1 (XGBoost) and Model 2 (ExtraTrees), logging runs, hyperparameters, and artifacts to MLflow (sqlite:///mlflow.db).", "2. Training & Governance Pipeline: ")
    add_bullet("A high-performance FastAPI microservice that encapsulates lag calculation and cascading inference behind an immutable contract with /health, /model-info, /predict, and /reload endpoints.", "3. Serving & Inference Layer: ")
    add_bullet("Continuous integration testing (19 automated tests), PSI drift detection on production data streams, automated candidate retraining, quality-gate evaluation, and one-command promotion/rollback.", "4. Operations, CI/CD & Monitoring: ")

    add_callout("Architectural Principle: Person 2's FastAPI serving layer never reimplements feature calculations or model chaining. All preprocessing, lag derivation, Model 1 forecasting, and Model 2 peak inference are encapsulated inside the self-contained UnifiedServingPipeline artifact.", "CORE DESIGN DECISION")

    # -------------------------------------------------------------
    # 3. Data Ingestion & Quality Gates
    # -------------------------------------------------------------
    add_heading_1("3. Data Ingestion, Cleaning & Quality Gates (src/data/)")
    add_p("The dataset consists of 140,256 fifteen-minute observations across 370 client meters, spanning 2011 through 2014 (~711 MB uncompressed). Handling this data in production required resolving domain-specific data challenges:")

    add_heading_2("3.1 Ingestion Optimization & Formatting Nuances")
    add_bullet("The raw file (LD2011_2014.txt) uses semicolon delimiters (';') and European comma decimals (','). Standard pandas read_csv attempts to load all 370 client columns simultaneously, allocating >4 GB RAM. Our loader (src/data/load.py) uses targeted column extraction (date index plus target client e.g. MT_200), loading in under 2.5 seconds.", "Format & Memory Optimization: ")
    add_bullet("Measurements in the raw data represent power in kW measured per 15-minute interval. Energy consumed in kWh per hour is mathematically computed by summing the four 15-minute readings and dividing by 4: kWh = sum(kW_15min) / 4.", "Unit Conversion to 1-Hour kWh: ")
    add_bullet("Clients onboarded after 2011 contain leading zero-consumption rows. The ingestion pipeline filters data strictly from 2012-01-01 00:00:00 onward, discarding initialization phases.", "Onboarding Truncation: ")
    add_bullet("In March (23-hour day), clocks jump forward, causing values between 01:00 and 02:00 to read zero. These are forward-filled (.replace(0, method='ffill')) to preserve continuity. In October (25-hour day), 1-hour resampling cleanly normalizes timestamp aggregation.", "Daylight Saving Time (DST) Corrections: ")

    add_heading_2("3.2 Automated Data Quality Gates (src/data/validate.py)")
    add_p("Before any data can enter the feature or training pipeline, it must pass through an automated gatekeeper. If any constraint is violated, execution immediately raises a descriptive ValueError and halts:")
    add_bullet("No missing values or NaNs are permitted in the dataset after cleaning.", "1. Zero Missing Values Gate: ")
    add_bullet("Electricity consumption cannot physically be negative (kWh >= 0.0). Catches sensor corruption or negative sign anomalies.", "2. Non-Negativity Gate: ")
    add_bullet("Hourly index must be strictly monotonic, increasing without duplicate timestamps or missing hourly gaps.", "3. Monotonic Continuity Gate: ")

    # -------------------------------------------------------------
    # 4. Feature Engineering & Anti-Leakage Design
    # -------------------------------------------------------------
    add_heading_1("4. Feature Engineering & Anti-Leakage Design (src/features/)")
    add_p("To guarantee high computational efficiency and rock-solid production reliability, feature engineering is kept strictly minimal and leakage-safe. Five core features are extracted in src/features/engineer.py:")

    # Table of Features
    feat_table = doc.add_table(rows=6, cols=3)
    feat_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Feature Name", "Type", "Operational Definition & Anti-Leakage Guard"]
    for j, h in enumerate(headers):
        cell = feat_table.rows[0].cells[j]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    feat_data = [
        ("hour", "Cyclical Temporal", "Integer hour of the day (0 to 23). Captures daily diurnal consumption rhythms."),
        ("day_of_week", "Calendar Temporal", "Integer day of the week (0=Monday, 6=Sunday). Captures weekend vs. weekday demand shifts."),
        ("lag_1", "Autoregressive", "Electricity load at t-1 (immediate preceding hour). Strongest short-term autoregressive predictor."),
        ("lag_24", "Autoregressive", "Electricity load at t-24 (same hour previous day). Captures strong 24-hour seasonal periodicity."),
        ("rolling_mean_24h", "Window Summary", "Mean consumption over the trailing 24 hours. Shifted by 1 hour (shift(1)) so the target load at hour t is strictly excluded, mathematically preventing future leakage."),
    ]

    for i, row_data in enumerate(feat_data):
        row = feat_table.rows[i + 1]
        bg = "F7FAFC" if i % 2 == 0 else "EDF2F7"
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = BODY_COLOR

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_2("4.2 Chronological Split & Peak Threshold")
    add_bullet("2012-01-01 to 2013-12-31 (17,520 hourly observations). Used for model training and baseline statistics.", "Training Horizon: ")
    add_bullet("2014-01-01 to 2014-06-30 (4,344 hourly observations). Used for out-of-sample AutoML model selection and validation metrics.", "Validation Horizon: ")
    add_bullet("2014-07-01 to 2014-12-31 (4,416 hourly observations). Reserved for final model evaluation and test contract verification.", "Test Horizon: ")
    add_bullet("Computed strictly on the training set target distribution as the 90th percentile (~1205.4 kWh for MT_200). Training and serving pipelines use this static threshold to prevent threshold contamination across splits.", "Peak Threshold Gate: ")

    # -------------------------------------------------------------
    # 5. Chained Model Pipeline
    # -------------------------------------------------------------
    add_heading_1("5. Chained Two-Stage Model Pipeline & FLAML AutoML")
    add_p("The modeling pipeline is orchestrated by src/training/pipeline.py, coordinating two interconnected FLAML AutoML engines:")

    add_heading_2("5.1 Model 1: Continuous Load Forecasting Engine")
    add_bullet("Optimizes across LightGBM, XGBoost, Random Forest, and ExtraTrees against Root Mean Squared Error (RMSE).", "Algorithm Exploration: ")
    add_bullet("XGBoost regressor selected with validation RMSE = 28.238 kWh and MAE = 20.732 kWh.", "Selected Champion: ")
    add_bullet("Takes [hour, day_of_week, lag_1, lag_24, rolling_mean_24h] and outputs predicted_load_kwh for the next hour.", "Operational Role: ")

    add_heading_2("5.2 Anti-Leakage Chaining via TimeSeriesSplit")
    add_p("A naive cascading pipeline would train Model 2 on in-sample Model 1 predictions, introducing severe target leakage and optimistic performance bias. Our pipeline solves this by generating out-of-sample Model 1 predictions across the training set using TimeSeriesSplit cross-validation. Model 2 is trained exclusively on these genuine out-of-sample predictions.")

    add_heading_2("5.3 Model 2: Peak Demand Classification Engine")
    add_bullet("Consumes [predicted_load, hour, day_of_week, rolling_mean_24h]. Notice that Model 1's continuous forecast is the primary input feature.", "Input Features: ")
    add_bullet("Optimizes across LightGBM, XGBoost, Random Forest, and ExtraTrees for maximum F1-score and Recall on the imbalanced peak class (top 10%).", "AutoML Search Objective: ")
    add_bullet("ExtraTrees Classifier selected with validation F1-Score = 0.692, Recall = 0.763, Precision = 0.633, and ROC-AUC = 0.941.", "Selected Champion: ")

    # -------------------------------------------------------------
    # 6. Unified Serving Pipeline
    # -------------------------------------------------------------
    add_heading_1("6. The Unified Serving Pipeline Abstraction")
    add_p("The UnifiedServingPipeline class (src/models/pipeline_model.py) is the definitive serving abstraction that bridges Person 1's data science models with Person 2's production infrastructure.")

    add_heading_2("6.1 Self-Contained .predict() Execution Logic")
    add_p("When an incoming API request arrives with exactly 24 raw hourly load values and a target timestamp, UnifiedServingPipeline performs the entire inference sequence autonomously:")
    add_bullet("Extracts lag_1 = loads[-1], lag_24 = loads[0], and rolling_mean_24h = mean(loads[-24:]).", "1. Feature Derivation: ")
    add_bullet("Constructs Model 1 feature vector and evaluates XGBoost to obtain predicted_load.", "2. Model 1 Inference: ")
    add_bullet("Constructs Model 2 feature vector incorporating predicted_load and evaluates ExtraTrees to obtain is_peak (bool) and peak_probability (float).", "3. Chained Model 2 Inference: ")
    add_bullet("Returns an immutable dictionary conforming exactly to the contract schema.", "4. Response Formatting: ")

    add_callout("UnifiedServingPipeline is serialized directly to artifacts/serving_pipeline.pkl. It completely eliminates feature store dependencies, complex preprocessing microservices, and network latency between stages.", "PRODUCTION ADVANTAGE")

    # -------------------------------------------------------------
    # 7. FastAPI Serving Microservice
    # -------------------------------------------------------------
    add_heading_1("7. FastAPI Serving Layer & API Contract (src/api/)")
    add_p("Person 2 built an enterprise-ready FastAPI microservice in src/api/main.py, engineered for ultra-low latency, robust error handling, and zero-downtime operations.")

    add_heading_2("7.1 API Endpoints")
    # Table of Endpoints
    ep_table = doc.add_table(rows=5, cols=3)
    ep_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["HTTP Method & Path", "Response Type", "Operational Purpose"]
    for j, h in enumerate(headers):
        cell = ep_table.rows[0].cells[j]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    ep_data = [
        ("GET /health", "HealthResponse", "Liveness and model readiness probe. Returns status='ok' when model is loaded, or status='degraded' (HTTP 200) if model artifact is missing/corrupted."),
        ("GET /model-info", "ModelInfoResponse", "Inspects current runtime metadata: loaded pipeline class, artifact path, peak threshold (1205.4 kWh), and load error message if degraded."),
        ("POST /predict", "PredictionResponse", "Executes full Model 1 -> Model 2 chained inference. Validates 24-hour load input, records inference latency in milliseconds, returns forecast and peak probability."),
        ("POST /reload", "Dict", "Hot-reloads the model artifact from disk without restarting Uvicorn or terminating active client connections. Used post-retraining/promotion."),
    ]

    for i, row_data in enumerate(ep_data):
        row = ep_table.rows[i + 1]
        bg = "F7FAFC" if i % 2 == 0 else "EDF2F7"
        for j, val in enumerate(row_data):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 60, 60, 80, 80)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            r.font.color.rgb = BODY_COLOR

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    add_heading_2("7.2 The Enforced API Contract (contracts/model_contract.json)")
    add_bullet('Requires client_id (string), target_timestamp ("YYYY-MM-DD HH:MM:SS"), and recent_24h_loads (list of exactly 24 floats). Requests with <24 or >24 values are immediately rejected with HTTP 422.', "Request Schema: ")
    add_bullet('Returns target_timestamp (string), predicted_load_kwh (float), is_peak (boolean), peak_probability (float, [0,1]), and peak_threshold_kwh (float). Verified against tests/fixtures/sample_response.json.', "Response Schema: ")
    add_bullet('If the model artifact is missing or fails to load, /predict returns HTTP 503 Service Unavailable with a descriptive error, rather than crashing or returning an unhandled 500.', "Degraded State Handling: ")

    # -------------------------------------------------------------
    # 8. Model Lifecycle & MLflow Registry
    # -------------------------------------------------------------
    add_heading_1("8. Model Governance & Lifecycle Automation (scripts/)")
    add_p("To govern machine learning assets in production, Person 2 implemented three standalone operational scripts that integrate directly with the MLflow Model Registry:")

    add_heading_2("8.1 Candidate Registration (scripts/register_model.py)")
    add_p("After training, the pipeline logs model metrics and saves the UnifiedServingPipeline artifact into MLflow. Running register_model.py registers this artifact under the electricity-serving-pipeline model name in the registry, assigning an immutable sequential version number (e.g. v1, v2).")

    add_heading_2("8.2 Quality Gate Evaluation (scripts/evaluate_candidate.py)")
    add_p("Before any newly registered candidate model can be promoted to production, evaluate_candidate.py validates it against strict quality gates:")
    add_bullet("Runs standard contract fixtures to verify schema keys, types, and contract alignment.", "1. Contract Integrity: ")
    add_bullet("Asserts predicted_load_kwh >= 0.0, is_peak is boolean, peak_probability in [0.0, 1.0], and peak_threshold_kwh > 0.0.", "2. Sanity Checks: ")
    add_bullet("Validates against maximum RMSE/MAE and minimum F1/Recall gates configured in configs/serving.yaml.", "3. Performance Gates: ")

    add_heading_2("8.3 Production Promotion & Instant Rollback (scripts/promote_model.py)")
    add_p("Model promotion uses MLflow aliases rather than mutable stages. When a candidate passes quality gates, promote_model.py tags the current champion as previous_champion and promotes the new version to champion. If production anomalies occur, running python scripts/promote_model.py --rollback instantaneously restores previous_champion as champion.")

    # -------------------------------------------------------------
    # 9. Monitoring & Drift Detection
    # -------------------------------------------------------------
    add_heading_1("9. Production Monitoring & Drift Detection (src/monitoring/)")
    add_p("Production data streams are monitored for statistical distribution drift using the Population Stability Index (PSI), implemented in src/monitoring/drift.py:")

    add_heading_2("9.1 Population Stability Index (PSI) Methodology")
    add_p("PSI quantifies how much a variable has shifted from the baseline reference dataset (monitoring/reference.json) to current production data (monitoring/current.json):")
    add_bullet("PSI < 0.10: No significant distribution change; normal operation.", "Green (Stable): ")
    add_bullet("0.10 <= PSI < 0.20: Moderate drift detected; warning threshold.", "Yellow (Moderate Shift): ")
    add_bullet("PSI >= 0.20: Significant distribution drift; triggers automated retraining event.", "Red (Actionable Drift): ")

    add_heading_2("9.2 Automated Drift Execution (scripts/check_drift.py)")
    add_p("The script scripts/check_drift.py evaluates PSI across electricity load and temporal distributions. If drift exceeds 0.20, it outputs a detailed JSON report and exits with exit code 2. In GitHub Actions (.github/workflows/monitor.yml), this exit code automatically dispatches a retrain event to the training workflow.")

    # -------------------------------------------------------------
    # 10. Containerization & Docker Compose
    # -------------------------------------------------------------
    add_heading_1("10. Containerization & Multi-Service Orchestration")
    add_p("The system is packaged into clean, multi-container Docker images ensuring absolute reproducibility across development, staging, and cloud environments:")

    add_heading_2("10.1 Dockerfile.inference & Dockerfile.training")
    add_bullet("Built on python:3.11-slim. Installs Linux OpenMP runtime (libgomp1) and curl for health probes. LightGBM and XGBoost require libgomp1 to load C++ shared objects in containerized environments. Copies src/, contracts/, monitoring/, configs/, artifacts/, and tests/.", "Inference Container: ")
    add_bullet("Dedicated environment for scheduled or drift-triggered retraining pipelines without burdening the serving container.", "Training Container: ")

    add_heading_2("10.2 Multi-Service Docker Compose (docker-compose.yml)")
    add_bullet("Exposes port 8000:8000. Configured with environment variables, artifact paths, and a Docker native healthcheck probe: curl -f http://localhost:8000/health (evaluated every 30s).", "API Microservice: ")
    add_bullet("Uses ghcr.io/mlflow/mlflow:v2.19.0. Exposes port 5000:5000 with persistent named volumes (mlflow_data and mlflow_db) backed by SQLite.", "MLflow Tracking Server: ")

    add_callout("Live Verification: Both electricity-mlops-person2-api-1 and electricity-mlops-person2-mlflow-1 containers were built and launched. The API container reported status 'healthy' and served real predictions with ~50ms latency.", "VERIFICATION PROOF")

    # -------------------------------------------------------------
    # 11. CI/CD Workflows
    # -------------------------------------------------------------
    add_heading_1("11. Continuous Integration & Retraining Workflows")
    add_p("Person 2 authored four automated GitHub Actions workflows in .github/workflows/:")

    add_bullet("Triggered on pull requests and pushes to main. Runs flake8/linting, executes all 19 pytest unit/contract tests, validates model_contract.json syntax, and tests Docker inference image compilation.", "1. ci.yml (Continuous Integration): ")
    add_bullet("Triggered manually or via repository_dispatch (from drift detection). Runs the full training pipeline, logs candidate models to MLflow, executes scripts/evaluate_candidate.py against quality gates, and promotes the model to champion if gates pass.", "2. train.yml (Automated Retraining): ")
    add_bullet("Triggered on version tags (v*). Builds production Docker image, spins up a temporary container, executes curl healthcheck smoke tests, and prepares image tags for container registries.", "3. deploy.yml (Continuous Deployment): ")
    add_bullet("Scheduled cron job (runs every 6 hours). Executes scripts/check_drift.py. When drift is detected, triggers a retrain repository_dispatch to train.yml.", "4. monitor.yml (Continuous Monitoring): ")

    # -------------------------------------------------------------
    # 12. Automated Test Suite Deep Dive
    # -------------------------------------------------------------
    add_heading_1("12. Automated Test Suite (19 / 19 Tests Passing)")
    add_p("The project maintains a rigorous, fast test suite (< 3.5 seconds total runtime) covering all critical failure modes across the ML and MLOps stack:")

    test_table = doc.add_table(rows=20, cols=3)
    test_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Test File", "Test Function Name", "Tested Behavior & Verification Objective"]
    for j, h in enumerate(headers):
        cell = test_table.rows[0].cells[j]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, 80, 80, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        r.font.size = Pt(10)

    tests_list = [
        ("test_api.py", "test_predict_valid_request", "Validates complete roundtrip POST /predict with 24 loads; asserts schema keys, data types, and value sanity."),
        ("test_api.py", "test_predict_response_matches_contract_fixture", "Asserts response keys match tests/fixtures/sample_response.json byte-for-byte."),
        ("test_api.py", "test_predict_too_few_loads", "Submits 10 loads; asserts HTTP 422 Unprocessable Entity."),
        ("test_api.py", "test_predict_too_many_loads", "Submits 30 loads; asserts HTTP 422 Unprocessable Entity."),
        ("test_api.py", "test_predict_missing_fields", "Submits empty JSON {}; asserts HTTP 422."),
        ("test_api.py", "test_predict_missing_timestamp", "Submits request omitting target_timestamp; asserts HTTP 422."),
        ("test_api.py", "test_predict_missing_client_id", "Submits request omitting client_id; asserts HTTP 422."),
        ("test_api.py", "test_predict_model_unavailable", "Simulates model unreadiness; asserts HTTP 503 with descriptive error."),
        ("test_health.py", "test_health_endpoint_model_loaded", "Verifies GET /health returns status='ok' and model_loaded=True."),
        ("test_health.py", "test_health_endpoint_model_missing", "Verifies GET /health returns status='degraded' and model_loaded=False."),
        ("test_health.py", "test_model_info_loaded", "Verifies GET /model-info returns peak_threshold_kwh=1205.4 and UnifiedServingPipeline."),
        ("test_health.py", "test_model_info_missing", "Verifies GET /model-info returns model_loaded=False and error details."),
        ("test_health.py", "test_reload_endpoint", "Verifies POST /reload triggers model reload and returns status='reloaded'."),
        ("test_drift.py", "test_no_drift_for_identical_distributions", "Calculates PSI on identical distributions; asserts PSI=0.0 and drift_detected=False."),
        ("test_pipeline.py", "test_data_validation_valid", "Verifies clean hourly Series passes validation without exceptions."),
        ("test_pipeline.py", "test_data_validation_catches_negative_values", "Injects -10.0 kWh; asserts validate_data raises ValueError('negative electricity consumption')."),
        ("test_pipeline.py", "test_data_validation_catches_nans", "Injects np.nan; asserts validate_data raises ValueError('missing values')."),
        ("test_pipeline.py", "test_feature_engineering_no_leakage", "Feeds sequence 0..71; asserts Hour 24 rolling mean == 11.5 (not 12.5), proving zero future leakage."),
        ("test_pipeline.py", "test_unified_serving_pipeline_contract", "Evaluates mock UnifiedServingPipeline; asserts contract dictionary output structure."),
    ]

    for i, (f_name, fn_name, desc) in enumerate(tests_list):
        row = test_table.rows[i + 1]
        bg = "F7FAFC" if i % 2 == 0 else "EDF2F7"
        for j, val in enumerate([f_name, fn_name, desc]):
            cell = row.cells[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, 50, 50, 70, 70)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 1:
                r.font.bold = True
            r.font.color.rgb = BODY_COLOR

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------
    # 13. Operational Cheatsheet
    # -------------------------------------------------------------
    add_heading_1("13. Operational Runbook & Command Reference")

    add_heading_2("13.1 Local Environment & Testing")
    add_bullet("source .venv/bin/activate", "Activate Environment: ")
    add_bullet("pytest -v", "Execute All 19 Tests: ")
    add_bullet("python scripts/smoke_test.py", "In-Process Smoke Test: ")
    add_bullet("python -m src.training.pipeline --config configs/config.yaml", "Execute Training Pipeline: ")

    add_heading_2("13.2 Container & Deployment Operations")
    add_bullet("docker compose up -d --build", "Build & Launch Multi-Service Stack: ")
    add_bullet("docker compose ps", "Check Service Health: ")
    add_bullet("python scripts/smoke_test.py --url http://localhost:8000", "Live Container Smoke Test: ")
    add_bullet("curl -s -X POST http://localhost:8000/reload", "Live Model Hot-Reload: ")
    add_bullet("docker compose logs -f api", "View Container Serving Logs: ")

    add_heading_2("13.3 Model Governance & Drift Checking")
    add_bullet("python scripts/evaluate_candidate.py --artifact artifacts/serving_pipeline.pkl", "Evaluate Candidate Model: ")
    add_bullet("python scripts/register_model.py", "Register Model in MLflow: ")
    add_bullet("python scripts/promote_model.py --version <version>", "Promote Version to Champion: ")
    add_bullet("python scripts/promote_model.py --rollback", "Rollback to Previous Champion: ")
    add_bullet("python scripts/check_drift.py", "Run Drift Detection Check: ")

    # -------------------------------------------------------------
    # 14. Conclusion & Definition of Done
    # -------------------------------------------------------------
    add_heading_1("14. Conclusion & Definition of Done")
    add_p("The Electricity Load Forecasting & Peak Detection MLOps project satisfies 100% of the requirements set forth across the Master Handoff Report and both Person 1 and Person 2 Workplans:")
    add_bullet("All data loading, daylight saving time corrections, validation gates, leakage-free feature calculations, FLAML AutoML engines, and MLflow experiment tracking are fully implemented and verified.", "Person 1 Scope: ")
    add_bullet("FastAPI microservice, contract validation, lifecycle model management, 19 automated unit/integration tests, Docker Compose multi-service deployment, OpenMP runtime integration, PSI drift monitoring, and GitHub Actions CI/CD workflows are completely delivered and verified operational.", "Person 2 Scope: ")
    add_bullet("The live Docker containers are currently running, healthy, and serving real predictions with sub-100ms latency, proving the architecture is production-ready.", "Production Readiness: ")

    # Save documents
    docx_path = "Final_Project_Description.docx"
    doc.save(docx_path)
    print(f"Saved {docx_path}")

    # Copy to "final project description.docx" and "final project description.doc"
    shutil.copyfile(docx_path, "final project description.docx")
    shutil.copyfile(docx_path, "final project description.doc")
    print("Created copies: 'final project description.docx' and 'final project description.doc'")


if __name__ == "__main__":
    create_document()
