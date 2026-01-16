# AVM Project
# Automated Valuation Model (AVM) for Real Estate

Production-ready machine learning system for automated property valuation.

## 🏗️ Architecture

```
avm-project/
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt
├── setup.py
├── Makefile                      # train / test / deploy shortcuts
│
├── config/
│   ├── __init__.py
│   ├── settings.py               # GCP project, dataset, table names
│   ├── features.py               # Canonical feature list
│   ├── model_config.py           # Hyperparameters, CV, thresholds
│   └── env.py                    # Env-specific config (dev/stg/prod)
│
├── src/
│   ├── __init__.py
│   │
│   ├── auth/
│   │   ├── __init__.py
│   │   └── gcp_auth.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── bigquery_client.py
│   │   ├── loader.py
│   │   ├── validator.py
│   │   └── contracts/            # 🔐 Data contracts
│   │       ├── __init__.py
│   │       ├── raw_schema.py
│   │       ├── feature_schema.py
│   │       └── prediction_schema.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaners.py
│   │   ├── transformers.py
│   │   ├── imputers.py
│   │   └── validators.py         # Inference-safe checks
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── engineering.py
│   │   ├── selectors.py
│   │   └── spatial.py            # Amenities, geo, address features
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── linear.py
│   │   ├── residual.py
│   │   ├── ensemble.py
│   │   ├── trainer.py
│   │   └── registry.py           # Model + artifact registration
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── diagnostics.py
│   │   └── drift.py              # Feature & prediction drift
│   │
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── request_parser.py     # Vertex request normalization
│   │   ├── response_formatter.py
│   │   └── guards.py             # Fail-safe prediction logic
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── helpers.py
│       └── monitoring.py         # Stats → BigQuery / logs
│
├── pipelines/
│   ├── __init__.py
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│   └── validation_pipeline.py    # Schema + drift validation
│
├── deployment/
│   ├── predictor.py              # Vertex AI Predictor
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── deploy.sh
│   └── vertex_config.yaml        # Machine, autoscaling, traffic split
│
├── artifacts/
│   └── vYYYY_MM_DD/               # 🔖 Versioned artifacts
│       ├── models/
│       │   ├── linreg.pkl
│       │   ├── residual_lgbm.pkl
│       │   └── smearing.pkl
│       ├── preprocessors/
│       │   ├── imputer.pkl
│       │   └── scaler.pkl
│       └── metadata/
│           ├── feature_names.json
│           ├── feature_lineage.json
│           ├── training_stats.json
│           └── model_card.md      # Explainability + limitations
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_evaluation.ipynb
│   └── 04_error_analysis.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_data/
│   ├── test_preprocessing/
│   ├── test_features/
│   ├── test_models/
│   ├── test_serving/
│   └── test_pipeline/
│
└── scripts/
    ├── train.py
    ├── predict.py
    ├── validate_data.py           # Contract + drift checks
    ├── upload_to_gcs.py
    └── deploy_to_vertex.py

```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd avm-project

# Install dependencies
pip install -r requirements.txt
```

### Training

**From BigQuery:**
```bash
python scripts/train.py --project-id your-project-id
```

**From CSV:**
```bash
python scripts/train.py --csv data/properties.csv
```

**With Colab authentication:**
```bash
python scripts/train.py --use-colab
```

### Prediction

```bash
python scripts/predict.py --input data/new_properties.csv --output predictions.csv
```

### Upload to GCS

```bash
python scripts/upload_to_gcs.py --bucket your-bucket --prefix models
```

## 📊 Model Architecture

The system uses a two-stage ensemble approach:

1. **Linear Regression** with smearing correction for bias
2. **LightGBM Residual Model** to capture non-linear patterns

**Final Prediction = Linear Prediction + Residual Correction**

## 🔧 Configuration

Edit `config/settings.py` to customize:
- GCP project settings
- BigQuery dataset/table
- Geographic bounds
- Model hyperparameters

## 📈 Features

### Input Features
- Property configuration (beds, baths, toilets)
- Geographic coordinates
- Distance to amenities (schools, hospitals, malls, etc.)
- Address-based features

### Engineered Features
- Room totals and aggregates
- Consistency checks (list vs detail)
- Accessibility scores
- Log-transformed distances

## 🎯 Evaluation Metrics

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MRE**: Median Relative Error
- **MAPE**: Mean Absolute Percentage Error
- **Tier-based analysis**: Error breakdown by price tier

## 🐳 Deployment

### Build Container

```bash
cd deployment
docker build -t avm-predictor .
```

### Deploy to Vertex AI

```bash
# Upload models to GCS
python scripts/upload_to_gcs.py

# Deploy endpoint (use GCP Console or gcloud CLI)
```

## 📝 Environment Variables

Create `.env` file:

```bash
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
BIGQUERY_DATASET=your_dataset
BIGQUERY_TABLE=master_listings
GCS_BUCKET=your-bucket
LOG_LEVEL=INFO
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

## 📚 Documentation

- **Training Pipeline**: See `pipelines/train_pipeline.py`
- **Model Details**: See `src/models/`
- **Feature Engineering**: See `src/features/engineering.py`
- **API Documentation**: See `src/pipelines/inference_pipeline.py`



