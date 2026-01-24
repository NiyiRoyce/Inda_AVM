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

### GCP Setup & Authentication

#### 1. Configure Environment Variables

Copy the example environment file and update with your settings:

```bash
cp .env.example .env.local
# Edit .env.local with your GCP project details
```

Key variables to configure:

```bash
# GCP Authentication - Choose ONE:
# Option A: Service Account File (Recommended for local development)
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
SERVICE_ACCOUNT_JSON_PATH=/path/to/service_account.json

# GCP Project Settings
GCP_PROJECT_ID=your-gcp-project-id
GCP_REGION=us-central1

# BigQuery Configuration
BIGQUERY_DATASET=your_dataset_name
BIGQUERY_TRAIN_TABLE=your_training_table
GCS_BUCKET=your-gcs-bucket-name
```

#### 2. Set Up Service Account Credentials

**Option A: Service Account File (Recommended)**

1. Download your service account JSON from GCP Console:
   - Go to `APIs & Services > Service Accounts`
   - Select your service account
   - Click `Keys` tab
   - Create a new JSON key
   
2. Place the file in your project:
   ```bash
   # Copy to the default location (src/config/service_account.json)
   cp /downloads/my-service-account.json src/config/service_account.json
   
   # OR point to it via environment variable
   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service_account.json
   ```

**Option B: Application Default Credentials (ADC)**

```bash
# Authenticate via gcloud CLI (uses your user credentials)
gcloud auth application-default login

# This creates credentials in ~/.config/gcloud/application_default_credentials.json
# Libraries will automatically find and use them
```

**Option C: Google Colab**

For notebooks running in Colab:
```python
from src.config.notebook_setup import setup_notebook_environment
setup_notebook_environment(load_credentials=True)
# This will use Colab's built-in authentication
```

#### 3. Validate Configuration

```bash
# Test that environment is properly configured
python -c "from src.config.env import validate_environment; validate_environment(require_gcp_credentials=True); print('✓ Configuration valid!')"
```

### Using Notebooks

Each notebook includes environment setup. For example:

```python
# At the top of your notebook (jupyter/colab)
from src.config.notebook_setup import setup_notebook_environment, get_bigquery_client

# Initialize (loads .env, sets up GCP auth, validates config)
setup_notebook_environment(load_credentials=True)

# Create authenticated clients
client = get_bigquery_client()
df = client.query("SELECT * FROM `project.dataset.table`").to_dataframe()
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



