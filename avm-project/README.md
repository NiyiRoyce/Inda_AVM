# AVM Project
# Automated Valuation Model (AVM) for Real Estate

Production-ready machine learning system for automated property valuation in Nigeria.

## 🏗️ Architecture

```
avm-project/
├── config/              # Configuration files
├── src/                 # Source code
│   ├── auth/           # GCP authentication
│   ├── data/           # Data loading and validation
│   ├── preprocessing/  # Data cleaning and transformation
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   ├── evaluation/     # Metrics and diagnostics
│   └── utils/          # Utilities
├── pipelines/          # Training and inference pipelines
├── deployment/         # Vertex AI deployment
├── scripts/            # CLI scripts
└── artifacts/          # Saved models
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
- **API Documentation**: See `deployment/predictor.py`



