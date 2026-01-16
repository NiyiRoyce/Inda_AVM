#!/bin/bash

# Deployment script for Vertex AI
# Usage: ./deploy.sh [project-id] [region]

set -euo pipefail

# ==================================================
# Configuration
# ==================================================
PROJECT_ID=${1:-"primal-result-478707-k2"}
REGION=${2:-"us-central1"}

IMAGE_NAME="avm-predictor"
IMAGE_TAG="latest"

# Artifact Registry (recommended)
REPO_NAME="avm-repo"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

MODEL_NAME="avm-real-estate"
ENDPOINT_NAME="avm-endpoint"

# Dedicated service account for Vertex AI
SERVICE_ACCOUNT="vertex-avm@${PROJECT_ID}.iam.gserviceaccount.com"

# ==================================================
# Output
# ==================================================
echo "=================================================="
echo "Deploying AVM to Vertex AI"
echo "Project ID : ${PROJECT_ID}"
echo "Region     : ${REGION}"
echo "Image URI  : ${IMAGE_URI}"
echo "Service SA : ${SERVICE_ACCOUNT}"
echo "=================================================="

# ==================================================
# Step 0: Enable required APIs
# ==================================================
echo ""
echo "Step 0: Enabling required APIs..."
gcloud services enable \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  --project=${PROJECT_ID}

# ==================================================
# Step 1: Ensure Artifact Registry repo exists
# ==================================================
echo ""
echo "Step 1: Ensuring Artifact Registry repository exists..."
gcloud artifacts repositories describe ${REPO_NAME} \
  --location=${REGION} \
  --project=${PROJECT_ID} >/dev/null 2>&1 || \
gcloud artifacts repositories create ${REPO_NAME} \
  --repository-format=docker \
  --location=${REGION} \
  --description="AVM Docker images" \
  --project=${PROJECT_ID}

# ==================================================
# Step 2: Build Docker image
# ==================================================
echo ""
echo "Step 2: Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# ==================================================
# Step 3: Tag image
# ==================================================
echo ""
echo "Step 3: Tagging image..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_URI}

# ==================================================
# Step 4: Authenticate Docker to Artifact Registry
# ==================================================
echo ""
echo "Step 4: Authenticating Docker..."
gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet

# ==================================================
# Step 5: Push image
# ==================================================
echo ""
echo "Step 5: Pushing image..."
docker push ${IMAGE_URI}

# ==================================================
# Step 6: Upload model to Vertex AI
# ==================================================
echo ""
echo "Step 6: Uploading model to Vertex AI..."
MODEL_ID=$(gcloud ai models upload \
  --region=${REGION} \
  --display-name=${MODEL_NAME} \
  --container-image-uri=${IMAGE_URI} \
  --project=${PROJECT_ID} \
  --format="value(name)")

echo "Model ID: ${MODEL_ID}"

# ==================================================
# Step 7: Create endpoint if not exists
# ==================================================
echo ""
echo "Step 7: Creating endpoint (if needed)..."
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=${REGION} \
  --filter="display_name=${ENDPOINT_NAME}" \
  --format="value(name)" \
  --project=${PROJECT_ID})

if [[ -z "${ENDPOINT_ID}" ]]; then
  ENDPOINT_ID=$(gcloud ai endpoints create \
    --region=${REGION} \
    --display-name=${ENDPOINT_NAME} \
    --project=${PROJECT_ID} \
    --format="value(name)")
fi

echo "Endpoint ID: ${ENDPOINT_ID}"

# ==================================================
# Step 8: Deploy model to endpoint
# ==================================================
echo ""
echo "Step 8: Deploying model to endpoint..."
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --model=${MODEL_ID} \
  --display-name=${MODEL_NAME}-deployment \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=3 \
  --service-account=${SERVICE_ACCOUNT} \
  --project=${PROJECT_ID}

# ==================================================
# Done
# ==================================================
echo ""
echo "=================================================="
echo "✅ Deployment completed successfully!"
echo "=================================================="
echo "Endpoint ID: ${ENDPOINT_ID}"
echo "Region: ${REGION}"
echo ""
echo "Test with:"
echo "gcloud ai endpoints predict ${ENDPOINT_ID} \\"
echo "  --region=${REGION} \\"
echo "  --json-request=test_request.json"
echo "=================================================="
