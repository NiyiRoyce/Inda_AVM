#!/bin/bash

# Deployment script for Vertex AI
# Usage: ./deploy.sh [project-id] [region]

set -e

# Configuration
PROJECT_ID=${1:-"primal-result-478707-k2"}
REGION=${2:-"us-central1"}
IMAGE_NAME="avm-predictor"
IMAGE_TAG="latest"
IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
MODEL_NAME="avm-real-estate"
ENDPOINT_NAME="avm-endpoint"

echo "=================================================="
echo "Deploying AVM to Vertex AI"
echo "=================================================="
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Image URI: ${IMAGE_URI}"
echo "=================================================="

# Step 1: Build Docker image
echo ""
echo "Step 1: Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

# Step 2: Tag for GCR
echo ""
echo "Step 2: Tagging image for GCR..."
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_URI}

# Step 3: Push to GCR
echo ""
echo "Step 3: Pushing image to GCR..."
docker push ${IMAGE_URI}

# Step 4: Upload model to Vertex AI
echo ""
echo "Step 4: Uploading model to Vertex AI..."
gcloud ai models upload \
  --region=${REGION} \
  --display-name=${MODEL_NAME} \
  --container-image-uri=${IMAGE_URI} \
  --project=${PROJECT_ID}

# Step 5: Create endpoint
echo ""
echo "Step 5: Creating endpoint..."
gcloud ai endpoints create \
  --region=${REGION} \
  --display-name=${ENDPOINT_NAME} \
  --project=${PROJECT_ID}

# Get endpoint ID
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=${REGION} \
  --filter="display_name=${ENDPOINT_NAME}" \
  --format="value(name)" \
  --project=${PROJECT_ID})

echo "Endpoint ID: ${ENDPOINT_ID}"

# Step 6: Deploy model to endpoint
echo ""
echo "Step 6: Deploying model to endpoint..."
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --model=${MODEL_NAME} \
  --display-name=${MODEL_NAME}-deployment \
  --machine-type=n1-standard-4 \
  --min-replica-count=1 \
  --max-replica-count=3 \
  --project=${PROJECT_ID}

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