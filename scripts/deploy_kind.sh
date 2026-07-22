#!/usr/bin/env bash
# Full local MLOps pipeline + Kind deployment script.
# Make sure you have python, docker, kind, and kubectl installed.
set -e

IMAGE="loan-approval-mlops:latest"
CLUSTER="${KIND_CLUSTER_NAME:-loan-cluster}"
BUILD_ID="${BUILD_ID:-local-$(date +%Y%m%d%H%M%S)}"
GITHUB_SHA="${GITHUB_SHA:-$(git rev-parse --short HEAD 2>/dev/null || echo local)}"

echo "==> Running data validation..."
python -m src.pipeline.validate_data

echo "==> Cleaning data..."
python -m src.pipeline.clean_data

echo "==> Training models and logging MLflow metrics for build ${BUILD_ID}..."
BUILD_ID="${BUILD_ID}" GITHUB_SHA="${GITHUB_SHA}" MLFLOW_TRACKING_URI="mlruns" \
  python -m src.pipeline.train_pipeline

echo "==> Running performance gate..."
python -m src.pipeline.performance_gate

echo "==> Building Docker image..."
docker build -t "$IMAGE" .

echo "==> Creating Kind cluster (skips if already exists)..."
kind create cluster --name "$CLUSTER" --config k8s/kind-config.yaml 2>/dev/null || true

echo "==> Loading image into Kind cluster..."
kind load docker-image "$IMAGE" --name "$CLUSTER"

echo "==> Copying MLflow runs into Kind node..."
for NODE in $(kind get nodes --name "$CLUSTER"); do
  docker exec "$NODE" sh -c "rm -rf /mlruns && mkdir -p /mlruns/.trash"
  docker cp "mlruns/." "${NODE}:/mlruns"
  docker exec "$NODE" sh -c "mkdir -p /mlruns/.trash"
done

echo "==> Applying Kubernetes manifests..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl set env deployment/loan-approval-deployment BUILD_ID="$BUILD_ID" GIT_SHA="$GITHUB_SHA"
kubectl annotate deployment/loan-approval-deployment \
  ci.github.com/build-id="$BUILD_ID" ci.github.com/git-sha="$GITHUB_SHA" --overwrite

kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl set env deployment/mlflow-deployment BUILD_ID="$BUILD_ID" GIT_SHA="$GITHUB_SHA"
kubectl annotate deployment/mlflow-deployment \
  ci.github.com/build-id="$BUILD_ID" ci.github.com/git-sha="$GITHUB_SHA" --overwrite

echo "==> Waiting for pods to be ready..."
kubectl rollout status deployment/loan-approval-deployment
kubectl rollout status deployment/mlflow-deployment

echo "==> Pods and services:"
kubectl get pods
kubectl get svc

echo ""
echo "==> Access the API via:"
echo "    kubectl port-forward service/loan-approval-service 5000:5000"
echo "    curl http://localhost:5000/health"
echo ""
echo "==> Access MLflow metrics for build ${BUILD_ID} via:"
echo "    kubectl port-forward service/mlflow-service 5001:5001"
echo "    open http://localhost:5001"
