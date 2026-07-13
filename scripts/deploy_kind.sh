#!/usr/bin/env bash
# Full Kind deployment script
# Make sure you have kind, kubectl, and docker installed
set -e

IMAGE="loan-approval-mlops:latest"
CLUSTER="loan-cluster"

echo "==> Building Docker image..."
docker build -t $IMAGE .

echo "==> Creating Kind cluster (skips if already exists)..."
kind create cluster --name $CLUSTER --config k8s/kind-config.yaml 2>/dev/null || true

echo "==> Loading image into Kind cluster..."
kind load docker-image $IMAGE --name $CLUSTER

echo "==> Applying Kubernetes manifests..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "==> Waiting for pods to be ready..."
kubectl rollout status deployment/loan-approval-deployment

echo "==> Pods and services:"
kubectl get pods
kubectl get svc

echo ""
echo "==> Access the API via:"
echo "    kubectl port-forward service/loan-approval-service 5000:5000"
echo "    curl http://localhost:5000/health"
