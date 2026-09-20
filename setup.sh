#!/usr/bin/env bash
#
# setup.sh — one script to install prerequisites (only if missing), build the
# app image, and bring the whole architecture up on a single-node minikube
# cluster, exactly as laid out in the architecture diagram.
#
# Safe to re-run any time: every install step checks "is this already
# there?" first, and every deploy step uses `kubectl apply` (idempotent) plus
# an explicit rollout restart so new code always reaches the pods.
#
set -euo pipefail

NAMESPACE="rag-demo"
IMAGE="localhost:5001/rag-app:fixed"
REGISTRY_FORWARD_NAME="rag-registry-forward"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
c_green() { echo -e "\033[0;32m$*\033[0m"; }
c_yellow() { echo -e "\033[0;33m$*\033[0m"; }
c_red() { echo -e "\033[0;31m$*\033[0m"; }
step() { echo; c_green "==> $*"; }

have() { command -v "$1" >/dev/null 2>&1; }

require_ubuntu() {
  if [ -f /etc/os-release ] && grep -qi ubuntu /etc/os-release; then
    return 0
  fi
  c_yellow "Warning: this script is written for Ubuntu. Continuing anyway."
}

# ---------------------------------------------------------------------------
# 1. Prerequisite checks / installs (each is a no-op if already installed)
# ---------------------------------------------------------------------------
install_docker() {
  if have docker; then
    c_green "docker already installed: $(docker --version)"
  else
    step "Installing Docker Engine"
    sudo apt-get update -y
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER" || true
    c_yellow "Added $USER to the docker group. If 'docker ps' below fails with a permission error, run 'newgrp docker' or log out/in, then re-run this script."
  fi
}

install_kubectl() {
  if have kubectl; then
    c_green "kubectl already installed: $(kubectl version --client --short 2>/dev/null || kubectl version --client)"
  else
    step "Installing kubectl"
    KVER="$(curl -L -s https://dl.k8s.io/release/stable.txt)"
    curl -LO "https://dl.k8s.io/release/${KVER}/bin/linux/amd64/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/kubectl
  fi
}

install_minikube() {
  if have minikube; then
    c_green "minikube already installed: $(minikube version --short 2>/dev/null || minikube version)"
  else
    step "Installing minikube"
    curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
    chmod +x minikube-linux-amd64
    sudo mv minikube-linux-amd64 /usr/local/bin/minikube
  fi
}

install_socat() {
  if have socat; then
    c_green "socat already installed"
  else
    step "Installing socat (used only as a fallback if the docker forwarder image is unavailable)"
    sudo apt-get update -y && sudo apt-get install -y socat
  fi
}

require_ubuntu
install_docker
install_kubectl
install_minikube
install_socat

# ---------------------------------------------------------------------------
# 2. .env check
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
  step ".env not found — creating it from .env.example"
  cp .env.example .env
  c_red "Edit .env now and fill in OPENAI_API_KEY, TAVILY_API_KEY and LANGSMITH_API_KEY,"
  c_red "then re-run ./setup.sh"
  exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a

for var in OPENAI_API_KEY TAVILY_API_KEY; do
  val="${!var:-}"
  if [ -z "$val" ] || [[ "$val" == *REPLACE_ME* ]] || [[ "$val" == *xxxx* ]]; then
    c_red "Please set a real value for $var in .env before continuing."
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 3. minikube cluster (single node, docker driver)
# ---------------------------------------------------------------------------
step "Ensuring minikube cluster is running"
if minikube status >/dev/null 2>&1; then
  c_green "minikube already running"
else
  minikube start --driver=docker --cpus=4 --memory=8192 --disk-size=20g
fi

step "Enabling the minikube registry addon"
minikube addons enable registry >/dev/null

MINIKUBE_IP="$(minikube ip)"

step "Ensuring the local registry is reachable at localhost:5001"
if docker ps --format '{{.Names}}' | grep -q "^${REGISTRY_FORWARD_NAME}$"; then
  c_green "registry port-forward already running"
else
  docker rm -f "${REGISTRY_FORWARD_NAME}" >/dev/null 2>&1 || true
  docker run --rm -d --name "${REGISTRY_FORWARD_NAME}" --network=host \
    alpine/socat TCP-LISTEN:5001,reuseaddr,fork "TCP:${MINIKUBE_IP}:5000" >/dev/null
  c_green "started ${REGISTRY_FORWARD_NAME} forwarding localhost:5001 -> ${MINIKUBE_IP}:5000"
fi

# ---------------------------------------------------------------------------
# 4. Build & push the app image ("Local Docker + Kubernetes Delivery")
# ---------------------------------------------------------------------------
step "Building Docker image ${IMAGE}"
docker build -t "${IMAGE}" .

step "Pushing image to local registry"
# Localhost registries are treated as insecure by Docker automatically, so no
# daemon.json changes are required.
for i in 1 2 3; do
  if docker push "${IMAGE}"; then
    break
  fi
  c_yellow "push attempt $i failed, retrying in 3s..."
  sleep 3
done

# ---------------------------------------------------------------------------
# 5. Namespace, ConfigMap, Secret
# ---------------------------------------------------------------------------
step "Applying namespace and config"
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/config.yaml

step "Creating/updating Secret rag-secrets from .env"
kubectl create secret generic rag-secrets \
  --namespace "${NAMESPACE}" \
  --from-literal=OPENAI_API_KEY="${OPENAI_API_KEY}" \
  --from-literal=TAVILY_API_KEY="${TAVILY_API_KEY}" \
  --from-literal=LANGSMITH_API_KEY="${LANGSMITH_API_KEY:-}" \
  --dry-run=client -o yaml | kubectl apply -f -

# ---------------------------------------------------------------------------
# 6. Infra: Postgres + pgvector, Redis Stack, RabbitMQ
# ---------------------------------------------------------------------------
step "Applying infra (postgres, redis, rabbitmq)"
kubectl apply -f k8s/infrastructure.yaml

step "Waiting for infra to become ready"
kubectl rollout status deployment/postgres -n "${NAMESPACE}" --timeout=180s
kubectl rollout status deployment/redis -n "${NAMESPACE}" --timeout=180s
kubectl rollout status deployment/rabbitmq -n "${NAMESPACE}" --timeout=180s

# ---------------------------------------------------------------------------
# 7. App layer: backend, worker, frontend
# ---------------------------------------------------------------------------
step "Applying app layer (backend, worker, frontend)"
kubectl apply -f k8s/application.yaml

step "Restarting app pods so they pick up the freshly-pushed image"
kubectl rollout restart deployment/rag-backend deployment/rag-worker deployment/rag-frontend -n "${NAMESPACE}"

step "Waiting for app layer to become ready"
kubectl rollout status deployment/rag-backend -n "${NAMESPACE}" --timeout=180s
kubectl rollout status deployment/rag-worker -n "${NAMESPACE}" --timeout=180s
kubectl rollout status deployment/rag-frontend -n "${NAMESPACE}" --timeout=180s

# ---------------------------------------------------------------------------
# 8. Done
# ---------------------------------------------------------------------------
MINIKUBE_IP="$(minikube ip)"
echo
c_green "=================================================================="
c_green " Everything is up."
c_green " Open the demo at:   http://${MINIKUBE_IP}:30085"
c_green " Backend health:     kubectl port-forward -n ${NAMESPACE} svc/rag-backend-service 8000:8000"
c_green " RabbitMQ mgmt UI:   kubectl port-forward -n ${NAMESPACE} svc/rabbitmq-service 15672:15672"
c_green " Pods:               kubectl get pods -n ${NAMESPACE}"
c_green "=================================================================="
