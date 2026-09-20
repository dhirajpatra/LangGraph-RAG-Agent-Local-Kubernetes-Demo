#!/usr/bin/env bash
# Tears down just the app namespace (keeps minikube itself running), or
# pass --all to delete the whole minikube cluster + registry forwarder.
set -euo pipefail

NAMESPACE="rag-demo"

if [ "${1:-}" == "--all" ]; then
  echo "Deleting entire minikube cluster and registry forwarder..."
  docker rm -f rag-registry-forward >/dev/null 2>&1 || true
  minikube delete
else
  echo "Deleting namespace ${NAMESPACE} (minikube cluster stays running)..."
  kubectl delete namespace "${NAMESPACE}" --ignore-not-found
fi

echo "Done."
