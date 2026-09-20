#!/usr/bin/env bash
#
# tests/smoke/smoke_test.sh
#
# End-to-end smoke test against your ACTUAL running deployment (real
# OpenAI/Tavily calls happen here — unlike everything under tests/unit and
# tests/api, this is not mocked). Run this after `./setup.sh` to confirm
# the whole stack works together:
#
#   ./tests/smoke/smoke_test.sh
#
# It port-forwards rag-backend-service, health-checks it, ingests a small
# test document, polls until ingestion finishes, then asks a question that
# should be answerable from that document and checks the response shape.
#
set -euo pipefail

NAMESPACE="rag-demo"
LOCAL_PORT=8000
BACKEND_URL="http://localhost:${LOCAL_PORT}"
PF_PID=""

c_green() { echo -e "\033[0;32m$*\033[0m"; }
c_red() { echo -e "\033[0;31m$*\033[0m"; }
step() { echo; c_green "==> $*"; }

cleanup() {
  if [ -n "${PF_PID}" ] && kill -0 "${PF_PID}" 2>/dev/null; then
    kill "${PF_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

step "Port-forwarding rag-backend-service -> localhost:${LOCAL_PORT}"
kubectl port-forward -n "${NAMESPACE}" svc/rag-backend-service "${LOCAL_PORT}:8000" >/tmp/rag-smoke-pf.log 2>&1 &
PF_PID=$!

step "Waiting for backend health check"
for i in $(seq 1 30); do
  if curl -sf "${BACKEND_URL}/health" >/dev/null 2>&1; then
    c_green "backend is healthy"
    break
  fi
  if [ "$i" -eq 30 ]; then
    c_red "backend never became healthy — check: kubectl get pods -n ${NAMESPACE}"
    exit 1
  fi
  sleep 2
done

MARKER="smoke-test-marker-$(date +%s)"
step "Ingesting a test document (marker: ${MARKER})"
INGEST_RESPONSE=$(curl -sf -X POST "${BACKEND_URL}/api/v1/advanced/ingest-async" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"The secret smoke test code word is ${MARKER}. This sentence exists only to verify ingestion and retrieval work end to end.\", \"source\": \"smoke-test\"}")

TASK_ID=$(echo "${INGEST_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['task_id'])")
c_green "queued ingestion task: ${TASK_ID}"

step "Polling ingestion status"
for i in $(seq 1 30); do
  STATUS_RESPONSE=$(curl -sf "${BACKEND_URL}/api/v1/advanced/ingest-status/${TASK_ID}")
  STATUS=$(echo "${STATUS_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
  if [ "${STATUS}" == "SUCCESS" ]; then
    c_green "ingestion finished"
    break
  fi
  if [ "${STATUS}" == "FAILURE" ]; then
    c_red "ingestion task failed: ${STATUS_RESPONSE}"
    exit 1
  fi
  if [ "$i" -eq 30 ]; then
    c_red "ingestion never finished (last status: ${STATUS}) — check: kubectl logs -f deploy/rag-worker -n ${NAMESPACE}"
    exit 1
  fi
  sleep 2
done

step "Asking a question that should retrieve the ingested document"
QUERY_RESPONSE=$(curl -sf -X POST "${BACKEND_URL}/api/v1/advanced/query" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is the secret smoke test code word?\"}")

echo "${QUERY_RESPONSE}" | python3 -m json.tool

ANSWER=$(echo "${QUERY_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['answer'])")
STRATEGY=$(echo "${QUERY_RESPONSE}" | python3 -c "import sys, json; print(json.load(sys.stdin)['strategy'])")

if echo "${ANSWER}" | grep -q "${MARKER}"; then
  c_green "PASS: answer contains the expected marker '${MARKER}' (strategy: ${STRATEGY})"
else
  c_red "FAIL: answer did not contain the expected marker."
  c_red "This can happen legitimately if grade_context routed to web_search"
  c_red "instead of pgvector — check the 'strategy' field above."
  exit 1
fi

step "Smoke test passed."
