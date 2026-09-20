#!/usr/bin/env bash
#
# run_tests.sh
#
# Runs the fast, fully-mocked unit + API test suite (tests/unit, tests/api).
# No cluster, no API keys, no network needed.
#
#   ./run_tests.sh              # unit + api tests
#   ./run_tests.sh --all        # also attempts tests/integration (skips
#                                  automatically if infra isn't reachable)
#   ./run_tests.sh --coverage   # adds a coverage report
#
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ ! -d venv ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Tip: consider running this inside a virtualenv:"
  echo "  python3 -m venv venv && source venv/bin/activate"
fi

pip install --quiet -r requirements-dev.txt

ARGS=("tests/unit" "tests/api")
PYTEST_FLAGS=()

for arg in "$@"; do
  case "$arg" in
    --all)
      ARGS+=("tests/integration")
      ;;
    --coverage)
      PYTEST_FLAGS+=("--cov=app" "--cov-report=term-missing")
      ;;
    *)
      echo "Unknown option: $arg"
      exit 1
      ;;
  esac
done

PYTHONPATH="${ROOT_DIR}" python3 -m pytest "${PYTEST_FLAGS[@]}" "${ARGS[@]}"
