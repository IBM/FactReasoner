#!/usr/bin/env bash
# Build the demo image in the cluster from this checkout.
#
#   deploy/openshift/build.sh [namespace]      (default: factuality)
#
# Uploads only what the Containerfile copies -- about 1 MB -- rather than the
# whole repository (gigabytes of .venv, results and data), then follows the
# build log. A successful build pushes factreasoner-demo:latest, and the
# Deployment's image trigger rolls the app out.
set -euo pipefail

NS="${1:-factuality}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT

cd "$ROOT"
mkdir -p "$CTX/deploy/openshift" "$CTX/data" "$CTX/configs"
cp pyproject.toml uv.lock README.md "$CTX/"
cp deploy/openshift/Containerfile "$CTX/deploy/openshift/"
cp -R src "$CTX/src"
cp -R data/lcs "$CTX/data/lcs"
cp configs/rits_models.json "$CTX/configs/"
# No bytecode or editor droppings in the upload.
find "$CTX" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$CTX" -name '*.pyc' -delete

echo "build context: $(du -sh "$CTX" | cut -f1) from $(git rev-parse --short HEAD)$(git diff --quiet || echo ' (uncommitted changes included)')"
oc -n "$NS" start-build factreasoner-demo --from-dir="$CTX" --follow --wait
