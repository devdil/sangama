#!/bin/sh
set -eu
SANGAMA_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$SANGAMA_ROOT/.tools/opencode"
cp "$SANGAMA_ROOT/integrations/opencode/package.json" "$SANGAMA_ROOT/integrations/opencode/package-lock.json" "$SANGAMA_ROOT/.tools/opencode/"
npm ci --prefix "$SANGAMA_ROOT/.tools/opencode" --ignore-scripts
# Reviewed upstream installer copies the platform binary and verifies --version.
node "$SANGAMA_ROOT/.tools/opencode/node_modules/opencode-ai/postinstall.mjs"
