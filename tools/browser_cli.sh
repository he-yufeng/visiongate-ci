#!/usr/bin/env bash
set -euo pipefail
# Bundled Playwright skill wrapper pattern, exact installed package instead of latest.
if ! command -v npx >/dev/null 2>&1; then
  echo 'npx is required' >&2
  exit 1
fi
exec npx --no-install --package @playwright/cli@0.1.22 playwright-cli "$@"
