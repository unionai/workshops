#!/bin/sh
# Container entrypoint for the UI app.
#
# The build happens here rather than at image-build time so that VITE_API_BASE — which
# is only known once the API app has an endpoint — gets baked into the bundle.
set -e

# Run from this script's own directory. The code bundle is rooted at the project root,
# so these files land under ui/ while the container's working directory is the bundle
# root — without this, npm finds no package.json.
cd "$(dirname "$0")"

if [ -z "$VITE_API_BASE" ]; then
  echo "WARNING: VITE_API_BASE is unset. The UI will try same-origin /api and fail."
  echo "         Deploy via deploy.py, which wires it to the API app's endpoint."
fi

echo "Building UI against API base: ${VITE_API_BASE:-<same origin>}"

npm install --no-audit --no-fund
npm run build
npm run preview
