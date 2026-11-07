#!/usr/bin/env bash
# Stage a price-change workbook for import (rung 3). Validates it, copies it into the import
# folder, logs, and prints a JSON summary on stdout for the caller.
set -euo pipefail

src="${1:?usage: stage_import.sh <price_changes.xlsx>}"
dest_dir="${ACME_IMPORT:-/srv/import}"
log="${ACME_LOG_DIR:-/var/log/acme}/import.log"

[[ -f "$src" ]] || { echo "{\"ok\": false, \"error\": \"missing $src\"}"; exit 2; }

rows=$(/opt/acme/venv/bin/python - "$src" <<'EOF'
import sys
from openpyxl import load_workbook
ws = load_workbook(sys.argv[1], read_only=True).active
header = [c.value for c in next(ws.iter_rows(max_row=1))]
need = {"sku", "current_price", "new_price"}
missing = need - set(header)
if missing:
    sys.exit(f"missing columns: {sorted(missing)}")
print(sum(1 for r in ws.iter_rows(min_row=2, values_only=True) if r[0]))
EOF
) || { echo "{\"ok\": false, \"error\": \"validation failed\"}"; exit 3; }

mkdir -p "$dest_dir" "$(dirname "$log")"
name="$(date +%Y%m%dT%H%M%S)_$(basename "$src")"
cp "$src" "$dest_dir/$name"
sha=$(sha256sum "$dest_dir/$name" | cut -d' ' -f1)
echo "$(date -Is) staged $name rows=$rows sha256=$sha" >> "$log"
echo "{\"ok\": true, \"staged\": \"$dest_dir/$name\", \"rows\": $rows, \"sha256\": \"$sha\"}"
