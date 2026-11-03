#!/usr/bin/env bash
# Download the T15 files from Dryad (doi:10.5061/dryad.dncjsxm85). Needs a Dryad API bearer token in $DRYAD_TOKEN.
# The token is read from the environment and never printed. UNTESTED with a real token (none was available).
# Usage: DRYAD_TOKEN=... scripts/download_t15.sh <dest_dir> [file_name ...]   (default: t15_copyTask_neuralData.zip)
set -euo pipefail
: "${DRYAD_TOKEN:?set DRYAD_TOKEN (Dryad API bearer token) - see https://datadryad.org/api}"
dest="${1:?dest dir}"; shift || true
files=("${@:-t15_copyTask_neuralData.zip}")
mkdir -p "$dest"
auth=$(mktemp); trap 'rm -f "$auth"' EXIT; printf 'Authorization: Bearer %s\n' "$DRYAD_TOKEN" > "$auth"
api=https://datadryad.org/api/v2
ver=$(curl -fsS "$api/datasets/doi%3A10.5061%2Fdryad.dncjsxm85" | python3 -c "import sys,json;print(json.load(sys.stdin)['_links']['stash:version']['href'])")
curl -fsS "$api$ver/files" > "$dest/.files.json"
for f in "${files[@]}"; do
  id=$(python3 - "$f" "$dest/.files.json" <<'PY'
import sys, json
name, path = sys.argv[1], sys.argv[2]
for x in json.load(open(path))['_embedded']['stash:files']:
    if x['path'] == name: print(x['_links']['stash:download']['href'].split('/')[-2]); break
else: sys.exit(f"{name} not found in dataset")
PY
)
  echo "downloading $f (file id $id) ..."
  curl -fL -H @"$auth" --retry 3 -C - -o "$dest/$f" "$api/files/$id/download"
done
echo "done. next: unzip, then python scripts/inspect_hdf5.py <root>"
