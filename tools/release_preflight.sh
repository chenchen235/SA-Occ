#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

test -f projects/configs/sa_occ/sa-occ-r50-aux.py
test -f projects/mmdet3d_plugin/models/backbones/resunet18.py
test -f projects/mmdet3d_plugin/models/necks/view_transformer_sat_dual.py

release_sources=(
  projects/configs/sa_occ/sa-occ-r50-aux.py
  projects/mmdet3d_plugin/models/backbones/resunet18.py
  projects/mmdet3d_plugin/models/detectors/sa_occ.py
  projects/mmdet3d_plugin/models/necks/view_transformer_sat_dual.py
  projects/mmdet3d_plugin/datasets/pipelines/loading.py
  tools/gen_sat.py
  tools/make_scenes_infos.py
)
if grep -nE '/workspace/|/root/|/export/|projects_ori|val_frame_dump' "${release_sources[@]}"; then
  echo 'release preflight: machine-local path or debug writer found' >&2
  exit 1
fi

python - <<'PY'
from pathlib import Path
for path in Path('projects').rglob('*.py'):
    if '__pycache__' in path.parts:
        continue
    compile(path.read_bytes(), str(path), 'exec')
print('python syntax: ok')
PY

if git diff --cached --name-only --diff-filter=ACMR \
    | grep -E '\.(pth|so|pyc|o|a|ninja|ninja_deps|ninja_log)$|(^|/)build/' >/dev/null; then
  echo 'release preflight: generated artifact found in release tree' >&2
  exit 1
fi

echo 'SA-Occ+ release preflight passed.'
