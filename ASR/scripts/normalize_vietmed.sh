#!/usr/bin/env bash

set -euo pipefail

python local/normalize_vietmed.py \
  --src-dir /path/to/raw/VietMed \
  --metadata /path/to/raw/VietMed/metadata.csv \
  --output-dir download/VietMed \
  --default-split train
