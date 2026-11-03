#!/usr/bin/env bash
# Run the paired study sequentially, with pretraining before downstream stages.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${EEGX_PYTHON:-$PWD/.venv/bin/python}"
config_dir="${EEGX_CONFIG_DIR:-$PWD/configs}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

run_config() {
    printf 'Running %s\n' "$1"
    "$python_bin" Main.py --config "$config_dir/$1.yaml"
}

run_config eegx
run_config dreamer-arousal-finetune
for task in stew tuab tuev-binary \
    crowdsourced-in-domain bciciv-2a-in-domain kalunga-in-domain physionet-mi-in-domain; do
    run_config "$task"
    run_config "$task-finetune"
done

run_config tuab-source
for task in crowdsourced bciciv-2a kalunga physionet-mi; do
    run_config "$task-linear-probe"
    run_config "$task-finetune"
done
