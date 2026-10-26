#!/usr/bin/env bash
# Run four dataset pipelines at a time. Each YAML runs its saved seeds sequentially.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${EEGX_PYTHON:-$PWD/.venv/bin/python}"
if [[ -f "$PWD/.runs/current-paper-queues.json" ]]; then
    exec "$python_bin" scripts/paper/launch.py --current "$PWD/.runs/current-paper-queues.json"
fi
config_dir="${EEGX_CONFIG_DIR:-$PWD/configs}"
log_dir="${EEGX_LOG_DIR:-$PWD/.runs/paper-v1/logs}"
mkdir -p -- "$log_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

run_config() {
    printf '%s START %s\n' "$(date -u +%FT%TZ)" "$1"
    "$python_bin" Main.py --config "$config_dir/$1.yaml" || return "$?"
    printf '%s COMPLETE %s\n' "$(date -u +%FT%TZ)" "$1"
}

in_domain() {
    run_config "$1" && run_config "$2"
}

cross_domain() {
    run_config "$1-linear-probe" && run_config "$1-finetune"
}

wait_all() {
    local failed=0 pid
    for pid in "$@"; do
        wait "$pid" || failed=1
    done
    return "$failed"
}

in_domain eegx dreamer-arousal-finetune >"$log_dir/dreamer-arousal.log" 2>&1 &
dreamer_pid=$!
in_domain stew stew-finetune >"$log_dir/stew.log" 2>&1 &
stew_pid=$!
in_domain tuab tuab-finetune >"$log_dir/tuab.log" 2>&1 &
tuab_pid=$!
in_domain tuev-binary tuev-binary-finetune >"$log_dir/tuev-binary.log" 2>&1 &
tuev_pid=$!
wait_all "$dreamer_pid" "$stew_pid" "$tuab_pid" "$tuev_pid"

run_config tuab-source >"$log_dir/tuab-source.log" 2>&1

cross_domain crowdsourced >"$log_dir/crowdsourced.log" 2>&1 &
crowdsourced_pid=$!
cross_domain bciciv-2a >"$log_dir/bciciv-2a.log" 2>&1 &
bciciv_pid=$!
cross_domain kalunga >"$log_dir/kalunga.log" 2>&1 &
kalunga_pid=$!
cross_domain physionet-mi >"$log_dir/physionet-mi.log" 2>&1 &
physionet_pid=$!
wait_all "$crowdsourced_pid" "$bciciv_pid" "$kalunga_pid" "$physionet_pid"
printf '%s All paper runs complete.\n' "$(date -u +%FT%TZ)"
