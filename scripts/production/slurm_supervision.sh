#!/usr/bin/env bash
# Fail-closed Slurm polling shared by Qwen3-v2 supervisors.

slurm_query_with_retry() {
  local output_file=$1
  shift
  local retries=${SLURM_QUERY_RETRIES:-6}
  local delay=${SLURM_QUERY_INITIAL_BACKOFF_SECONDS:-5}
  local attempt
  for ((attempt=1; attempt<=retries; attempt++)); do
    if "$@" >"${output_file}" 2>"${output_file}.stderr"; then
      return 0
    fi
    if (( attempt < retries )); then
      sleep "${delay}"
      delay=$((delay * 2))
    fi
  done
  echo "Slurm query failed after ${retries} attempts: $*" >&2
  return 1
}

require_no_competing_opd_gpu_job() {
  local candidate=${1:-four_gpu_stage}
  local listing
  listing=$(mktemp "${TMPDIR:-/tmp}/opd-gpu-jobs.XXXXXX")
  if ! slurm_query_with_retry "${listing}" squeue -u "${USER:?}" -h -o '%i|%j|%T|%b'; then
    rm -f "${listing}" "${listing}.stderr"
    return 1
  fi
  local conflicts
  conflicts=$(awk -F'|' '
    $2 ~ /^opd-/ && $3 ~ /^(PENDING|RUNNING|CONFIGURING|COMPLETING)$/ &&
      (tolower($4) ~ /gpu/ || tolower($4) ~ /gres\/gpu/) {print}
  ' "${listing}")
  rm -f "${listing}" "${listing}.stderr"
  if [[ -n "${conflicts}" ]]; then
    echo "${candidate} refused: another OPD GPU job could allocate concurrently:" >&2
    echo "${conflicts}" >&2
    return 1
  fi
}

wait_for_slurm_terminal() {
  local job_id=$1
  local terminal=$2
  local poll_file="${terminal}.squeue"
  local accounting_retries=${SLURM_ACCOUNTING_RETRIES:-12}
  local accounting_delay=${SLURM_ACCOUNTING_INITIAL_BACKOFF_SECONDS:-5}
  while true; do
    slurm_query_with_retry "${poll_file}" squeue -h -j "${job_id}" -o '%i|%T' || return 1
    if [[ ! -s "${poll_file}" ]]; then
      break
    fi
    sleep "${SLURM_POLL_SECONDS:-30}"
  done
  local attempt
  for ((attempt=1; attempt<=accounting_retries; attempt++)); do
    if slurm_query_with_retry "${terminal}.candidate" \
      sacct -nP -X -j "${job_id}" --format=JobIDRaw,State,ExitCode &&
      awk -F'|' '
        BEGIN {seen=0; bad=0; pending=0}
        NF >= 3 && $1 != "" {seen=1; if ($2 == "" || $2 ~ /^UNKNOWN/) bad=1;
          else if ($2 !~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE)/) pending=1}
        END {exit (!seen || bad || pending)}
      ' "${terminal}.candidate"; then
      mv "${terminal}.candidate" "${terminal}"
      rm -f "${poll_file}" "${poll_file}.stderr" "${terminal}.candidate.stderr"
      if awk -F'|' 'NF >= 3 && ($2 !~ /^COMPLETED/ || $3 != "0:0") {bad=1} END {exit bad}' \
        "${terminal}"; then
        return 0
      fi
      echo "Slurm job ${job_id} reached a non-success terminal state" >&2
      return 1
    fi
    if (( attempt < accounting_retries )); then
      sleep "${accounting_delay}"
      accounting_delay=$((accounting_delay * 2))
    fi
  done
  echo "Slurm accounting for ${job_id} stayed empty, UNKNOWN, or nonterminal" >&2
  return 1
}
