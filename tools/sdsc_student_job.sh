#!/bin/bash
# Two-H100 adapted-teacher preflight or accepted-demo canonical-SFT calibration.
set -euo pipefail
if [[ $# != 9 || $4 != /* || ! -x $4 ]]; then
    printf '%s\n' 'usage: sdsc_student_job.sh RELEASE SUBMISSION RESULTS ABSOLUTE_PYTHON RUN_ID CODE_SHA256 HF_HOME PROVENANCE_DIR PROVENANCE_SHA256' >&2
    exit 2
fi
exec "$4" -I -B -u "$(dirname -- "$0")/sdsc_student_job.py" "$@"
