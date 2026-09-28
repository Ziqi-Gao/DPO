#!/bin/bash
# Slurm spools this launcher elsewhere; locate the accepted worker in RELEASE.
set -euo pipefail
if [[ $# != 9 || $1 != /* || $4 != /* || ! -x $4 ]]; then
    printf '%s\n' 'usage: sdsc_student_launch.sh RELEASE SUBMISSION RESULTS ABSOLUTE_PYTHON RUN_ID CODE_SHA256 HF_HOME PROVENANCE_DIR PROVENANCE_SHA256' >&2
    exit 2
fi
exec "$4" -I -B -u "$1/source/tools/sdsc_student_job.py" "$@"
