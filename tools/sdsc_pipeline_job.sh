#!/bin/bash
set -euo pipefail
if [[ $# != 3 || $1 != /* || $2 != /* ]]; then exit 2; fi
exec "$1" -I -B -u "$(dirname -- "$0")/sdsc_pipeline_job.py" --request "$2" --request-sha256 "$3"
