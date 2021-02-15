#!/usr/bin/env bash
# Number of BOHB workers
NUM_WORKERS=${NUM_WORKERS:-8}
# Job name
JOB_NAME=${JOB_NAME:-abas-alda}
# Maximum time
TIME=${TIME:-"20:00:00"}

# Launch the job array
sbatch $SBATCH_FLAGS \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  --array="1-$NUM_WORKERS" \
  ./scripts/slurm_job_stub.job $@
