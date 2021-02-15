#!/usr/bin/env bash
# Job name for the scheduler
JOB_NAME=${JOB_NAME:-single}
# Max time
TIME=${TIME:-"08:00:00"}

sbatch \
  --job-name="$JOB_NAME" \
  --time="$TIME" \
  ./scripts/single_model/slurm_job_stub.job $@
