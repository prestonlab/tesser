#!/bin/bash
#
# Run heudiconv to convert raw data for one subject.

if [[ $# -lt 4 ]]; then
    echo "Usage: tesser_heudiconv.sh subject raw_dir heuristic bids_dir"
    exit 1
fi

subject=$1
raw_dir=$2
heuristic=$3
bids_dir=$4

subj_raw_dir=$raw_dir/$subject/raw/$subject
if [[ ! -d $subj_raw_dir ]]; then
    echo "Error: raw directory not found: $subj_raw_dir"
    exit 1
fi

if [[ ! -f $heuristic ]]; then
    echo "Error: heuristic file does not exist: $heuristic"
    exit 1
fi

image=$STOCKYARD2/software/images/heudiconv-0.9.0.sif
module load tacc-singularity
singularity run "$image" \
    -s "$subject" \
    -f "$heuristic" \
    -b \
    -o "$bids_dir" \
    --minmeta \
    --files "$subj_raw_dir"/*/*.dcm
