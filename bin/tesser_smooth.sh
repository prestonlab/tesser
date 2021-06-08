#!/bin/bash
#
# Smooth functional data and copy related metadata to the results directory.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_smooth.sh prepdir outdir subject"
    exit 1
fi

prepdir=$1
outdir=$2
subject=$3

prepsubj=$prepdir/sub-$subject
outsubj=$outdir/sub-$subject
mkdir -p "$outsubj/anat"
mkdir -p "$outsubj/func"

cp "$prepsubj/anat/sub-100_from-fsnative_to-T1w_mode-image_xfm.txt" "$outsubj/anat"
for run in 1 2 3 4 5 6; do
    base="sub-${subject}_task-struct_run-${run}"
    cp "$prepsubj/func/${base}_desc-brain_mask".{nii.gz,json} "$outsubj/func"
    cp "$prepsubj/func/${base}_desc-confounds_timeseries".{tsv,json} "$outsubj/func"
    cp "$prepsubj/func/${base}_desc-preproc_bold.json" "$outsubj/func"
    cp "$prepsubj/func/${base}_from-scanner_to-T1w_mode-image_xfm.txt" "$outsubj/func"
    cp "$prepsubj/func/${base}_from-T1w_to-scanner_mode-image_xfm.txt" "$outsubj/func"
    smooth_susan \
        "$prepsubj/func/${base}_desc-preproc_bold.nii.gz" \
        "$prepsubj/func/${base}_desc-brain_mask.nii.gz" \
        4.0 \
        "$outsubj/func/${base}_desc-smooth4mm_bold.nii.gz"
done
