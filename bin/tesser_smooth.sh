#!/bin/bash
#
# Smooth functional data and copy related metadata to the results directory.

if [[ $# -lt 4 ]]; then
    echo "Usage: tesser_smooth.sh prepdir outdir subject space"
    exit 1
fi

prepdir=$1
outdir=$2
subject=$3
space=$4

prepsubj=$prepdir/sub-$subject
outsubj=$outdir/sub-$subject
mkdir -p "$outsubj/anat"
mkdir -p "$outsubj/func"

cp "$prepsubj/anat/sub-${subject}_desc-"{preproc_T1w,brain_mask}.{nii.gz,json} "$outsubj/anat"
cp "$prepsubj/anat/sub-${subject}_from-fsnative_to-T1w_mode-image_xfm.txt" "$outsubj/anat"
cp "$prepsubj/anat/sub-${subject}_from-T1w_to-fsnative_mode-image_xfm.txt" "$outsubj/anat"
cp "$prepsubj/anat/sub-${subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5" "$outsubj/anat"
cp "$prepsubj/anat/sub-${subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" "$outsubj/anat"

for run in 1 2 3 4 5 6; do
    echo "Smoothing run ${run}..."
    base="sub-${subject}_task-struct_run-${run}"
    cp "$prepsubj/func/${base}_desc-confounds_timeseries".{tsv,json} "$outsubj/func"
    cp "$prepsubj/func/${base}_from-scanner_to-T1w_mode-image_xfm.txt" "$outsubj/func"
    cp "$prepsubj/func/${base}_from-T1w_to-scanner_mode-image_xfm.txt" "$outsubj/func"

    cp "$prepsubj/func/${base}_space-${space}_desc-brain_mask".{nii.gz,json} "$outsubj/func"
    cp "$prepsubj/func/${base}_space-${space}_boldref.nii.gz" "$outsubj/func"
    cp "$prepsubj/func/${base}_space-${space}_desc-preproc_bold.json" "$outsubj/func"

    smooth_susan \
        "$prepsubj/func/${base}_space-${space}_desc-preproc_bold.nii.gz" \
        "$prepsubj/func/${base}_space-${space}_desc-brain_mask.nii.gz" \
        4.0 \
        "$outsubj/func/${base}_space-${space}_desc-smooth4mm_bold.nii.gz"
done
