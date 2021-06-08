#!/bin/bash
#
# Consolidate mPFC ROIs to make three anterior/posterior ROIs.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_mpfc_rois.sh seg outdir subject"
    exit 1
fi

seg=$1
outdir=$2
subject=$3

temp=$SCRATCH/model/$subject
mkdir -p "$temp"
cp "$seg" "$temp/mpfc_mni.nii.gz"

# transform to native space
anatdir=$outdir/sub-${subject}/anat
funcdir=$outdir/sub-${subject}/func
for run in 1 2 3 4 5 6; do
    mkdir -p "$temp/$run"
    base=sub-${subject}_task-struct_run-${run}
    antsApplyTransforms -d 3 -e 0 \
        -i "$temp/mpfc_mni.nii.gz" \
        -r "$funcdir/${base}_space-T1w_boldref.nii.gz" \
        -o "$temp/$run/mpfc.nii.gz" \
        -n MultiLabel \
        -t "$anatdir/sub-${subject}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5"
done

# do image math to get masks of interest
for run in 1 2 3 4 5 6; do
    base=sub-${subject}_task-struct_run-${run}_space-T1w
    cd "$temp/$run" || exit

    # posterior
    fslmaths 25 -add 32pl -add 14c -add 24 -add 10m -add 14r -bin pmpfc

    # mid
    fslmaths 10r -add 11m -bin mmpfc

    # anterior
    imcp 10p ampfc

    # copy to output with BIDS naming
    for roi in pmpfc mmpfc ampfc; do
        cp "${roi}.nii.gz" "$funcdir/${base}_desc-${roi}_mask.nii.gz"
    done
done
