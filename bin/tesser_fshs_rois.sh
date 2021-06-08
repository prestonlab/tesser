#!/bin/bash
#
# Convert FreeSurfer hippocampal segmentation and project to native space.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_fshs_rois.sh studydir outdir subject"
    exit 1
fi

studydir=$1
outdir=$2
subject=$3

mri="$studydir/tesser_${subject}/anatomy/tesser_${subject}/mri"
l_hbt="$mri/lh.hippoAmygLabels-T1-T2.v21.HBT.mgz"
r_hbt="$mri/lh.hippoAmygLabels-T1-T2.v21.HBT.mgz"

# convert to nifti
temp=$SCRATCH/model/$subject
mkdir -p "$temp"
mri_convert "$l_hbt" "$temp/l_hbt_fsnative.nii.gz"
mri_convert "$r_hbt" "$temp/r_hbt_fsnative.nii.gz"
fslreorient2std "$temp/l_hbt_fsnative" "$temp/l_hbt_fsnative"
fslreorient2std "$temp/r_hbt_fsnative" "$temp/r_hbt_fsnative"

# transform to native space
anatdir=$outdir/sub-${subject}/anat
funcdir=$outdir/sub-${subject}/func
for run in 1 2 3 4 5 6; do
    mkdir -p "$temp/$run"
    base=sub-${subject}_task-struct_run-${run}
    for hemi in l r; do
        antsApplyTransforms -d 3 -e 0 \
            -i "$temp/${hemi}_hbt_fsnative.nii.gz" \
            -r "$funcdir/${base}_desc-brain_mask.nii.gz" \
            -o "$temp/$run/${hemi}_hbt.nii.gz" \
            -n MultiLabel \
            -t "$funcdir/${base}_from-T1w_to-scanner_mode-image_xfm.txt" \
            -t "$anatdir/sub-${subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    done
done

# do image math to get masks of interest
for run in 1 2 3 4 5 6; do
    base=sub-${subject}_task-struct_run-${run}
    cd "$temp/$run" || exit

    # tail 226
    fslmaths l_hbt -thr 226 -uthr 226 -bin l_tail
    fslmaths r_hbt -thr 226 -uthr 226 -bin r_tail
    fslmaths l_tail -add r_tail -bin b_tail

    # body 231
    fslmaths l_hbt -thr 231 -uthr 231 l_body
    fslmaths r_hbt -thr 231 -uthr 231 r_body
    fslmaths l_body -add r_body -bin b_body

    # head 232
    fslmaths l_hbt -thr 232 -uthr 232 l_head
    fslmaths r_hbt -thr 232 -uthr 232 r_head
    fslmaths l_head -add r_head -bin b_head

    # posterior
    fslmaths l_tail -add l_body -bin l_post
    fslmaths r_tail -add r_body -bin r_post
    fslmaths l_post -add r_post -bin b_post

    # copy to output with BIDS naming
    for roi in tail body head post; do
        for hemi in l r b; do
            cp "${hemi}_${roi}.nii.gz" "$funcdir/${base}_desc-${hemi}hip${roi}_mask.nii.gz"
        done
    done
done
