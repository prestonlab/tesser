#!/bin/bash
#
# Convert FreeSurfer hippocampal segmentation and project to native space.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_fs_rois.sh prepdir subject"
    exit 1
fi

prepdir=$1
subject=$2

# convert to nifti
temp=$SCRATCH/model/$subject
mkdir -p "$temp"

# do image math to get masks of interest
funcdir=$prepdir/sub-${subject}/func
for run in 1 2 3 4 5 6; do
    mkdir -p "$temp/$run"

    base=sub-${subject}_task-struct_run-${run}_space-T1w
    aparc=$funcdir/${base}_desc-aparcaseg_dseg.nii.gz
    cp "$aparc" "$temp/$run/aparc.nii.gz"
    cd "$temp/$run" || exit

    # medialorbitofrontal X014
    fslmaths aparc -thr 1014 -uthr 1014 -bin l_morb
    fslmaths aparc -thr 2014 -uthr 2014 -bin r_morb
    fslmaths l_morb -add r_morb -bin b_morb

    # parsopercularis X018
    fslmaths aparc -thr 1018 -uthr 1018 -bin l_oper
    fslmaths aparc -thr 2018 -uthr 2018 -bin r_oper
    fslmaths l_oper -add r_oper -bin b_oper

    # parsorbitalis X019
    fslmaths aparc -thr 1019 -uthr 1019 -bin l_orbi
    fslmaths aparc -thr 2019 -uthr 2019 -bin r_orbi
    fslmaths l_orbi -add r_orbi -bin b_orbi

    # parstriangularis X020
    fslmaths aparc -thr 1020 -uthr 1020 -bin l_tria
    fslmaths aparc -thr 2020 -uthr 2020 -bin r_tria
    fslmaths l_tria -add r_tria -bin b_tria

    # ifg
    fslmaths l_oper -add l_orbi -add l_tria -bin l_ifg
    fslmaths r_oper -add r_orbi -add r_tria -bin r_ifg
    fslmaths l_ifg -add r_ifg -bin b_ifg

    # pericalcarine X021
    fslmaths aparc -thr 1021 -uthr 1021 -bin l_peri
    fslmaths aparc -thr 2021 -uthr 2021 -bin r_peri
    fslmaths l_peri -add r_peri -bin b_peri

    # copy to output with BIDS naming
    for roi in morb oper orbi tria ifg peri; do
        for hemi in l r b; do
            cp "${hemi}_${roi}.nii.gz" "$funcdir/${base}_desc-${hemi}${roi}_mask.nii.gz"
        done
    done
done
