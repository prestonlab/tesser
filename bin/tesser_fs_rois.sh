#!/bin/bash
#
# Convert FreeSurfer hippocampal segmentation and project to native space.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_fs_rois.sh prepdir outdir subject"
    exit 1
fi

prepdir=$1
outdir=$2
subject=$3

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
    fslmaths aparc -thr 1014 -uthr 1014 -bin l_fsmorb
    fslmaths aparc -thr 2014 -uthr 2014 -bin r_fsmorb
    fslmaths l_fsmorb -add r_fsmorb -bin b_fsmorb

    # parsopercularis X018
    fslmaths aparc -thr 1018 -uthr 1018 -bin l_fsoper
    fslmaths aparc -thr 2018 -uthr 2018 -bin r_fsoper
    fslmaths l_fsoper -add r_fsoper -bin b_fsoper

    # parsorbitalis X019
    fslmaths aparc -thr 1019 -uthr 1019 -bin l_fsorbi
    fslmaths aparc -thr 2019 -uthr 2019 -bin r_fsorbi
    fslmaths l_fsorbi -add r_fsorbi -bin b_fsorbi

    # parstriangularis X020
    fslmaths aparc -thr 1020 -uthr 1020 -bin l_fstria
    fslmaths aparc -thr 2020 -uthr 2020 -bin r_fstria
    fslmaths l_fstria -add r_fstria -bin b_fstria

    # ifg
    fslmaths l_fsoper -add l_fsorbi -add l_fstria -bin l_fsifg
    fslmaths r_fsoper -add r_fsorbi -add r_fstria -bin r_fsifg
    fslmaths l_fsifg -add r_fsifg -bin b_fsifg

    # pericalcarine X021
    fslmaths aparc -thr 1021 -uthr 1021 -bin l_fsperi
    fslmaths aparc -thr 2021 -uthr 2021 -bin r_fsperi
    fslmaths l_fsperi -add r_fsperi -bin b_fsperi

    # copy to output with BIDS naming
    resdir=$outdir/sub-${subject}/func
    mkdir -p "$resdir"
    for roi in morb oper orbi tria ifg peri; do
        for hemi in l r b; do
            cp "${hemi}_${roi}.nii.gz" "$resdir/${base}_desc-${hemi}${roi}_mask.nii.gz"
        done
    done
done
