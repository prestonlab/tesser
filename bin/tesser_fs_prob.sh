#!/bin/bash
#
# Calculate group-level probability images from FreeSurfer.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_fs_prob.sh fsdir space outdir"
    exit 1
fi

fsdir=$1
space=$2
outdir=$3

# get a list of the subjects
cd "${fsdir}" || exit 1
subjects=""
for d in sub-*/; do
    subject=${d%/}
    subjects="${subjects} ${subject}"
done

# create masks from FreeSurfer output
tempdir=${SCRATCH}/fs_prob
ifgb_files=""
ifgl_files=""
ifgr_files=""
for subject in ${subjects}; do
    mkdir -p "${tempdir}/${subject}"
    cd "${tempdir}/${subject}" || exit 1
    anatdir=${outdir}/${subject}/anat

    # convert FreeSurfer results to Nifti and transform to template
    seg=${fsdir}/${subject}/mri/aparc+aseg.mgz
    mri_convert "${seg}" seg_fsnative.nii.gz
    fslreorient2std seg_fsnative seg_fsnative
    antsApplyTransforms -d 3 -e 0 \
        -i seg_fsnative.nii.gz \
        -r "${anatdir}/${subject}_space-${space}_desc-brain_mask.nii.gz" \
        -o parcels.nii.gz \
        -n MultiLabel \
        -t "${anatdir}/${subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" \
        -t "${anatdir}/${subject}_from-fsnative_to-T1w_mode-image_xfm.txt"

    # parsopercularis X018
    fslmaths parcels -thr 1018 -uthr 1018 -bin l_oper
    fslmaths parcels -thr 2018 -uthr 2018 -bin r_oper
    fslmaths l_oper -add r_oper -bin b_oper

    # parsorbitalis X019
    fslmaths parcels -thr 1019 -uthr 1019 -bin l_orbi
    fslmaths parcels -thr 2019 -uthr 2019 -bin r_orbi
    fslmaths l_orbi -add r_orbi -bin b_orbi

    # parstriangularis X020
    fslmaths parcels -thr 1020 -uthr 1020 -bin l_tria
    fslmaths parcels -thr 2020 -uthr 2020 -bin r_tria
    fslmaths l_tria -add r_tria -bin b_tria

    # ifg
    fslmaths l_oper -add l_orbi -add l_tria -bin ifgl
    fslmaths r_oper -add r_orbi -add r_tria -bin ifgr
    fslmaths ifgl -add ifgr -bin ifgb

    ifgb_files="${ifgb_files} ${tempdir}/${subject}/ifgb.nii.gz"
    ifgl_files="${ifgl_files} ${tempdir}/${subject}/ifgl.nii.gz"
    ifgr_files="${ifgr_files} ${tempdir}/${subject}/ifgr.nii.gz"
done

# average subject images to create probabilistic masks
cd "${tempdir}" || exit 1
fslmerge -t ifgl_all ${ifgl_files}
fslmaths ifgl_all -Tmean ifgl_prob
fslmerge -t ifgr_all ${ifgr_files}
fslmaths ifgr_all -Tmean ifgr_prob
fslmerge -t ifgb_all ${ifgb_files}
fslmaths ifgb_all -Tmean ifgb_prob

# copy to standard files in the output directory
for subject in $subjects; do
    for roi in ifgb ifgl ifgr; do
        imcp \
            "${roi}_prob" \
            "${outdir}/${subject}/anat/${subject}_space-${space}_label-${roi}_probseg"
    done
done
