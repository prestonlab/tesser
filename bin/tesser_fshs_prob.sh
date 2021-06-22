#!/bin/bash
#
# Probabilistic hippocampus mask from FreeSurfer hippocampal segmentation.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_fshs_prob.sh studydir space outdir"
    exit 1
fi

studydir=$1
space=$2
outdir=$3

# get a list of the subjects
cd "${outdir}" || exit 1
subjects=""
for d in sub-*/; do
    subject=${d%/}
    subjects="${subjects} ${subject}"
done

tempdir=${SCRATCH}/fshs_prob
hipl_files=''
hipr_files=''
hipb_files=''
for subject in $subjects; do
    mkdir -p "${tempdir}/${subject}"
    cd "${tempdir}/${subject}" || exit 1
    anatdir=${outdir}/${subject}/anat
    subjno=${subject/sub-/}

    # FreeSurfer hippocampal segmentation output
    mri=${studydir}/tesser_${subjno}/anatomy/tesser_${subjno}/mri
    l_hbt=${mri}/lh.hippoAmygLabels-T1-T2.v21.HBT.mgz
    r_hbt=${mri}/lh.hippoAmygLabels-T1-T2.v21.HBT.mgz

    # convert to nifti
    mri_convert "$l_hbt" l_hbt_fsnative.nii.gz
    mri_convert "$r_hbt" r_hbt_fsnative.nii.gz
    fslreorient2std l_hbt_fsnative l_hbt_fsnative
    fslreorient2std r_hbt_fsnative r_hbt_fsnative

    # transform to template space
    for hemi in l r; do
        antsApplyTransforms -d 3 -e 0 \
            -i "${hemi}_hbt_fsnative.nii.gz" \
            -r "${anatdir}/${subject}_space-${space}_desc-brain_mask.nii.gz" \
            -o "${hemi}_hbt.nii.gz" \
            -n MultiLabel \
            -t "${anatdir}/${subject}_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5" \
            -t "${anatdir}/${subject}_from-fsnative_to-T1w_mode-image_xfm.txt"
    done

    # pool over head/body/tail ROIs
    fslmaths l_hbt -thr 1 -bin hipl
    fslmaths r_hbt -thr 1 -bin hipr
    fslmaths hipl -add hipr -bin hipb
    hipl_files="${hipl_files} ${tempdir}/${subject}/hipl"
    hipr_files="${hipr_files} ${tempdir}/${subject}/hipr"
    hipb_files="${hipb_files} ${tempdir}/${subject}/hipb"
done

# average subject images to create probabilistic masks
cd "${tempdir}" || exit 1
fslmerge -t hipl_all ${hipl_files}
fslmaths hipl_all -Tmean hipl_prob
fslmerge -t hipr_all ${hipr_files}
fslmaths hipr_all -Tmean hipr_prob
fslmerge -t hipb_all ${hipb_files}
fslmaths hipb_all -Tmean hipb_prob

# copy to standard files in the output directory
for subject in $subjects; do
    for roi in hipb hipl hipr; do
        imcp \
            "${roi}_prob" \
            "${outdir}/${subject}/anat/${subject}_space-${space}_label-${roi}_probseg"
    done
done
