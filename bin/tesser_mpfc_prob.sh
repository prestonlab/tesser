#!/bin/bash
#
# Create a mask of mPFC.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_mpfc_prob.sh outdir space"
    exit 1
fi

outdir=$1
space=$2

# get a list of the subjects
cd "${outdir}" || exit 1
subjects=""
for d in sub-*/; do
    subject=${d%/}
    subjects="${subjects} ${subject}"
done

mpfc_1mm=${outdir}/template/mni/tpl-MNI152NLin2009cAsym_res-01_desc-mpfc_mask.nii.gz
for subject in $subjects; do
    anatdir=${outdir}/${subject}/anat
    antsApplyTransforms -d 3 -e 0 \
        -i "${mpfc_1mm}" \
        -r "${anatdir}/${subject}_space-${space}_desc-brain_mask.nii.gz" \
        -o "${anatdir}/${subject}_space-${space}_desc-mpfc_mask.nii.gz" \
        -n MultiLabel
    fslmaths \
        "${anatdir}/${subject}_space-${space}_desc-mpfc_mask.nii.gz" -mul \
        "${anatdir}/${subject}_space-${space}_label-gray_probseg.nii.gz" \
        "${anatdir}/${subject}_space-${space}_label-mpfc_probseg.nii.gz"
done
