#!/bin/bash
#
# Calculate a group-level cortex probability image.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_gray_prob.sh prepdir space outdir"
    exit 1
fi

prepdir=$1
space=$2
outdir=$3

# get a list of the subjects
cd "${prepdir}" || exit 1
subjects=""
for d in sub-*/; do
    subject=${d%/}
    subjects="${subjects} ${subject}"
done

# create gray matter masks from FreeSurfer output
tempdir=${SCRATCH}/gray_prob
files=""
for subject in ${subjects}; do
    mkdir -p "${tempdir}/${subject}"
    cd "${tempdir}/${subject}" || exit 1

    seg="${prepdir}/${subject}/${subject}_task-struct_run-1_space-${space}_desc-aparcaseg_dseg"
    imcp "${seg}" parcels

    fslmaths parcels -thr 1000 -uthr 1035 -bin l_ctx
    fslmaths parcels -thr 2000 -uthr 2035 -bin r_ctx

    fslmaths parcels -thr 9 -uthr 13 -bin l_subco
    fslmaths parcels -thr 17 -uthr 17 -bin l_hip
    fslmaths parcels -thr 18 -uthr 18 -add l_subco -add l_hip -bin l_subco
    fslmaths parcels -thr 48 -uthr 54 -bin r_subco

    fslmaths l_subco -add r_subco -add l_ctx -add r_ctx -bin b_gray

    files="${files} ${tempdir}/${subject}/b_gray.nii.gz"
done

# average subject images to create a probabilistic mask
cd "${tempdir}" || exit 1
fslmerge -t b_gray_all ${files}
fslmaths b_gray_all -Tmean b_gray_prob
for subject in $subjects; do
    imcp b_gray_prob "${outdir}/${subject}/anat/${subject}_space-${space}_label-gray_probseg"
done
