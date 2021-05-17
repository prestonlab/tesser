#!/bin/bash
#
# Pull Tesser RSA betaseries and masks.

if [[ $# -lt 2 ]]; then
    echo "Usage:   pull_tesser_betaseries.sh src dest [rsync flags]"
    echo "Example: pull_tesser_betaseries.sh /corral-repl/utexas/prestonlab/tesser/ $SCRATCH/tesser"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include="tesser_*/" \
    --include="tesser_*/anatomy/" \
    --include="tesser_*/anatomy/antsreg/" \
    --include="tesser_*/anatomy/antsreg/data/" \
    --include="tesser_*/anatomy/antsreg/data/funcunwarpspace/" \
    --include="tesser_*/anatomy/antsreg/data/funcunwarpspace/rois/" \
    --include="tesser_*/anatomy/antsreg/data/funcunwarpspace/rois/freesurfer/" \
    --include="tesser_*/anatomy/antsreg/data/funcunwarpspace/rois/mni/" \
    --include={highres,coronal_mean}.nii.gz \
    --include={10m,10p,10r,11m,14c,14r,24,25,32pl}.nii.gz \
    --exclude="*" \
    "$@"
