#!/bin/bash
#
# Consolidate mPFC ROIs to make three anterior/posterior ROIs.

subject=$1
roi_dir="$STUDYDIR/$subject/anatomy/antsreg/data/funcunwarpspace/rois/mni"
cd "$roi_dir" || exit

echo "Consolidating ROIs..."
fslmaths 25 -add 32pl -add 14c -add 24 -add 10m -add 14r -bin pmpfc
fslmaths 10r -add 11m -bin mmpfc
imcp 10p ampfc
