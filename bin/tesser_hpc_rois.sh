#!/bin/bash
#
# Consolidate hippocampal body and tail to make a posterior hippocampus ROI.

subject=$1
roi_dir="$STUDYDIR/$subject/anatomy/antsreg/data/funcunwarpspace/rois/mni"
cd "$roi_dir" || exit

echo "Consolidating ROIs..."
fslmaths b_hip_body -add b_hip_tail -bin b_hip_post
fslmaths r_hip_body -add r_hip_tail -bin r_hip_post
fslmaths l_hip_body -add l_hip_tail -bin l_hip_post
