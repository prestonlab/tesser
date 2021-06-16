#!/bin/bash
#
# Transform template ROIs to native resolution template space.

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_fshs_rois.sh templatedir outdir subject"
    exit 1
fi

templatedir=$1
outdir=$2
subject=$3
space=$4

subjdir=${outdir}/sub-${subject}
reference_name=task-struct_run-1_space-${space}_desc-brain_mask
reference_file=${subjdir}/func/sub-${subject}_${reference_name}.nii.gz

# brain mask
name=desc-brain_mask
template_file=${templatedir}/tpl-${space}/tpl-${space}_res-01_${name}.nii.gz
if [[ ! -f ${template_file} ]]; then
    echo "Template file not found: ${template_file}"
    exit 1
fi
output_file=${subjdir}/anat/sub-${subject}_space-${space}_${name}.nii.gz
antsApplyTransforms -d 3 -e 0 \
    -i "${template_file}" \
    -r "${reference_file}" \
    -o "${output_file}" \
    -n NearestNeighbor

# probabilistic labels
for name in label-{brain,GM,WM,CSF}_probseg; do
    template_file=${templatedir}/tpl-${space}/tpl-${space}_res-01_${name}.nii.gz
    if [[ ! -f ${template_file} ]]; then
        echo "Template file not found: ${template_file}"
        exit 1
    fi
    output_file=${subjdir}/anat/sub-${subject}_space-${space}_${name}.nii.gz
    antsApplyTransforms -d 3 -e 0 \
        -i "${template_file}" \
        -r "${reference_file}" \
        -o "${output_file}" \
        -n Linear
done
