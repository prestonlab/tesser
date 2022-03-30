#!/bin/bash
#
# Deface anatomical scans.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_deface.sh src dest"
    exit 1
fi

src=$1
dest=$2

for subj_dir in ${src}/sub-*; do
    subject=$(basename "${subj_dir}")
    mkdir -p "${dest}/${subject}/anat"
    mkdir -p "${dest}/${subject}/fmap"

    # T1w
    filepath=${subject}/anat/${subject}_T1w.nii.gz
    if [[ ! -f ${src}/${filepath} ]]; then
        echo "Warning: No T1w image found for ${subject} in ${src}."
    else
        pydeface "${src}/${filepath}" --outfile "${dest}/${filepath}" --force
    fi

    # fieldmap magnitude
    for run in 1 2; do
        for scan in 1 2; do
            filepath=${subject}/fmap/${subject}_run-${run}_magnitude${scan}.nii.gz
            if [[ ! -f ${src}/${filepath} ]]; then
                echo "Warning: No run ${run} magnitude image ${scan} found for ${subject} in ${src}."
            else
                pydeface "${src}/${filepath}" --outfile "${dest}/${filepath}" --force
            fi
        done
    done
done
