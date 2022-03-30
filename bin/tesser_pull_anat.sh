#!/bin/bash
#
# Pull anatomical images from a BIDS directory.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_pull_anat.sh src dest [rsync flags]"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include="sub-???" \
    --include="sub-???/anat/" \
    --include="sub-???/fmap/" \
    --include="sub-???_T1w.nii.gz" \
    --include="sub-???_run-?_magnitude?.nii.gz" \
    --exclude="*" \
    "$@"
