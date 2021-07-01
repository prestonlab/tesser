#!/bin/bash
#
# Pull Tesser fMRIprep results.

if [[ $# -lt 2 ]]; then
    echo "Usage:   pull_tesser_fmriprep.sh src dest [rsync flags]"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include="sub-*/" \
    --include="sub-*/figures/" \
    --include="*.html" \
    --include="*.svg" \
    --exclude="*" \
    "$@"
