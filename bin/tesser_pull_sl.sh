#!/bin/bash
#
# Pull searchlight results.

if [[ $# -lt 2 ]]; then
    echo "Usage: pull_tesser_sl.sh src dest [rsync flags]"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include=beta/ \
    --include="beta/smooth?mm/" \
    --include="beta/smooth?mm/*/" \
    --include="beta/smooth?mm/*/community*/" \
    --include="beta/smooth?mm/*/community*/clusters/" \
    --include="*.txt" \
    --include="*.npy" \
    --include="*.tsv" \
    --include="*.nii.gz" \
    --exclude="*" \
    --prune-empty-dirs \
    "$@"
