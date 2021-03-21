#!/bin/bash
#
# Pull Tesser RSA results for local analysis.

if [[ $# -lt 2 ]]; then
    echo "Usage: pull_tesser_rsa.sh src dest [rsync flags]"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include="batch/" \
    --include="batch/rsa/" \
    --include="batch/rsa/*/" \
    --include="batch/rsa/*/*/" \
    --include="batch/prsa/" \
    --include="batch/prsa/*/" \
    --include="batch/prsa/*/*/" \
    --include="*.npz" \
    --include="*.csv" \
    --exclude="*" "$@"
