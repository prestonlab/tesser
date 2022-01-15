#!/bin/bash
#
# Pull source data from raw data directory.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_pull_sourcedata.sh src dest [rsync flags]"
    exit 1
fi

src=$1
dest=$2
shift 2

rsync -azvu "$src" "$dest" \
    --include=Data \
    --include="Data/tesserScan_*" \
    --include=TesserScan/ \
    --include=TesserScan/rsa_allevents_info/ \
    --include="tesserScan_*.txt" \
    --include="tesserScan_*.mat" \
    --include="tesser_*_run?_info.txt" \
    --exclude="*" \
    "$@"
