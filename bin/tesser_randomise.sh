#!/bin/bash
#
# Run randomise to test z-statistic images.

if [[ $# -lt 4 ]]; then
    echo "tesser_randomise.sh resdir contrast resname n_perm"
    exit 1
fi

resdir=$1
contrast=$2
resname=$3
n_perm=$4

mkdir -p "${resdir}/${resname}"
fslmerge -t "${resdir}/desc-${contrast}_zstat.nii.gz" \
    "${resdir}/${resname}"/zstat.nii.gz

randomise -i "${resdir}/${resname}/zstat.nii.gz" \
    -o "${resdir}/${resname}/zstat" \
    -1 -n "$n_perm" -x --uncorrp
