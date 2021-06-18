#!/bin/bash
#
# Estimate smoothness based on a spatial autocorrelation function (ACF).

if [[ $# -lt 3 ]]; then
    echo "Usage: tesser_acf.sh resdir subject nthreads"
    exit 1
fi

resdir=$1
subject=$2
nthreads=$3

export OMP_NUM_THREADS=$nthreads
base=${resdir}/sub-${subject}/sub-${subject}
mask=${base}_mask.nii.gz

for run in 1 2 3 4 5 6; do
    resid=${base}_run-${run}_resid.nii.gz
    output=${base}_run-${run}_smoothness.tsv
    temp=${base}_run-${run}_temp.tsv

    if [[ ! -f $resid ]]; then
        echo "Error: residuals file does not exist: $resid"
        exit 1
    fi

    if 3dFWHMx -mask "${mask}" -acf "${temp}" -input "${resid}" -arith > "${output}"; then
        rm "${temp}"
    else
        echo "Problem running 3dFWHMx. Error code: $?"
    fi
done
