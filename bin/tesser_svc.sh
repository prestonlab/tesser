#!/bin/bash
#
# Run small volume correction for a searchlight.

if [[ $# -lt 2 ]]; then
    echo "Usage: tesser_svc.sh [-o] betadir contrast"
    exit 1
fi

overwrite=false
while getopts ":ot:" opt; do
    case $opt in
        o)
            overwrite=true
            ;;
        *)
            echo "Invalid option: $opt"
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

betadir=$1
contrast=$2
template=$betadir/template.nii.gz

# fixed voxelwise alpha because we just have a thresholded image, not
# a zscore image
alpha=0.01

clustsimdir=$betadir/clustsim
if [[ ! -d $clustsimdir ]]; then
    echo "ClustSim dir does not exist: $clustsimdir"
    exit 1
fi

statdir=$betadir/$contrast
if [[ ! -d $statdir ]]; then
    echo "Statistics image dir does not exist: $statdir"
    exit 1
fi
echo "Processing files in: $statdir"
imcp "$betadir/sub-100/sub-100_mask" "$statdir/mask"

# get cluster extent based on the existing clustsim results, based on
# the residuals from the betaseries model that the searchlight was
# based on
cfile=$clustsimdir/clustsim.NN3_1sided.1D
clust_extent=$(grep "^ $alpha" < "$cfile" | awk '{ print $3 }')
cd "$statdir" || exit
echo "Minimum cluster extent: $clust_extent"
echo "$clust_extent" > clust_thresh

recalc=false
if [[ $overwrite = true ]]; then
    recalc=true
fi
if [[ $(imtest zstat_vox_p_tstat1) = 1 && $(imtest vox_mask) = 0 ]]; then
   recalc=true
fi

if [[ $recalc = true ]]; then
    # have results from randomise, which have not been
    # thresholded. Apply the voxelwise alpha
    echo "Recalculating thresholded images..."

    # significant voxels
    thresh=$(python -c "print(1 - $alpha)")
    fslmaths zstat_vox_p_tstat1 -thr "$thresh" -bin vox_mask

    # corresponding z-stat image (for display)
    fslmaths zstat_vox_p_tstat1 -mul -1 -add 1 -ptoz zstat1

    # thresholded z-stat
    fslmaths zstat1 -mas vox_mask cope1

    # pull a copy of the template image
    ln -sf "$template" bg_image.nii.gz

    # get uncorrected clusters of at least minimal size (useful for
    # defining masks, getting cluster size)
    "$FSLDIR/bin/cluster" -i vox_mask -t 0.5 --minextent=10 \
        -o cluster_mask10 > cluster10.txt
fi

# report corrected clusters
fslmaths cope1 -mas mask thresh_cope1
"$FSLDIR/bin/cluster" -i thresh_cope1 -c cope1 -t 0.0001 \
    --minextent="$clust_extent" --othresh=cluster_thresh_cope1 \
    -o cluster_mask_cope1 --connectivity=26 --mm \
    --olmax=lmax_cope1_std.txt --scalarname=Z > cluster_cope1_std.txt

range=$(fslstats cluster_thresh_cope1 -l 0.0001 -R 2>/dev/null)
low=$(echo "$range" | awk '{print $1}')
high=$(echo "$range" | awk '{print $2}')
echo "Rendering using zmin=$low zmax=$high"

imcp "$statdir/bg_image" example_func
overlay 1 0 example_func -a cluster_thresh_cope1 "$low" "$high" rendered_thresh_cope1
slicer rendered_thresh_cope1 -S 2 750 rendered_thresh_cope1.png
