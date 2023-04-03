SUBJECT_ID="sub-0${1}"
PROJECT_ID=project-0b-pRF-tutorial-3T

WHICHCOREG=func2struct

echo "Affine coregistration for subject $SUBJECT_ID"

PROJECT_DIR=/home/mayajas/scratch/$PROJECT_ID
FS_DIR=$PROJECT_DIR/data_FS/$SUBJECT_ID/


T1_IMG=$FS_DIR/mri/T1.mgz
INPLANE_IMG=$PROJECT_DIR/raw/$SUBJECT_ID/anat/inplane.nii
MEANFUNC_IMG=$PROJECT_DIR/output/func/meanFunc/_subject_id_${SUBJECT_ID}/merged_func_mcf.nii_mean_reg.nii
#merged_func_mean.nii

SUBDIR=$PROJECT_DIR/output/coreg/${SUBJECT_ID}
OUTDIR=$SUBDIR/${WHICHCOREG}

# make coreg output directory
if [ ! -d "$OUTDIR" ]; then
  mkdir $SUBDIR
  mkdir $OUTDIR
  echo "Making directory ${OUTDIR}..."
else
  echo "Outputs will be saved to ${OUTDIR}..."
fi

# copy images to outdir
if [ ! -f "$SUBDIR/T1.nii" ]; then
    cp ${T1_IMG} $SUBDIR/T1.mgz
    mri_convert $SUBDIR/T1.mgz $SUBDIR/T1.nii
    echo "Copied T1.nii to output directory"
else
    echo "Output directory already contains T1.nii"
fi
if [ ! -f "$SUBDIR/inplane.nii" ]; then
    cp ${INPLANE_IMG} $SUBDIR/inplane.nii
    echo "Copied inplane.nii to output directory"
else
    echo "Output directory already contains inplane.nii"
fi
if [ ! -f "$SUBDIR/meanFunc.nii" ]; then
    cp ${MEANFUNC_IMG} $SUBDIR/meanFunc.nii
    echo "Copied meanFunc.nii to output directory"
else
    echo "Output directory already contains meanFunc.nii"
fi

# # bias correct mean functional
# cd $SUBDIR
# N4BiasFieldCorrection -d 3 -v 1 -s 4 -b [ 180 ] -c [ 50x50x50x50, 0.0 ] \
#   -i $SUBDIR/meanFunc.nii -o [ corrected_meanFunc.nii, meanFunc_BiasField.nii ]

cd $OUTDIR
if [ "$WHICHCOREG" = "func2struct" ]; then
    # coreg meanFunc to inplane
    PREFIX=FUNC2INPLANE
    ITKSNAP_TRANSFORM=$OUTDIR/coreg_itksnap_${PREFIX}.txt
    REF_IMG=$SUBDIR/inplane.nii
    MOVING_IMG=$SUBDIR/corrected_meanFunc.nii
    MANUAL_MASK=$SUBDIR/occ.nii
    antsRegistration --dimensionality 3 --float 1 \
        --initial-moving-transform [ $ITKSNAP_TRANSFORM, 0 ] \
        --initialize-transforms-per-stage 0 --interpolation Linear --output [ reg${PREFIX}_, reg${PREFIX}_Warped.nii.gz, reg${PREFIX}_InverseWarped.nii.gz ] \
        --transform Rigid[ 0.1 ] \
        --metric MI[ $REF_IMG, $MOVING_IMG, 1, 32, Regular, 0.25 ] \
        --convergence [ 1000x500x250x100, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 0 \
        --masks [ ${MANUAL_MASK}, NULL ] \
        --transform Affine[ 0.1 ] \
        --metric MI[ $REF_IMG, $MOVING_IMG, 1, 32, Regular, 0.25 ] \
        --convergence [ 1000x500x250x100, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 0 \
        --masks [ ${MANUAL_MASK}, NULL ] \
        --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 --collapse-output-transforms 0

    # coreg inplane to T1
    PREFIX=INPLANE2T1
    ITKSNAP_TRANSFORM=$OUTDIR/coreg_itksnap_${PREFIX}.txt
    REF_IMG=$SUBDIR/T1.nii
    MOVING_IMG=$SUBDIR/inplane.nii
    antsRegistration --dimensionality 3 --float 1 \
            --initial-moving-transform [ $ITKSNAP_TRANSFORM, 0 ] \
            --initialize-transforms-per-stage 0 --interpolation Linear --output [ reg${PREFIX}_, reg${PREFIX}_Warped.nii.gz, reg${PREFIX}_InverseWarped.nii.gz ] \
            --transform Rigid[ 0.1 ] \
            --metric MI[ $REF_IMG, $MOVING_IMG, 1, 32, Regular, 0.25 ] \
            --convergence [ 1000x500x250x100, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 0 \
            --transform Affine[ 0.1 ] \
            --metric MI[ $REF_IMG, $MOVING_IMG, 1, 32, Regular, 0.25 ] \
            --convergence [ 1000x500x250x100, 1e-06, 10 ] --smoothing-sigmas 3.0x2.0x1.0x0.0 --shrink-factors 8x4x2x1 --use-histogram-matching 0 \
            --winsorize-image-intensities [ 0.005, 0.995 ]  --write-composite-transform 1 --collapse-output-transforms 0


    # apply transform to inplane
    MOVING_IMG=$SUBDIR/inplane.nii
    REF_IMG=$SUBDIR/inplane.nii
    OUTPUT_IMG=$OUTDIR/reg_inplane.nii
    antsApplyTransforms --default-value 0 \
        -d 3 \
        --input $MOVING_IMG \
        --interpolation Linear \
        --output $OUTPUT_IMG \
        --reference-image $REF_IMG \
        --transform $OUTDIR/regINPLANE2T1_Composite.h5

    # apply both transforms to meanFunc (keep meanFunc resolution)
    MOVING_IMG=$SUBDIR/meanFunc.nii
    REF_IMG=$SUBDIR/meanFunc.nii
    OUTPUT_IMG=$OUTDIR/reg_meanFunc.nii
    antsApplyTransforms --default-value 0 \
        -d 3 \
        --input $MOVING_IMG \
        --interpolation Linear \
        --output $OUTPUT_IMG \
        --reference-image $REF_IMG \
        --transform $OUTDIR/regFUNC2INPLANE_Composite.h5 \
        --transform $OUTDIR/regINPLANE2T1_Composite.h5

fi


