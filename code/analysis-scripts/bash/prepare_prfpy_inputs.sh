SUBJECT_ID="sub-0${1}"
PROJECT_ID=project-0b-pRF-tutorial-3T
PRF_MODEL=prfpy

echo "Preparing prfpy inputs for subject $SUBJECT_ID"

###########################################################################################################################
## directories, filenames, important variables
PROJECT_DIR=/home/mayajas/scratch/$PROJECT_ID
FS_DIR=$PROJECT_DIR/data_FS

SUBDIR=$PROJECT_DIR/output/coreg/${SUBJECT_ID}
COREG_DIR=$SUBDIR/func2struct
OUT_DIR=/home/mayajas/scratch/$PROJECT_ID/output/${PRF_MODEL}/$SUBJECT_ID

MEANFUNC_IMG=$PROJECT_DIR/output/func/meanFunc/_subject_id_${SUBJECT_ID}/merged_func_mcf.nii_mean_reg.nii
BAR1_IMG=$PROJECT_DIR/output/func/sliceTimeCorr/_subject_id_$SUBJECT_ID/_sess_id_task-bar_run-01_sess_nr_0_sess_nvol_168/atask-bar_run-01_roi_warp4D.nii
BAR2_IMG=$PROJECT_DIR/output/func/sliceTimeCorr/_subject_id_$SUBJECT_ID/_sess_id_task-bar_run-02_sess_nr_1_sess_nvol_168/atask-bar_run-02_roi_warp4D.nii
BAR3_IMG=$PROJECT_DIR/output/func/sliceTimeCorr/_subject_id_$SUBJECT_ID/_sess_id_task-bar_run-03_sess_nr_2_sess_nvol_168/atask-bar_run-03_roi_warp4D.nii

#UNI_IMG=$PROJECT_DIR/output/anat/_subject_id_$SUBJECT_ID/UNI_corrected.nii
T1_IMG=$SUBDIR/T1.nii
OCC_MASK=$SUBDIR/occ.nii

rGM_val=42
rWM_val=41
lGM_val=3
lWM_val=2

###########################################################################################################################
## make output directories, copy needed files
# make prfpy general output directory
if [ ! -d "$PROJECT_DIR/output/${PRF_MODEL}" ]; then
  mkdir $PROJECT_DIR/output/${PRF_MODEL}
fi

# make prfpy subject output directory
if [ ! -d "$OUT_DIR" ]; then
  mkdir $OUT_DIR
  echo "Making directory ${OUT_DIR}..."
else
  echo "Outputs will be saved to ${OUT_DIR}..."
fi

# cp $FS_DIR/$SUBJECT_ID/mri/ribbon.mgz $OUT_DIR/ribbon.mgz
# cp $MEANFUNC_IMG $OUT_DIR/meanFunc.nii
# cp $T1_IMG $OUT_DIR/T1_out.nii
# cp $BAR1_IMG $OUT_DIR/bar1.nii
# cp $BAR2_IMG $OUT_DIR/bar2.nii
# cp $BAR3_IMG $OUT_DIR/bar3.nii

# # ###########################################################################################################################
# # # coregister mean functional & bar run volumes to func resolution
# antsApplyTransforms --default-value 0 \
#     --input $OUT_DIR/meanFunc.nii \
#     --interpolation BSpline[5] \
#     --output $OUT_DIR/reg_meanFunc.nii \
#     --reference-image $OUT_DIR/T1_out.nii \
#     --transform $COREG_DIR/regFUNC2INPLANE_Composite.h5 \
#     --transform $COREG_DIR/regINPLANE2T1_Composite.h5

# antsApplyTransforms --default-value 0 \
#     -e 3 \
#     --input $OUT_DIR/bar1.nii \
#     --interpolation BSpline[5] \
#     --output $OUT_DIR/reg_bar1.nii \
#     --reference-image $OUT_DIR/meanFunc.nii \
#     --transform $COREG_DIR/regFUNC2INPLANE_Composite.h5 \
#     --transform $COREG_DIR/regINPLANE2T1_Composite.h5

# antsApplyTransforms --default-value 0 \
#     -e 3 \
#     --input $OUT_DIR/bar2.nii \
#     --interpolation BSpline[5] \
#     --output $OUT_DIR/reg_bar2.nii \
#     --reference-image $OUT_DIR/meanFunc.nii \
#     --transform $COREG_DIR/regFUNC2INPLANE_Composite.h5 \
#     --transform $COREG_DIR/regINPLANE2T1_Composite.h5

# antsApplyTransforms --default-value 0 \
#     -e 3 \
#     --input $OUT_DIR/bar3.nii \
#     --interpolation BSpline[5] \
#     --output $OUT_DIR/reg_bar3.nii \
#     --reference-image $OUT_DIR/meanFunc.nii \
#     --transform $COREG_DIR/regFUNC2INPLANE_Composite.h5 \
#     --transform $COREG_DIR/regINPLANE2T1_Composite.h5   

# ###########################################################################################################################
# ## surface project bar data
# SUBJECTS_DIR=$FS_DIR

# # lh
# mri_vol2surf --mov $OUT_DIR/reg_meanFunc.nii  --regheader $SUBJECT_ID --hemi lh --out $OUT_DIR/lh.meanFunc.mgh --interp trilinear --projfrac 0 --cortex

# # rh
# mri_vol2surf --mov $OUT_DIR/reg_meanFunc.nii  --regheader $SUBJECT_ID --hemi rh --out $OUT_DIR/rh.meanFunc.mgh --interp trilinear --projfrac 0 --cortex


# # lh
# mri_vol2surf --mov $OUT_DIR/reg_bar1.nii  --regheader $SUBJECT_ID --hemi lh --out $OUT_DIR/lh.bar1.mgh --interp trilinear --projfrac 0 --cortex
# mri_vol2surf --mov $OUT_DIR/reg_bar2.nii  --regheader $SUBJECT_ID --hemi lh --out $OUT_DIR/lh.bar2.mgh --interp trilinear --projfrac 0 --cortex
# mri_vol2surf --mov $OUT_DIR/reg_bar3.nii  --regheader $SUBJECT_ID --hemi lh --out $OUT_DIR/lh.bar3.mgh --interp trilinear --projfrac 0 --cortex

# # rh
# mri_vol2surf --mov $OUT_DIR/reg_bar1.nii  --regheader $SUBJECT_ID --hemi rh --out $OUT_DIR/rh.bar1.mgh --interp trilinear --projfrac 0 --cortex
# mri_vol2surf --mov $OUT_DIR/reg_bar2.nii  --regheader $SUBJECT_ID --hemi rh --out $OUT_DIR/rh.bar2.mgh --interp trilinear --projfrac 0 --cortex
# mri_vol2surf --mov $OUT_DIR/reg_bar3.nii  --regheader $SUBJECT_ID --hemi rh --out $OUT_DIR/rh.bar3.mgh --interp trilinear --projfrac 0 --cortex


###########################################################################################################################
## make occipital label from volumetric (manual) mask
SUBJECTS_DIR=$FS_DIR

# lh
mri_vol2surf --mov $OCC_MASK --regheader $SUBJECT_ID --hemi lh --out $OUT_DIR/lh.occMask.mgh --interp nearest --projfrac 0 --cortex
#mri_vol2label --i $OUT_DIR/lh.occMask.mgh --id 1 --l $OUT_DIR/lh.occMask.label --surf $SUBJECT_ID lh

# rh
mri_vol2surf --mov $OCC_MASK --regheader $SUBJECT_ID --hemi rh --out $OUT_DIR/rh.occMask.mgh --interp nearest --projfrac 0 --cortex
#mri_vol2label --i $OUT_DIR/rh.occMask.mgh --id 1 --l $OUT_DIR/rh.occMask.label --surf $SUBJECT_ID rh