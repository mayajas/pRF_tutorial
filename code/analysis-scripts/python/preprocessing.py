#!/usr/bin/env python
# coding: utf-8

# # fMRI processing pipeline: functional processing for pRF mapping
# 
# This workflow is run in parallel to recon-all on the T1-MPRAGE anatomical image.
# 
# The functional processing pipeline aims to follow the steps followed in fMRIPrep, adapted for pRF mapping. The steps implemented in the pipeline are as follows:
# - discard initial fMRI volumes to allow for T1 equilibration
# - realignment: head-motion estimation and correction (FSL MCFLIRT); within and between sessions
# - susceptibility-derived distortion estimation and unwarping (FUGUE)
# - slice-timing correction (SPM)
# - co-registration of functional and structural data (ANTs)
# 

# ### Set preferences
# Whether or not to write the workflow viz graph, run pipeline, run specific branches of workflow...

# 

# In[1]:


# whether or not to run the pipeline
run_pipeline = True   

# whether or not to write workflow graph (svg)
write_graph  = True                           
                        
# whether manual edits exist (for coregistration)
manual_edits = False      

# whether to do unwarping
unwarp = False   
precalc_fmap = False # if fmap has been precalculated outside pipeline

# whether to apply coregistration
coregister = False
precalc_coreg = False # if coregistration transform has been precalculated outside pipeline

# coregistration method: 'flirt','freesurfer', 'antsRegistration' or 'itk-snap'
coreg_method = 'antsRegistration' 

# coregistration direction: either from functional to structural ('func2struct') or vice versa ('struct2func')
coreg_dir = 'func2struct'

# number of cores to use: either set explicitly or based on settings in batch file
import os
n_procs = 1
# n_procs = int(os.getenv('OMP_NUM_THREADS'))   
print(n_procs)

# field map method (https://lcni.uoregon.edu/kb-articles/kb-0003)
# Method 1 calculates a field map based on the difference in phase 
# between two different echos in a double echo sequence. 
# Method 2 uses two separate acquisitions with opposite phase encoding 
# directions to calculate a field map based on the difference in 
# distortion between the two acquisitions.
fmap_method = 1 


# ### Set paths
# All computer-dependent paths

# In[2]:


from os.path import join as opj

local = False
if local:
    doc_dir = '/home/mayajas/Documents'
    data_dir = '/home/mayajas/scratch/project-0b-pRF-tutorial-3T/'
else:
    doc_dir = '/home/mayaaj90/'
    data_dir = '/scratch/mayaaj90/project-0b-pRF-tutorial-3T/'


# general software directory
software_dir = opj(doc_dir,'programs')

# SPM dir
spm_dir = opj(software_dir,'spm12')

# data directory
raw_data_dir = opj(data_dir,'raw')

# scripts directory
der_dir = opj(data_dir,'derivatives')

# dicoms directory
dicom_dir = opj(data_dir,'dicoms')

# pRF directory
pRF_dir = opj(data_dir,'output','prfpy')

# output directory for datasink
out_dir = opj(data_dir,'output')

# FS output from anatomy pipeline
subjects_dir = opj(data_dir,'data_FS')
os.environ['SUBJECTS_DIR']=subjects_dir


# In[3]:


os.environ['SUBJECTS_DIR']


# ### Imports
# Import required libraries, set paths

# In[4]:


import re

from nipype.interfaces.io import DataGrabber, DataSink

from nipype import Node, MapNode, JoinNode, Workflow

from nipype.interfaces.freesurfer import MRIConvert, SampleToSurface

#import nipype_settings
import nipype.interfaces.matlab as Matlab
from nipype.interfaces.base import (
    CommandLineInputSpec,
    CommandLine,
    TraitedSpec,
    BaseInterface, 
    BaseInterfaceInputSpec, 
    File,
    Directory
)

from string import Template

from nipype.interfaces.utility import Function, IdentityInterface, Select
from nipype.interfaces.utility import Merge as utilMerge
from nipype.interfaces.utility import Select as utilSelect
from nipype.interfaces.base import BaseInterface, BaseInterfaceInputSpec, CommandLine, Directory, File, TraitedSpec, traits

from nipype import config
config.enable_debug_mode()

from os.path import abspath

from nipype.interfaces.freesurfer.model import Binarize

import pygraphviz 

from pydicom.data import get_testdata_file
from pydicom import dcmread

from nipype.interfaces.fsl import BET, PrepareFieldmap, ExtractROI, MCFLIRT, ConvertXFM, FLIRT
from nipype.interfaces.fsl import Merge as fslMerge
from nipype.interfaces.fsl import Split as fslSplit

from nipype.interfaces.freesurfer.registration import MRICoreg
from nipype.interfaces.freesurfer.preprocess import BBRegister

from nipype.interfaces.dcmstack import MergeNifti

import sys
sys.path.append(software_dir)
from ApplyXfm4D import ApplyXfm4D

from nipype.interfaces.fsl.preprocess import FUGUE

from nipype.interfaces.afni import Warp, TShift

from nipype.interfaces.spm import SliceTiming, Reslice

# import neuropythy as ny

import nipype.interfaces.matlab as Matlab

from nipype.interfaces.ants.registration import Registration
from nipype.interfaces.ants import N4BiasFieldCorrection
from nipype.interfaces.ants import ApplyTransforms

from nipype.interfaces.fsl.maths import BinaryMaths

# set SPM path
os.environ['SPM_PATH']=spm_dir

from nipype.interfaces import spm
spm.SPMCommand.set_mlab_paths(paths=spm_dir)

from nipype.interfaces.fsl import MeanImage
#print(spm.Info.name())
#print(spm.SPMCommand().version)

import scipy.io
import json


# ### Neuropythy configuration
# On startup, neuropythy looks for file ~/.npythyrc. Here, we override these settings and environment variables

# ### Specify important variables

# In[5]:


wf_name = 'wf_func_preproc'

T1_id      = 'T1w.nii'                                                # name of preprocessed T1 (output of structural processing pipeline)
inplane_id = 'inplane.nii'

# manual correction files - from itksnap
coreg_itksnap_struct2func_txt_id = 'coreg_itksnap_struct2func.txt'
# coreg_itksnap_struct2func_mat_id = 'coreg_itksnap_struct2func.mat'
coreg_itksnap_func2struct_txt_id = 'coreg_itksnap_func2struct.txt'
coreg_regINPLANE2T1_id      = 'regINPLANE2T1_Composite.h5'
coreg_regFUNC2INPLANE_id    = 'regFUNC2INPLANE_Composite.h5'

manual_midoccmask_id    = 'midoccMask.nii'
manual_occipitalmask_id = 'occipitalMask.nii'

if fmap_method == 1:
    fmap_magnitude1_id  = 'magnitude1.nii'
    fmap_phasediff_id   = 'phasediff.nii'
    precalc_fmap_id     = 'funcReg_fmap_rads.nii'
#elif fmap_method == 2:
    # implement

    
n_dummy = 8                                                     # number of dummy scans to discard to allow for T1 equilibration

n_vol_bar1 = 168                                                # number of volumes of bar pRF mapping stimulus
n_vol_bar2 = n_vol_bar1                                         # number of volumes of bar pRF mapping stimulus
n_vol_bar3 = n_vol_bar1                                         # number of volumes of bar pRF mapping stimulus

# iterables
# subject_list = ['sub-01','sub-02','sub-03','sub-04']            # subject identifiers
subject_list = ['sub-011']            # subject identifiers
sess_id_list = ['task-bar_run-01', 'task-bar_run-02',           # func session identifiers
             'task-bar_run-03']#, 'task-bar_run-04']
sess_nvol_list = [n_vol_bar1, n_vol_bar2,
                  n_vol_bar3]
sess_nr_list = list(range(0, len(sess_id_list)))


# Get TR/TE/slice timing info

# In[6]:


fpath = opj(data_dir,'raw','sub-011','func','task-bar_run-01.json')
  
# Opening JSON file
f = open(fpath)
  
# returns JSON object as 
# a dictionary
func_json = json.load(f)
func_json


# In[7]:


print(func_json["SliceTiming"])


# In[8]:


# repetition time
TR = func_json["RepetitionTime"]                               # in seconds [s]
TR_str = '%.1fs' % TR

# echo time
TE = func_json["EchoTime"]                                     # in seconds [s]

# MR acquisition type
acquisition_type = func_json["MRAcquisitionType"]

# slice acquisition times
slice_timing = func_json["SliceTiming"]            # in seconds [s]

# number of slices
num_slices = len(slice_timing)

# time of volume acquisition
TA = TR-(TR/num_slices)


# In[9]:


TR


# Get parameters needed for fieldmap correction (see: https://lcni.uoregon.edu/kb-articles/kb-0003)

# In[10]:


fpath = opj(data_dir,'raw','sub-011','fmap','magnitude1.json')
  
# Opening JSON file
f = open(fpath)
  
# returns JSON object as 
# a dictionary
magnitude1_json = json.load(f)
magnitude1_json


# In[11]:


# effective echo spacing 
effective_echo_spacing = func_json["EffectiveEchoSpacing"]

# deltaTE
delta_TE = magnitude1_json["deltaTE"]# in milliseconds [ms]      
                                     # (a float, nipype default value: 2.46) 
                                     # echo time difference of the fieldmap 
                                     # sequence in ms. (usually 2.46ms in Siemens)


# In[12]:


effective_echo_spacing


# ### Create workflow
# About connecting nodes: https://nipype.readthedocs.io/en/0.11.0/users/joinnode_and_itersource.html

# In[13]:


wf = Workflow(name=wf_name, base_dir=der_dir)


# ### Subjects & functional sessions

# In[14]:


subjects = Node(IdentityInterface(fields=['subject_id']),name='subjects')
subjects.iterables = [('subject_id', subject_list)]


# In[15]:


sessions = Node(IdentityInterface(fields=['sess_id','sess_nvol','sess_nr']),name='sessions')
sessions.iterables = [('sess_id', sess_id_list), ('sess_nvol', sess_nvol_list), ('sess_nr', sess_nr_list)]
sessions.synchronize = True


# ### Acquisition parameters

# In[16]:


acquisitionParams = Node(IdentityInterface(fields=['n_dummy', 'TR'
                                                  'TA','TR_str','TE','acquisition_type',
                                                  'slice_timing','num_slices',
                                                  'effective_echo_spacing',
                                                  'delta_TE']),
                         name='acquisitionParams')

acquisitionParams.inputs.n_dummy = n_dummy
acquisitionParams.inputs.TR = TR
acquisitionParams.inputs.TA = TA
acquisitionParams.inputs.TR_str = TR_str
acquisitionParams.inputs.TE = TE
acquisitionParams.inputs.acquisition_type = acquisition_type
acquisitionParams.inputs.slice_timing = slice_timing
acquisitionParams.inputs.num_slices = num_slices
acquisitionParams.inputs.effective_echo_spacing = effective_echo_spacing
acquisitionParams.inputs.delta_TE = delta_TE


# ### Grab data
# 
# DataGrabber is an interface for collecting files from hard drive. It is very flexible and supports almost any file organization of your data you can imagine.
# <br>More info: https://nipype.readthedocs.io/en/0.11.0/users/grabbing_and_sinking.html

# #### Anatomical and field map data

# In[17]:


datasource = Node(DataGrabber(infields=['subject_id'], outfields=['T1', 
                                                                  'fmap_magnitude1', 
                                                                  'fmap_phasediff', 
                                                                  'subject_id']),
                 name='datasource')
datasource.inputs.base_directory = data_dir
datasource.inputs.sort_filelist = False
datasource.inputs.template = '*'
datasource.inputs.field_template = dict(T1='raw/%s/anat/'+T1_id,
                                        fmap_magnitude1='raw/%s/fmap/'+fmap_magnitude1_id,
                                        fmap_phasediff='raw/%s/fmap/'+fmap_phasediff_id
                                       )
datasource.inputs.template_args = dict(T1=[['subject_id']],
                                       fmap_magnitude1=[['subject_id']],
                                       fmap_phasediff=[['subject_id']]
                                       )


# In[18]:


wf.connect([(subjects, datasource,[('subject_id', 'subject_id')])])


# #### Functional data

# In[19]:


datasourceFunc = Node(DataGrabber(infields=['subject_id','sess_id'], outfields=['sess_id', 
                                                                  'subject_id']),
                 name='datasourceFunc')
datasourceFunc.inputs.base_directory = data_dir
datasourceFunc.inputs.sort_filelist = False
datasourceFunc.inputs.template = '*'
datasourceFunc.inputs.field_template = dict(sess_id='raw/%s/func/%s.nii'
                                       )
datasourceFunc.inputs.template_args = dict(sess_id=[['subject_id','sess_id']]
                                       )


# In[20]:


wf.connect([(subjects, datasourceFunc, [('subject_id', 'subject_id')])])
wf.connect([(sessions, datasourceFunc, [('sess_id', 'sess_id')])])


# #### Manual edits
# (if they exist)

# In[21]:


datasourceManualEdits = Node(DataGrabber(infields=['subject_id'], outfields=['coreg_regFUNC2INPLANE',
                                                                             'coreg_regINPLANE2T1',
                                                                             'subject_id']),
                 name='datasourceManualEdits')
datasourceManualEdits.inputs.base_directory = data_dir
datasourceManualEdits.inputs.sort_filelist = False
datasourceManualEdits.inputs.template = '*'
datasourceManualEdits.inputs.field_template = dict(coreg_regFUNC2INPLANE='output/coreg/%s/func2struct/'+coreg_regFUNC2INPLANE_id,
                                                   coreg_regINPLANE2T1='output/coreg/%s/func2struct/'+coreg_regINPLANE2T1_id
                                       )
datasourceManualEdits.inputs.template_args = dict(coreg_regFUNC2INPLANE=[['subject_id']],
                                                  coreg_regINPLANE2T1=[['subject_id']]
                                       )


# In[22]:


if manual_edits: 
    wf.connect([(subjects, datasourceManualEdits, [('subject_id', 'subject_id')])])


# #### Pre-calculated field-map
# (if it exists)

# In[23]:


datasourcePrecalcFmap = Node(DataGrabber(infields=['subject_id'], outfields=['precalc_fmap',
                                                                             'subject_id']),
                 name='datasourcePrecalcFmap')
datasourcePrecalcFmap.inputs.base_directory = data_dir
datasourcePrecalcFmap.inputs.sort_filelist = False
datasourcePrecalcFmap.inputs.template = '*'
datasourcePrecalcFmap.inputs.field_template = dict(precalc_fmap='output/fmap/_subject_id_%s/'+precalc_fmap_id

                                       )
datasourcePrecalcFmap.inputs.template_args = dict(precalc_fmap=[['subject_id']]
                                       )


# In[24]:


if precalc_fmap: 
    wf.connect([(subjects, datasourcePrecalcFmap, [('subject_id', 'subject_id')])])


# ### Calculate field map
# 
# Using fsl_prepare_fieldmap. 
# 
# "If you have data from a SIEMENS scanner then we strongly recommend that the tool fsl_prepare_fieldmap is used to generate the required input data for FEAT or fugue. Fieldmap data from a SIEMENS scanner takes the form of one phase difference image and two magnitude images (one for each echo time). In the following, where a magnitude image is required, pick the "best looking" one. This image is used for registration and masking but the process is not particularly sensitive to the quality and typically either image will work fine.
# 
# Brain extraction of the magnitude image is very important and must be tight - that is, it must exclude all non-brain voxels and any voxels with only a small partial volume contribution. The reason for this is that these areas are normally very noisy in the phase image (look at them in FSLView - if they are not noisy then this is not so important). It is crucial that the mask (derived from this brain extracted image) contains few of these noisy voxels. This is most easily done by making the brain extraction very tight, erring on excluding brain voxels. The exclusion of brain voxels in this instance is actually fine and will have no repercussions, since the fieldmap is extrapolated beyond this mask, and that is the only purpose that the mask plays. Therefore make sure your mask is (if it can't be perfect) too small. As noted above, either magnitude image (from the different echos) can normally be used here - it is not that important." 
# 
# Source: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide#SIEMENS_data

# #### Brain extract magnitude image
# Use magnitude1.nii by default

# In[25]:


# FSL BET - run on magnitude1.nii image
betMagnImg = Node(BET(),name='betMagnImg')


# In[26]:


if unwarp and not precalc_fmap:
    wf.connect([(datasource,betMagnImg,[('fmap_magnitude1','in_file')])])


# #### Prepare field map
# 
# Prepares a fieldmap suitable for FEAT from SIEMENS data - saves output in rad/s format (e.g. `fsl_prepare_fieldmap SIEMENS images_3_gre_field_mapping images_4_gre_field_mapping fmap_rads 2.65`).
# 
# 
# [Mandatory]
# delta_TE: (a float, nipype default value: 2.46)
#         echo time difference of the fieldmap sequence in ms. (usually 2.46ms
#         in Siemens)
#         flag: %f, position: -2
# in_magnitude: (an existing file name)
#         Magnitude difference map, brain extracted
#         flag: %s, position: 3
# in_phase: (an existing file name)
#         Phase difference map, in SIEMENS format range from 0-4096 or 0-8192)
#         flag: %s, position: 2
#         
#         
# https://nipype.readthedocs.io/en/0.12.1/interfaces/generated/nipype.interfaces.fsl.epi.html#preparefieldmap

# In[27]:


if unwarp and not precalc_fmap:
    prepFieldMap = Node(PrepareFieldmap(), name='prepFieldMap')


# In[28]:


if unwarp and not precalc_fmap:
    wf.connect([(acquisitionParams,prepFieldMap,[('delta_TE','delta_TE')])])
    wf.connect([(betMagnImg,prepFieldMap,[('out_file','in_magnitude')])])
    wf.connect([(datasource,prepFieldMap,[('fmap_phasediff','in_phase')])])


# In[29]:


PrepareFieldmap.help()


# ### Discard initial fMRI volumes to allow for T1 equilibration
# 

# In[30]:


discardDummies = Node(ExtractROI(t_min=n_dummy), name='discardDummies')


# In[31]:


wf.connect([(datasourceFunc, discardDummies,[('sess_id', 'in_file')])])
wf.connect([(sessions, discardDummies,[('sess_nvol', 't_size')])])


# In[32]:


ExtractROI.help()


# ### Realignment: head-motion estimation and correction (FSL MCFLIRT)
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/MCFLIRT
# 
# https://nipype.readthedocs.io/en/0.12.1/interfaces/generated/nipype.interfaces.fsl.preprocess.html
# 
# citation: Jenkinson, M., Bannister, P., Brady, J. M. and Smith, S. M. Improved Optimisation for the Robust and Accurate Linear Registration and Motion Correction of Brain Images. NeuroImage, 17(2), 825-841, 2002. 
# 
# First, motion-correction with MCFLIRT, within each session, saving the resulting transformation matrices. Then, concatenate the mean runs from each session and realign to each other with MCFLIRT, saving the transformation matrices. Loop through each matrix in the MCFLIRT output and do 'convert_xfm -omat CONCAT_0000 -concat reg_series1_to_series2.mat MAT_0000' for all MAT* files, then use applyxfm4D, with the "-userprefix CONCAT_" option. This does all transformations at once, directly from the original data and minimizes interpolation effects. Based on: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;21c97ca8.06
# 
# #### Within sessions

# In[33]:


mean_vol  = False             # (a boolean) register to mean volume
save_mats = True              # (a boolean) save transformation parameters
ref_vol   = 1                 # (an integer) volume to align frames to


# In[34]:


mcflirtWithinSess = Node(MCFLIRT(mean_vol = mean_vol, save_mats=save_mats, ref_vol=ref_vol), 
               name='mcflirtWithinSess')


# In[35]:


wf.connect([(discardDummies, mcflirtWithinSess,[('roi_file','in_file')])])


# In[36]:


MCFLIRT.help()


# #### Between sessions
# ##### Take mean of each run

# In[37]:


getMeanImg = Node(MeanImage(dimension='T'),name='getMeanImg')


# In[38]:


wf.connect([(mcflirtWithinSess,getMeanImg,[('out_file','in_file')])])


# In[39]:


MeanImage.help()


# ##### Concatenate mean runs

# In[40]:


dimension = 't'
output_type = 'NIFTI'
merged_file = 'merged_means.nii'


# In[41]:


concatenateMeans = JoinNode(fslMerge(dimension=dimension, output_type=output_type, merged_file=merged_file),
                        joinfield='in_files',
                        joinsource='sessions',
                        name="concatenateMeans")


# In[42]:


# wf.connect([(mcflirtWithinSess, concatenateMeans,[('mean_img', 'in_files')])])
wf.connect([(getMeanImg, concatenateMeans,[('out_file', 'in_files')])])


# ##### MCFLIRT on merged mean runs

# In[43]:


mean_vol = False               # (a boolean) register to mean volume
save_mats = True               # (a boolean) save transformation parameters
ref_vol   = 1                 # (an integer) volume to align frames to


# In[44]:


mcflirtBetweenSess = Node(MCFLIRT(mean_vol = mean_vol, save_mats=save_mats, ref_vol=ref_vol), 
               name='mcflirtBetweenSess')


# In[45]:


wf.connect([(concatenateMeans, mcflirtBetweenSess,[('merged_file','in_file')])])


# In[46]:


MCFLIRT.help()


# ##### Concatenate transformation matrices
# 

# Select given session's transformation mat

# In[47]:


betweenMat = Node(Select(), name='betweenMat')


# In[48]:


wf.connect([(mcflirtBetweenSess, betweenMat, [('mat_file', 'inlist')])])
wf.connect([(sessions, betweenMat, [('sess_nr', 'index')])])


# Concatenate within-session mat_files with corresponding session's between-session realignment mat_file

# In[49]:


concat_xfm = True         # (a boolean) write joint transformation of two input matrices
                          # flag: -concat, position: -3
                          # mutually_exclusive: invert_xfm, concat_xfm, fix_scale_skew
                          # requires: in_file2


# In[50]:


concatenateTransforms = MapNode(ConvertXFM(concat_xfm=concat_xfm),
                            name = 'concatenateTransforms', iterfield=['in_file2'])


# In[51]:


wf.connect([(betweenMat,concatenateTransforms,[('out','in_file')])])
wf.connect([(mcflirtWithinSess,concatenateTransforms,[('mat_file','in_file2')])])


# In[52]:


#ConvertXFM.help()


# Put all transformation matrices for given session in one folder
# 
# (Not the most elegant solution, but ApplyXfm4D requires a directory of tranformation mat files as input)

# In[53]:


def copy_transforms(subject_id,sess_id,sess_nr,sess_nvol,mat_files,working_dir):
    from os.path import join as opj
    import shutil
    import os
    
    transformMatDir = opj(working_dir,'_subject_id_'+subject_id,
                         '_sess_id_'+sess_id+'_sess_nr_'+str(sess_nr)+'_sess_nvol_'+str(sess_nvol),
                         'transformMats')
    
    if not os.path.isdir(transformMatDir):
        os.mkdir(transformMatDir)
    
    for mat in mat_files:
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        print(mat)
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

        # copy file
        shutil.copy(mat, transformMatDir)
        
        # remove .mat extension (this is how the ApplyXfm4D interface likes it)
        base=os.path.basename(mat)
        filename=os.path.splitext(base)[0]
        shutil.move(opj(transformMatDir,filename+'.mat'), opj(transformMatDir,filename)) 

    # session-dependent filename prefix
    prefix = f"MAT_000{sess_nr}_MAT_"

    return transformMatDir, prefix


# In[54]:


copyTransforms = Node(Function(input_names = ['subject_id','sess_id','sess_nr', 'sess_nvol',
                                             'mat_files','working_dir'],
                               output_names=['transformMatDir','prefix'],
                               function=copy_transforms),
                      name='copyTransforms')
copyTransforms.inputs.working_dir = opj(der_dir,wf_name)


# In[55]:


wf.connect([(subjects, copyTransforms, [('subject_id', 'subject_id')])])
wf.connect([(sessions, copyTransforms, [('sess_id', 'sess_id')])])
wf.connect([(sessions, copyTransforms, [('sess_nr', 'sess_nr')])])
wf.connect([(sessions, copyTransforms, [('sess_nvol', 'sess_nvol')])])
wf.connect([(concatenateTransforms, copyTransforms, [('out_file', 'mat_files')])])


# Apply transformation matrices to realign within and between sessions in one step

# In[56]:


applyRealign = Node(ApplyXfm4D(),name='applyRealign')


# In[57]:


wf.connect([(discardDummies,applyRealign,[('roi_file','in_file')])])
wf.connect([(discardDummies,applyRealign,[('roi_file','ref_vol')])])
wf.connect([(copyTransforms,applyRealign,[('transformMatDir','trans_dir')])])
wf.connect([(copyTransforms,applyRealign,[('prefix','user_prefix')])])


# In[58]:


#ApplyXfm4D.help()


# ### Unwarping
# 
# fugue (FMRIB's Utility for Geometrically Unwarping EPIs) performs unwarping of an EPI image based on fieldmap data. The input required consists of the EPI image, the fieldmap (as an unwrapped phase map or a scaled fieldmap in rad/s) and appropriate image sequence parameters for the EPI and fieldmap acquisitions: the dwell time for EPI (also known as the echo spacing); and the echo time difference (called asym time herein). 
# 
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FUGUE/Guide
# 
# https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.fsl.preprocess.html#fugue

# #### Get first dimension of 4D fieldmap image

# In[61]:


splitFieldMapImage = Node(fslSplit(dimension='t'),name='splitFieldMapImage')


# In[62]:


if unwarp and not precalc_fmap:
    wf.connect([(prepFieldMap,splitFieldMapImage,[('out_fieldmap','in_file')])])


# In[63]:


getFieldMap = Node(utilSelect(index=[0]),name='getFieldMap')


# In[64]:


if unwarp and not precalc_fmap:
    wf.connect([(splitFieldMapImage,getFieldMap,[('out_files','inlist')])])


# In[65]:


#fslSplit.help()


# #### Reslice fmap to functional

# In[66]:


resliceFieldMap = Node(Reslice(),name='resliceFieldMap')


# In[67]:


if unwarp and not precalc_fmap:
    wf.connect([(getFieldMap,resliceFieldMap,[('out','in_file')])])
    wf.connect([(mcflirtBetweenSess,resliceFieldMap,[('out_file','space_defining')])])


# In[68]:


Reslice.help()


# #### Apply unwarping

# In[69]:


unwarp_direction = 'z-'


# In[70]:


if unwarp: 
    unwarping = Node(FUGUE(unwarp_direction=unwarp_direction),name='unwarping')

# fugue -i epi --dwell=dwelltime --loadfmap=fieldmap -u result


# In[71]:


if unwarp:
    wf.connect([(applyRealign,unwarping,[('out_file','in_file')])])
    wf.connect([(acquisitionParams,unwarping,[('effective_echo_spacing','dwell_time')])])
    if precalc_fmap:
        wf.connect([(datasourcePrecalcFmap,unwarping,[('precalc_fmap','fmap_in_file')])])
    else:
        wf.connect([(resliceFieldMap,unwarping,[('out_file','fmap_in_file')])])


# In[72]:


FUGUE.help()


# ### Slice-timing correction (SPM)
# 
# Parker & Razlighi, 2019: "The Benefit of Slice Timing Correction in Common fMRI Preprocessing Pipelines."
# https://www.frontiersin.org/articles/10.3389/fnins.2019.00821/full

# In[73]:


ref_slice = 1                           # (an integer (int or long))
                                        # 1-based Number of the reference slice


# In[74]:


sliceTimingCorr = Node(SliceTiming(ref_slice=ref_slice),name='sliceTimingCorr')


# In[75]:


if unwarp:
    wf.connect([(unwarping,sliceTimingCorr,[('unwarped_file','in_files')])])
else:
    wf.connect([(applyRealign,sliceTimingCorr,[('out_file','in_files')])])
wf.connect([(acquisitionParams,sliceTimingCorr,[('num_slices','num_slices')])])
wf.connect([(acquisitionParams,sliceTimingCorr,[('slice_timing','slice_order')])])
wf.connect([(acquisitionParams,sliceTimingCorr,[('TR','time_repetition')])])
wf.connect([(acquisitionParams,sliceTimingCorr,[('TA','time_acquisition')])])


# In[76]:


#SliceTiming.help()


# ### Co-registration of functional and structural data (FreeSurfer bbregister, FLIRT FSL) 
# 
# Note: structural data is brought into functional space to avoid superfluous interpolation of functional volumes!
# 
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FLIRT
# 
# https://nipype.readthedocs.io/en/0.12.0/interfaces/generated/nipype.interfaces.fsl.preprocess.html#flirt
# 
# 

# #### Concatenate functional runs

# In[77]:


dimension = 't'
output_type = 'NIFTI'
merged_file = 'merged_func.nii'


# In[78]:


concatenateFunc = JoinNode(fslMerge(dimension=dimension, output_type=output_type, merged_file=merged_file),
                        joinfield='in_files',
                        joinsource='sessions',
                        name="concatenateFunc")


# In[79]:


wf.connect([(sliceTimingCorr, concatenateFunc,[('timecorrected_files', 'in_files')])])


# #### Get mean functional volume

# In[80]:


mean_vol = True                 # (a boolean) register to mean volume
save_mats = False               # (a boolean) save transformation parameters


# In[81]:


meanFunc = Node(MCFLIRT(mean_vol = mean_vol, save_mats=save_mats), 
                    name='meanFunc')


# In[82]:


wf.connect([(concatenateFunc, meanFunc,[('merged_file', 'in_file')])])


# In[83]:


MCFLIRT.help()


# #### Make functional brain mask for coregistration
# Note: this doesn't necessarily work well for all subjects. Therefore, this mask must be corrected manually. When the manual edits flag is true, the corrected binarized image is expected to be saved in the manual edits subject folder. Editing is done by loading the output of this node as a segmentation in ITKSNAP and then saving as a nifti file named brainMask.nii in the manual edits folder.

# In[84]:


thresh = 350

dilate = 3 # voxels
erode = 4


# In[85]:


binarizeMeanFunc = Node(Binarize(min=thresh, dilate=dilate, erode=erode),name='binarizeMeanFunc')


# In[86]:


wf.connect([(meanFunc,binarizeMeanFunc,[('mean_img','in_file')])])


# #### Coregister structural image to mean functional (FLIRT)

# In[87]:


out_matrix_file = 'struct2func.mat'     # (a pathlike object or string representing a file)
                                        # output affine matrix in 4x4 asciii format
apply_xfm = True                        # (a boolean)
                                        # apply transformation supplied by in_matrix_file or uses_qform to use
                                        # the affine matrix stored in the reference header
coarse_search = 4
fine_search = 2


# In[88]:


if coregister and coreg_method == 'flirt':
    coreg = Node(FLIRT(),name='coreg')
    
    # out_matrix_file=out_matrix_file, coarse_search=coarse_search, fine_search=fine_search


# In[89]:


if coregister and coreg_method == 'flirt':
    wf.connect([(datasource, coreg,[('T1', 'in_file')])])        
    #wf.connect([(convertT1ToNii, coreg,[('out_file', 'reference')])])   
    wf.connect([(meanFunc, coreg,[('mean_img', 'reference')])])  
    
    if manual_edits:
        wf.connect([(datasourceManualEdits, coreg,[('coreg_itksnap_struct2func_txt', 'in_matrix_file')])])
        coreg.inputs.apply_xfm = apply_xfm


# In[90]:


FLIRT.help()


# #### Coregister structural image to mean functional (FS)
# 
# not done!

# In[91]:


# contrast_type = 't2'


# In[92]:


# if coreg_method == 'freesurfer':
#     #coreg = Node(MRICoreg(),name='coreg')
    
#     coreg = Node(BBRegister(contrast_type=contrast_type),name='coreg')


# In[93]:


# if coreg_method == 'freesurfer':
#     wf.connect([(convertT1ToNii, coreg,[('out_file', 'source_file')])])
#     if manual_edits:
#         wf.connect([(datasourceManualEdits, coreg,[('coreg_itksnap_txt','init_reg_file')])])
            
#     wf.connect([(meanFunc, coreg,[('mean_img', 'source_file')])])


# In[94]:


# BBRegister.help()


# #### Coregister structural image to mean functional (apply manual ITK-snap correction)
# https://layerfmri.com/2019/02/11/high-quality-registration/
# 
# 
# ##### Mandatory inputs
# input_image (a pathlike object or string representing an existing file) – Image to apply transformation to (generally a coregistered functional). Maps to a command-line argument: --input %s.
# 
# reference_image (a pathlike object or string representing an existing file) – Reference image space that you wish to warp INTO. Maps to a command-line argument: --reference-image %s.
# 
# transforms (a list of items which are a pathlike object or string representing an existing file or ‘identity’) – Transform files: will be applied in reverse order. For example, the last specified transform will be applied first. Maps to a command-line argument: %s.
# 
# ##### Optional inputs
# args (a string) – Additional parameters to the command. Maps to a command-line argument: %s.
# 
# default_value (a float) – Maps to a command-line argument: --default-value %g. (Nipype default value: 0.0)
# 
# dimension (2 or 3 or 4) – This option forces the image to be treated as a specified-dimensional image. If not specified, antsWarp tries to infer the dimensionality from the input image. Maps to a command-line argument: --dimensionality %d.
# 
# environ (a dictionary with keys which are a bytes or None or a value of class ‘str’ and with values which are a bytes or None or a value of class ‘str’) – Environment variables. (Nipype default value: {})
# 
# float (a boolean) – Use float instead of double for computations. Maps to a command-line argument: --float %d. (Nipype default value: False)
# 
# input_image_type (0 or 1 or 2 or 3) – Option specifying the input image type of scalar (default), vector, tensor, or time series. Maps to a command-line argument: --input-image-type %d.
# 
# interpolation (‘Linear’ or ‘NearestNeighbor’ or ‘CosineWindowedSinc’ or ‘WelchWindowedSinc’ or ‘HammingWindowedSinc’ or ‘LanczosWindowedSinc’ or ‘MultiLabel’ or ‘Gaussian’ or ‘BSpline’) – Maps to a command-line argument: %s. (Nipype default value: Linear)
# 
# interpolation_parameters (a tuple of the form: (an integer) or a tuple of the form: (a float, a float))
# 
# invert_transform_flags (a list of items which are a boolean)
# 
# num_threads (an integer) – Number of ITK threads to use. (Nipype default value: 1)
# 
# out_postfix (a string) – Postfix that is appended to all output files (default = _trans). (Nipype default value: _trans)
# 
# output_image (a string) – Output file name. Maps to a command-line argument: --output %s.
# 
# print_out_composite_warp_file (a boolean) – Output a composite warp file instead of a transformed image. Requires inputs: output_image.
# 
# 

# In[95]:


#antsApplyTransforms --interpolation BSpline[5] -d 3 -i MP2RAGE.nii -r EPI.nii -t initial_matrix.txt -o registered_applied.nii


# In[96]:


interpolation = 'BSpline'
input_image_type = 3


# In[97]:


if coregister and coreg_method == 'itk-snap':
    coreg = Node(ApplyTransforms(interpolation=interpolation,
                                input_image_type=input_image_type), name='coreg')


# In[98]:


if coregister and coreg_method == 'itk-snap':
    wf.connect([(datasource, coreg,[('T1', 'input_image')])])        
    wf.connect([(meanFunc, coreg,[('mean_img', 'reference_image')])])  

    wf.connect([(datasourceManualEdits, coreg,[('coreg_itksnap_struct2func_txt', 'transforms')])])


# In[99]:


ApplyTransforms.help()


# #### Coregister mean functional to anatomical image (ANTs)
# https://layerfmri.com/2019/02/11/high-quality-registration/
# 
# antsRegistration \
# --verbose 1 \
# --dimensionality 3 \
# --float 1 \
# --output [registered_,registered_Warped.nii.gz,registered_InverseWarped.nii.gz] \
# --interpolation Linear \
# --use-histogram-matching 0 \
# --winsorize-image-intensities [0.005,0.995] \
# --initial-moving-transform initial_matrix.txt \
# --transform Rigid[0.05] \
# --metric CC[static_image.nii,moving_image.nii,0.7,32,Regular,0.1] \
# --convergence [1000x500,1e-6,10] \
# --shrink-factors 2x1 \
# --smoothing-sigmas 1x0vox \
# --transform Affine[0.1] \
# --metric MI[static_image.nii,moving_image.nii,0.7,32,Regular,0.1] \
# --convergence [1000x500,1e-6,10] \
# --shrink-factors 2x1 \
# --smoothing-sigmas 1x0vox \
# --transform SyN[0.1,2,0] \
# --metric CC[static_image.nii,moving_image.nii,1,2] \
# --convergence [500x100,1e-6,10] \
# --shrink-factors 2x1 \
# --smoothing-sigmas 1x0vox \
# -x mask.nii
# 
# 
# See also: https://github.com/ANTsX/ANTs/wiki/Anatomy-of-an-antsRegistration-call
# 
# about masking: https://github.com/ANTsX/ANTs/issues/483

# In[100]:


verbose = True                          # (a boolean, nipype default value: False)
                                        # argument: ``-v``
    
dimension = 3                           # dimension: (3 or 2, nipype default value: 3)
                                        # image dimension (2 or 3)
                                        # argument: ``--dimensionality %d``
        
float = True                            # (a boolean)
                                        # Use float instead of double for computations.
                                        # argument: ``--float %d``
        
output_transform_prefix = 'registered_' # (a string, nipype default value: transform)
                                        # argument: ``%s``
    
output_warped_image = 'registered_Warped.nii.gz'              
                                        # (a boolean or a pathlike object or string
                                        # representing a file)
    
output_inverse_warped_image = 'registered_InverseWarped.nii.gz'       
                                        # (a boolean or a pathlike object or
                                        # string representing a file)
                                        # requires: output_warped_image
        
interpolation = 'Linear'                # ('Linear' or 'NearestNeighbor' or 'CosineWindowedSinc'
                                        # or 'WelchWindowedSinc' or 'HammingWindowedSinc' or
                                        # 'LanczosWindowedSinc' or 'BSpline' or 'MultiLabel' or 'Gaussian',
                                        # nipype default value: Linear)
                                        # argument: ``%s``   
                
use_histogram_matching = False          #  (a boolean or a list of items which are a
                                        # boolean, nipype default value: True)
                                        # Histogram match the images before registration. 
        
winsorize_lower_quantile = 0.005        # (0.0 <= a floating point number <= 1.0,
                                        # nipype default value: 0.0)
                                        # The Lower quantile to clip image ranges
                                        # argument: ``%s``
            
winsorize_upper_quantile = 0.995        # (0.0 <= a floating point number <= 1.0,
                                        # nipype default value: 1.0)
                                        # The Upper quantile to clip image ranges
                                        # argument: ``%s``
            
#initial_moving_transform = 'initial_matrix.txt'   # (a list of items which are an existing file
                                        # name)
                                        # A transform or a list of transforms that should be appliedbefore the
                                        # registration begins. Note that, when a list is given,the
                                        # transformations are applied in reverse order.
                                        # argument: ``%s``
                                        # mutually_exclusive: initial_moving_transform_com
                        
transforms = ['Rigid','Affine','SyN']   # (a list of items which are 'Rigid' or 'Affine' or
                                        # 'CompositeAffine' or 'Similarity' or 'Translation' or 'BSpline' or
                                        # 'GaussianDisplacementField' or 'TimeVaryingVelocityField' or
                                        # 'TimeVaryingBSplineVelocityField' or 'SyN' or 'BSplineSyN' or
                                        # 'Exponential' or 'BSplineExponential')
                                        # argument: ``%s``
                    
transform_parameters = [(0.1,), (0.1,), (0.1, 3.0, 0.0)]        
                                        # (a list of items which are a tuple of the form:
                                        # (a float) or a tuple of the form: (a float, a float, a float) or a
                                        # tuple of the form: (a float, an integer (int or long), an integer
                                        # (int or long), an integer (int or long)) or a tuple of the form:
                                        # (a float, an integer (int or long), a float, a float, a float, a
                                        # float) or a tuple of the form: (a float, a float, a float, an
                                        # integer (int or long)) or a tuple of the form: (a float, an
                                        # integer (int or long), an integer (int or long), an integer (int
                                        # or long), an integer (int or long)))
                                
metric = ['MI', 'MI', 'CC']             # (a list of items which are 'CC' or 'MeanSquares' or 'Demons'
                                        # or 'GC' or 'MI' or 'Mattes' or a list of items which are 'CC' or
                                        # 'MeanSquares' or 'Demons' or 'GC' or 'MI' or 'Mattes')
                                        # the metric(s) to use for each stage. Note that multiple metrics per
                                        # stage are not supported in ANTS 1.9.1 and earlier.
                
metric_weight = [1.0,1.0,1.0]           # (a list of items which are a float or a list of items
                                        # which are a float, nipype default value: [1.0])
                                        # the metric weight(s) for each stage. The weights must sum to 1 per
                                        # stage.
                                        # requires: metric
                
radius_or_number_of_bins = [32,32,4]    # (a list of items which are an integer (int
                                        # or long) or a list of items which are an integer (int or long),
                                        # nipype default value: [5])
                                        # the number of bins in each stage for the MI and Mattes metric, the
                                        # radius for other metrics
                                        # requires: metric_weight
                    
sampling_strategy = ['Regular','Regular','None']              
                                        # (a list of items which are 'None' or 'Regular' or
                                        # 'Random' or None or a list of items which are 'None' or 'Regular'
                                        # or 'Random' or None)
                                        # the metric sampling strategy (strategies) for each stage
                                        # requires: metric_weight
                
sampling_percentage = [0.25, 0.25, None]         
                                        # (a list of items which are 0.0 <= a floating
                                        # point number <= 1.0 or None or a list of items which are 0.0 <= a
                                        # floating point number <= 1.0 or None)
                                        # the metric sampling percentage(s) to use for each stage
                                        # requires: sampling_strategy
                
convergence_threshold = [1e-6,1e-6,1e-6]# (a list of at least 1 items which are a float,
                                        # nipype default value: [1e-06])
                                        # requires: number_of_iterations
        
convergence_window_size = [10,10,10]    # (a list of at least 1 items which are an
                                        # integer (int or long), nipype default value: [10])
                                        # requires: convergence_threshold
        
number_of_iterations = [[1000,500,250,100], [1000,500,250,100],[100,70,50,20]]       
                                        # (a list of items which are a list of items
                                        # which are an integer (int or long))                
                                        
shrink_factors = [[8,4,2,1], [8,4,2,1], [8,4,2,1]]  
                                        # (a list of items which are a list of items which are
                                        # an integer (int or long))
    
smoothing_sigmas = [[3.0,2.0,1.0,0.0], [3.0,2.0,1.0,0.0], [3.0,2.0,1.0,0.0]]
                                        # (a list of items which are a list of items which
                                        # are a float)


# In[101]:


if coregister and not precalc_coreg and coreg_method == 'antsRegistration':
    coreg = Node(Registration(verbose=verbose,
                              dimension=dimension,
                              float=float,
                              output_transform_prefix=output_transform_prefix,
                              output_warped_image=output_warped_image,
                              output_inverse_warped_image=output_inverse_warped_image,
                              interpolation=interpolation, 
                              use_histogram_matching=use_histogram_matching,
                              winsorize_lower_quantile=winsorize_lower_quantile,
                              winsorize_upper_quantile=winsorize_upper_quantile, 
                              transforms=transforms, 
                              transform_parameters=transform_parameters,
                              metric=metric, 
                              metric_weight=metric_weight, 
                              radius_or_number_of_bins=radius_or_number_of_bins,
                              sampling_strategy=sampling_strategy, 
                              sampling_percentage=sampling_percentage,
                              convergence_threshold=convergence_threshold, 
                              convergence_window_size=convergence_window_size,
                              number_of_iterations=number_of_iterations, 
                              shrink_factors=shrink_factors, 
                              smoothing_sigmas=smoothing_sigmas),
                 name='coreg')


# In[102]:


if coregister and not precalc_coreg and coreg_method == 'antsRegistration':
    if coreg_dir == 'func2struct':
        # when moving func 2 struct
        wf.connect([(datasource, coreg,[('UNI', 'fixed_image')])])    
        wf.connect([(meanFunc, coreg,[('mean_img', 'moving_image')])])
        wf.connect([(datasourceManualEdits, coreg,[('coreg_itksnap_func2struct_txt', 'initial_moving_transform')])])    
        #wf.connect([(datasource, coreg,[('brainmask', 'fixed_image_masks')])])
        wf.connect([(datasourceManualEdits, coreg,[('manual_midoccmask', 'fixed_image_masks')])])
        
    elif coreg_dir == 'struct2func':
        # when moving struct 2 func:
        wf.connect([(meanFunc, coreg,[('mean_img', 'fixed_image')])]) 
        wf.connect([(datasource, coreg,[('UNI', 'moving_image')])])
        wf.connect([(datasourceManualEdits, coreg,[('coreg_itksnap_struct2func_txt', 'initial_moving_transform')])])
        #wf.connect([(datasource, coreg,[('brainmask', 'moving_image_masks')])])
        wf.connect([(datasourceManualEdits, coreg,[('manual_midoccmask', 'moving_image_masks')])])
        


# In[103]:


Registration.help()


# ### Combine composite transforms into list
# coreg_regFUNC2UNI.hd5, coreg_regUNI2T1.hd5
# 

# In[104]:


n_transforms = 2
combineCoregTransforms = Node(utilMerge(n_transforms),name='combineCoregTransforms')


# In[105]:


if coregister and precalc_coreg and coreg_method == 'antsRegistration' and coreg_dir == 'func2struct':
    wf.connect([(datasourceManualEdits, combineCoregTransforms,[('coreg_regFUNC2INPLANE', 'in1')])]) 
    wf.connect([(datasourceManualEdits, combineCoregTransforms,[('coreg_regINPLANE2T1', 'in2')])]) 


# In[106]:


utilMerge.help()


# #### Apply coregistration transforms to mean functional

# In[107]:


interpolation = 'BSpline'
interpolation_parameters = (5,)
input_image_type = 0


# In[108]:


applyCoreg2MeanFunc = Node(ApplyTransforms(interpolation=interpolation,
                                          interpolation_parameters=interpolation_parameters,
                                          input_image_type=input_image_type), name = 'applyCoreg2MeanFunc')


# In[109]:


if coregister and precalc_coreg and coreg_method == 'antsRegistration' and coreg_dir == 'func2struct':
    if precalc_coreg:
        output_image = 'reg_meanFunc.nii'
        wf.connect([(meanFunc, applyCoreg2MeanFunc,[('mean_img', 'input_image')])]) 
        wf.connect([(datasource, applyCoreg2MeanFunc,[('T1', 'reference_image')])]) 
        wf.connect([(combineCoregTransforms, applyCoreg2MeanFunc,[('out', 'transforms')])])
        applyCoreg2MeanFunc.inputs.output_image = output_image
    else:
        error()
        #wf.connect([(coreg, applyCoreg2MeanFunc,[('forward_transforms', 'transforms')])])


# In[110]:


ApplyTransforms.help()


# In[111]:


# if coregister and coreg_method == 'antsRegistration':
#     if coreg_dir == 'func2struct':
#         output_image = 'reg_meanFunc.nii'
#         wf.connect([(meanFunc, applyCoreg2MeanFunc,[('mean_img', 'input_image')])]) 
#         wf.connect([(datasource, applyCoreg2MeanFunc,[('UNI', 'reference_image')])]) 
#         wf.connect([(coreg, applyCoreg2MeanFunc,[('forward_transforms', 'transforms')])]) 
        
#     elif coreg_dir == 'struct2func':
#         output_image = 'reg_UNI.nii'
#         wf.connect([(meanFunc, applyCoreg2MeanFunc,[('mean_img', 'reference_image')])]) 
#         wf.connect([(datasource, applyCoreg2MeanFunc,[('UNI', 'input_image')])]) 
#         wf.connect([(coreg, applyCoreg2MeanFunc,[('forward_transforms', 'transforms')])]) 
        
        
#     applyCoreg2MeanFunc.inputs.output_image = output_image


# #### Apply coregistration transforms to all runs

# In[112]:


interpolation = 'BSpline'
interpolation_parameters = (5,)
input_image_type = 3


# In[113]:


applyCoreg = Node(ApplyTransforms(interpolation=interpolation,
                                  interpolation_parameters=interpolation_parameters,
                                  input_image_type=input_image_type), name = 'applyCoreg')


# In[114]:


if coregister and precalc_coreg and coreg_method == 'antsRegistration' and coreg_dir == 'func2struct':
    if precalc_coreg:
        output_image = 'reg_meanFunc.nii'
        wf.connect([(sliceTimingCorr, applyCoreg,[('timecorrected_files', 'input_image')])])
        wf.connect([(datasource, applyCoreg,[('T1', 'reference_image')])]) 
        wf.connect([(combineCoregTransforms, applyCoreg,[('out', 'transforms')])])
    else:
        error()


# In[115]:


#ApplyTransforms.help()


# ### Surface projection of functional runs

# In[116]:


hemi_list = ['lh','rh']
reg_header = True
sampling_range_list = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
sampling_method = 'point'
sampling_units = 'mm'
interp_method = 'trilinear'
out_type = 'mgh'


# #### Iterate over depths and hemispheres

# In[117]:


hemi_depth = Node(IdentityInterface(fields=['hemi','sampling_range']),name='hemi_depth')
hemi_depth.iterables = [('hemi', hemi_list), ('sampling_range', sampling_range_list)]
hemi_depth.synchronize = False


# #### Mean functional

# In[118]:


surfaceProjectMeanFunc = Node(SampleToSurface(reg_header=reg_header,
                                              sampling_method=sampling_method,
                                              sampling_units=sampling_units,
                                              out_type=out_type,
                                              interp_method=interp_method),name='surfaceProjectMeanFunc')


# In[119]:


# if coregister and coreg_method == 'antsRegistration':
#     if coreg_dir == 'func2struct':
#         wf.connect([(applyCoreg2MeanFunc,surfaceProjectMeanFunc,[('output_image', 'source_file')])]) 
#         wf.connect([(subjects,surfaceProjectMeanFunc,[('subject_id', 'subject_id')])])
#         wf.connect([(hemi_depth,surfaceProjectMeanFunc,[('hemi', 'hemi')])])
#         wf.connect([(hemi_depth,surfaceProjectMeanFunc,[('sampling_range', 'sampling_range')])])


# #### Other runs

# In[120]:


surfaceProject = Node(SampleToSurface(reg_header=reg_header,
                                      sampling_method=sampling_method,
                                      sampling_units=sampling_units,
                                      out_type=out_type,
                                      interp_method=interp_method),name='surfaceProject')


# In[121]:


# if coregister and coreg_method == 'antsRegistration':
#     if coreg_dir == 'func2struct':
#         wf.connect([(applyCoreg,surfaceProject,[('output_image', 'source_file')])]) 
#         wf.connect([(subjects,surfaceProject,[('subject_id', 'subject_id')])])
#         wf.connect([(hemi_depth,surfaceProject,[('hemi', 'hemi')])])
#         wf.connect([(hemi_depth,surfaceProject,[('sampling_range', 'sampling_range')])])


# #### Prepare occipital mask for pRF mapping

# Surface project manual occipital mask

# In[122]:


interp_method = 'nearest'


# In[123]:


surfaceProjectOccipitalMask = Node(SampleToSurface(reg_header=reg_header,
                                      sampling_method=sampling_method,
                                      sampling_units=sampling_units,
                                      out_type=out_type,
                                      interp_method=interp_method),name='surfaceProjectOccipitalMask')


# In[124]:


# if coregister and coreg_method == 'antsRegistration':
#     if coreg_dir == 'func2struct':
#         wf.connect([(datasourceManualEdits,surfaceProjectOccipitalMask,[('manual_occipitalmask', 'source_file')])]) 
#         wf.connect([(subjects,surfaceProjectOccipitalMask,[('subject_id', 'subject_id')])])
#         wf.connect([(hemi_depth,surfaceProjectOccipitalMask,[('hemi', 'hemi')])])
#         wf.connect([(hemi_depth,surfaceProjectOccipitalMask,[('sampling_range', 'sampling_range')])])


# Make lable out of surface projection

# In[125]:


def mri_vol2label_bash(subjects_dir,subject_id,working_dir,hemi,sampling_range,surface_file):
    from os.path import join as opj
    import os

    out_file = opj(working_dir,'_subject_id_'+subject_id,
                   '_hemi_'+hemi+'_sampling_range_'+str(sampling_range),'makeOccLabel',
                   hemi+'_occ_depth'+str(sampling_range)+'.label')
    bash_command = 'mri_vol2label --i '+surface_file+' --id 1 --surf '+subject_id + ' '+ hemi + ' --l '+out_file
    
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
    print(out_file)
    print(bash_command)
    print('mri_vol2label --i $OUTDIR/sub-01/lh_occ_depth0.0.mgh --id 1  --surf sub-01 lh  --l $OUTDIR/sub-01/lh_occ_depth0.0.label')
    print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

    os.system(bash_command)
    
    return out_file


# In[126]:


makeOccLabel = Node(Function(input_names = ['subjects_dir','subject_id','working_dir','hemi',
                                            'sampling_range','surface_file'],
                             output_names=['out_file'],
                             function=mri_vol2label_bash),
                    name='makeOccLabel')
makeOccLabel.inputs.working_dir = opj(der_dir,wf_name)
makeOccLabel.inputs.subjects_dir = subjects_dir


# In[127]:


# if coregister and coreg_method == 'antsRegistration':
#     if coreg_dir == 'func2struct':
#         wf.connect([(subjects,makeOccLabel,[('subject_id', 'subject_id')])])
#         wf.connect([(hemi_depth,makeOccLabel,[('hemi', 'hemi')])])
#         wf.connect([(hemi_depth,makeOccLabel,[('sampling_range', 'sampling_range')])])
#         wf.connect([(surfaceProjectOccipitalMask,makeOccLabel,[('out_file', 'surface_file')])])


# In[128]:


#operation = 'mul'   # ('add' or 'sub' or 'mul' or 'div' or 'rem' or 'max' or
                    # 'min')
                    # operation to perform
                    # flag: -%s, position: 4


# In[129]:


#if coregister:
#    occipitalGM = Node(BinaryMaths(operation=operation), name='occipitalGM')


# In[130]:


#if coregister:
#    wf.connect([(datasourceManualEdits,occipitalGM,[('binarizedmeanfunc', 'in_file')])]) 
#    wf.connect([(datasourceManualEdits,occipitalGM,[('occipital', 'operand_file')])]) 


# ### Put data in sink
# 
# A workflow working directory is like a cache. It contains not only the outputs of various processing stages, it also contains various extraneous information such as execution reports, hashfiles determining the input state of processes. All of this is embedded in a hierarchical structure that reflects the iterables that have been used in the workflow. This makes navigating the working directory a not so pleasant experience. And typically the user is interested in preserving only a small percentage of these outputs. The DataSink interface can be used to extract components from this cache and store it at a different location.

# In[131]:


dataSink = Node(DataSink(), name='dataSink')
dataSink.inputs.base_directory = out_dir


# In[132]:


# T1.nii
wf.connect([(datasource,dataSink,[('T1','func.anat')])])

# realigned func volumes (within and between session)
wf.connect([(applyRealign,dataSink,[('out_file','func.realign')])])

# mean functional
wf.connect([(meanFunc,dataSink,[('mean_img','func.meanFunc')])])

# binarized mean functional
wf.connect([(binarizeMeanFunc,dataSink,[('binary_file','func.meanFunc.@binarizedMeanFunc')])])


# # prepared fieldmap
# wf.connect([(prepFieldMap,dataSink,[('out_fieldmap','func.prepFieldMap')])])

# # unwarped func volumes
# if unwarp:
#     wf.connect([(unwarping,dataSink,[('unwarped_file','func.unwarp')])])

# slice-time corrected func volumes
wf.connect([(sliceTimingCorr,dataSink,[('timecorrected_files','func.sliceTimeCorr')])])


# coregistered T1 and transformation matrices
if coregister:
    if coreg_method == 'antsRegistration':
#         wf.connect([(coreg,dataSink,[('warped_image','func.coreg')])])
#         wf.connect([(coreg, dataSink,[('forward_transforms', 'func.coreg.@forwardTransform')])]) 
#         wf.connect([(coreg, dataSink,[('reverse_transforms', 'func.coreg.@reverseTransform')])])

        wf.connect([(applyCoreg, dataSink,[('output_image', 'func.coreg')])])
        wf.connect([(applyCoreg2MeanFunc, dataSink,[('output_image', 'func.coreg.@meanFunc')])])
        
    elif coreg_method == 'itk-snap':
        wf.connect([(coreg,dataSink,[('output_image','func.coreg')])])
    
## occipital GM mask for pRF mapping
#if manual_edits:
#    wf.connect([(occipitalGM,dataSink,[('out_file','func.occipitalGM')])])
    
# if coreg_method == 'flirt':
#    wf.connect([(coreg,dataSink,[('out_file','func.coreg')])])
#    wf.connect([(coreg,dataSink,[('out_matrix_file','func.coreg.@out_matrix_file')])])
# elif coreg_method == 'freesurfer':
#    wf.connect([(coreg,dataSink,[('out_file','func.coreg')])])
#    wf.connect([(coreg,dataSink,[('out_matrix_file','func.coreg.@out_matrix_file')])])
# elif coreg_method == 'antsRegistration':
#    wf.connect([(coreg,dataSink,[('warped_image','func.coreg')])])


# ### Put pRF analysis data in separate sink
# 

# In[133]:


prfSink = Node(DataSink(), name='prfSink')
prfSink.inputs.base_directory = pRF_dir


# In[134]:


# if coregister and coreg_method == 'antsRegistration':
    
#     # coregistered other functional runs
#     wf.connect([(applyCoreg,prfSink,[('output_image','data')])])
    
#     # coregistered mean functional
#     wf.connect([(applyCoreg2MeanFunc,prfSink,[('output_image','data.@meanFunc')])])
        
# #     # surface projected mean functional
# #     wf.connect([(surfaceProjectMeanFunc,prfSink,[('out_file','data.surfs_meanFunc')])])
    
# #     # surface projected other functional runs
# #     wf.connect([(surfaceProject,prfSink,[('out_file','data.surfs')])])
    
# #     # occipital labels
# #     wf.connect([(makeOccLabel,prfSink,[('out_file','data.occLabels')])])


# ### Write graph for visualization and run pipeline

# In[135]:


if write_graph:
    wf.write_graph("workflowgraph.dot",graph2use='exec', format='svg', simple_form=True)


# In[ ]:


if run_pipeline:
    if n_procs == 1:
        wf.run()
    else:
        wf.run('MultiProc', plugin_args={'n_procs': n_procs})


# In[ ]:




