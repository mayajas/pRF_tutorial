#!/usr/bin/env python
# coding: utf-8

# ## Population receptive field mapping workflow

# In[1]:


# !pip install brainspace
#!pip install pysurfer
#!pip install neuropythy
#!pip install fslpy
import numpy as np
from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel, DoG_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, DoG_Iso2DGaussianFitter
import os
import sys
from os.path import join as opj
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import image, surface, plotting, signal
import pickle
import math 
from scipy.io import savemat
#import runpy
#from freesurfer_surface import Surface, Vertex, Triangle

# from nibabel.freesurfer.mghformat import load
# from brainspace.plotting import plot_hemispheres
# from brainspace.mesh.mesh_io import read_surface

# import neuropythy as ny
from fsl.data.freesurfer import loadVertexDataFile

import matplotlib.pyplot as plt
from matplotlib import animation


# ### Set data directories

# In[2]:


local = False
if not local:
    slurm_run = True


# In[3]:


# get current sub and hem ids
if local or (local and not slurm_run):
    sub_id        = 0 
    hem_id        = 0 
elif not local and slurm_run:
    sub_id        = int(sys.argv[1])
    hem_id        = int(sys.argv[2])


# list of all sub and hem ids
subject_list  = ['sub-011']
hem_list      = ['lh','rh']
hem_text_list = ['left','right']

# directories
if local:
    proj_dir      = '/home/mayajas/scratch/project-0b-pRF-tutorial-3T/'
    home_dir      = '/home/mayajas/Documents/project-0b-pRF-tutorial-3T/'
else:
    proj_dir      = '/scratch/mayaaj90/project-0b-pRF-tutorial-3T/'
    home_dir      = '/home/mayaaj90/projects/project-0b-pRF-tutorial-3T/'
prfpy_dir     = opj(proj_dir,'output','prfpy',subject_list[sub_id])
FS_dir        = opj(proj_dir,'data_FS')   

# set FS subjects dir
os.environ["SUBJECTS_DIR"] = FS_dir

# set working dir
if os.getcwd() != opj(home_dir,'code','analysis-scripts','python'):
    os.chdir(opj(home_dir,'code','analysis-scripts','python'))
    
# number of cores to use: either set explicitly or base on settings in slurm job file
import os
if local:
    n_procs = 1
else:
    n_procs = int(os.getenv('OMP_NUM_THREADS'))   
print(n_procs)


# ### Input data filenames

# Image filenames

# In[4]:


# pRF mapping runs 
bar1_mgh_fn      = opj(prfpy_dir,hem_list[hem_id]+'.bar1.mgh')
bar2_mgh_fn      = opj(prfpy_dir,hem_list[hem_id]+'.bar2.mgh')
bar3_mgh_fn      = opj(prfpy_dir,hem_list[hem_id]+'.bar3.mgh')

bar1_nii_fn      = opj(prfpy_dir,'bar1.nii')
bar2_nii_fn      = opj(prfpy_dir,'bar2.nii')
bar3_nii_fn      = opj(prfpy_dir,'bar3.nii')

# mean functional
meanFunc_mgh_fn  = opj(prfpy_dir,hem_list[hem_id]+'.meanFunc.mgh')
meanFunc_nii_fn  = opj(prfpy_dir,'reg_meanFunc.nii')

# occ_mgh_fn     = opj(prfpy_dir,hem_list[hem_id]+'.occMask.mgh')
# occ_nii_fn       = opj(prfpy_dir,'occ.nii')

# anatomical image
T1_nii_fn        = opj(prfpy_dir,'T1_out.nii')

# Freesurfer mesh files
gm_surf_fn        = opj(FS_dir,subject_list[sub_id],'surf',hem_list[hem_id]+'.pial')
wm_surf_fn        = opj(FS_dir,subject_list[sub_id],'surf',hem_list[hem_id]+'.white')
inflated_surf_fn  = opj(FS_dir,subject_list[sub_id],'surf',hem_list[hem_id]+'.inflated')
sulc_surf_fn      = opj(FS_dir,subject_list[sub_id],'surf',hem_list[hem_id]+'.sulc')


# PRF output files

# In[5]:


grid_fit_fn      = opj(prfpy_dir,hem_list[hem_id]+'_grid_fit.pckl')
iterative_fit_fn = opj(prfpy_dir,hem_list[hem_id]+'_iterative_fit.pckl')

pRF_param_fn     = opj(prfpy_dir,hem_list[hem_id]+'_pRF_params.pckl')


# ### Load preprocessed data

# Volume data

# In[6]:


T1_nii       = image.load_img(T1_nii_fn)
meanFunc_nii = image.load_img(meanFunc_nii_fn)


# In[7]:


plotting.plot_stat_map(meanFunc_nii,bg_img=T1_nii,title='Mean functional')


# Freesurfer meshes

# In[8]:


gm_mesh       = surface.load_surf_mesh(gm_surf_fn) 
wm_mesh       = surface.load_surf_mesh(wm_surf_fn) 
inflated_mesh = surface.load_surf_mesh(inflated_surf_fn) 

wm_mesh.coordinates.shape


# Surface-projected functional data

# In[58]:


meanFunc      = loadVertexDataFile(meanFunc_mgh_fn)
bar1          = loadVertexDataFile(bar1_mgh_fn)
bar2          = loadVertexDataFile(bar2_mgh_fn)
bar3          = loadVertexDataFile(bar3_mgh_fn)


# In[10]:


view = plotting.view_surf(inflated_mesh, meanFunc,threshold=100,
                          bg_map=sulc_surf_fn,vmax=2000,vmin=0)
# view = plotting.view_surf(inflated_mesh, occ, threshold=None,
#                           bg_map=sulc_surf_fn,vmax=1,vmin=0)
view


# ### Make occipital mask
# (based on surface vertex y-coordinate cut-off, including only posterior vertices)

# In[83]:


y_coord_cutoff = -45

n_vtx = len(meanFunc[:])
n_vtx


# In[84]:


occ     = np.zeros(n_vtx)
occ[gm_mesh.coordinates[:,1]<y_coord_cutoff]=1.

occ_mask = np.nonzero(occ)[0]
occ_mask.shape


# ### Clean input data
# - apply occipital mask to constrain analysis to occipital pole
# - detrend, standardize, and bandpass filter each functional pRF run
# - average pRF runs

# In[13]:


detrend     = True
standardize = 'zscore'
low_pass    = 0.08       # Low pass filters out high frequency signals from our data: 
                         # fMRI signals are slow evolving processes, any high frequency signals 
                         # are likely due to noise 
high_pass   = 0.009      # High pass filters out any very low frequency signals (below 0.009Hz), 
                         # which may be due to intrinsic scanner instabilities
TR          = 1.5        # repetition time (s)

confounds   = None       # could add motion regressors here

# for details, see: https://nilearn.github.io/dev/modules/generated/nilearn.signal.clean.html


# In[59]:


masked_bar1 = bar1[occ_mask].T
masked_bar2 = bar2[occ_mask].T
masked_bar3 = bar3[occ_mask].T


# In[60]:


masked_bar1.shape


# In[63]:


masked_bar1  = signal.clean(masked_bar1,
                           confounds=confounds,
                           detrend=detrend, standardize=standardize, 
                           filter='butterworth', low_pass=low_pass, high_pass=high_pass, 
                           t_r=TR)
masked_bar2  = signal.clean(masked_bar2,
                           confounds=confounds,
                           detrend=detrend, standardize=standardize, 
                           filter='butterworth', low_pass=low_pass, high_pass=high_pass, 
                           t_r=TR)
masked_bar3  = signal.clean(masked_bar3,
                           confounds=confounds,
                           detrend=detrend, standardize=standardize, 
                           filter='butterworth', low_pass=low_pass, high_pass=high_pass, 
                           t_r=TR)


# In[64]:


avg_bar = ((masked_bar1 + masked_bar2 + masked_bar3) / 3).T


# Plot raw data

# In[68]:


vtx = 50


# In[69]:


plt.figure(figsize=(7, 5))
plt.plot(bar1[vtx, :],':')
plt.plot(bar2[vtx, :],':')
plt.plot(bar3[vtx, :],':')
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.title('raw data', fontsize=18)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)


# Plot filtered data

# In[70]:


plt.figure(figsize=(7, 5))
plt.plot(masked_bar1[:, vtx],':')
plt.plot(masked_bar2[:, vtx],':')
plt.plot(masked_bar3[:, vtx],':')
plt.plot(avg_bar[vtx, :],'k-')
plt.xlabel('Time [TRs]', fontsize=16)
plt.ylabel('Intensity', fontsize=16)
plt.title('filtered data', fontsize=18)
plt.subplots_adjust(bottom=.12, top=.95, right=.95, left=.12)


# ### Creating stimulus object

# Get pRF stimulus aperture file

# In[71]:


# set design mat from aperture file
Ap_file            = os.path.join(home_dir,'code','stim-scripts','apertures','stimulus_bar.mat')
mat                = scipy.io.loadmat(Ap_file)
design_matrix      = mat["stim"]

np.shape(design_matrix)


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import clear_output

plt.figure()
for i in range(np.shape(design_matrix)[2]):
    plt.imshow(design_matrix[:,:,i],cmap='gist_gray')
    plt.title('Frame %d' % (i+1))
    plt.show()
    clear_output(wait=True)


# Set max eccentricity

# In[72]:


# screen size parameters
screen_height_cm   = 12.65
screen_size_cm     = screen_height_cm/2 
screen_distance_cm = 45.0

# calculate max stim ecc
max_ecc            = math.atan(screen_size_cm/screen_distance_cm)
max_ecc_deg        = round(math.degrees(max_ecc))
max_ecc_deg


# Define stimulus object

# In[73]:


prf_stim = PRFStimulus2D(screen_size_cm=screen_size_cm,
                             screen_distance_cm=screen_distance_cm,
                             design_matrix=design_matrix,
                             TR=TR)


# In[74]:


help(PRFStimulus2D)


# ### Creating Gaussian model and fitter objects

# Define two-dimensional isotropic Gaussian pRF model and model fitter

# In[75]:


# Input parameters of Iso2DGaussianModel
hrf                = None     # string, list or numpy.ndarray, optional
                              # HRF shape for this Model.
                              # Can be 'direct', which implements nothing (for eCoG or later convolution),
                              # a list or array of 3, which are multiplied with the three spm HRF basis functions,
                              # and an array already sampled on the TR by the user.
                              # (the default is None, which implements standard spm HRF)
filter_predictions = False    # boolean, optional
                              # whether to high-pass filter the predictions, default False
filter_type        = 'sg'

sg_filter_window_length = 201
sg_filter_polyorder     = 3

filter_params      = {'window_length':sg_filter_window_length, 
                      'polyorder':sg_filter_polyorder}
normalize_RFs      = False    # whether or not to normalize the RF volumes (generally not needed).

# Input parameters of Iso2DGaussianFitter
n_jobs             = n_procs  # int, optional
                              # number of jobs to use in parallelization (iterative search), by default 1
fit_hrf            = False    # boolean, optional
                              # Whether or not to fit two extra parameters for hrf derivative and
                              # dispersion. The default is False.


# In[76]:


# Define 2D iso Gaussian model
gg = Iso2DGaussianModel(stimulus=prf_stim,
                          filter_predictions=filter_predictions,
                          filter_type=filter_type,
                          filter_params=filter_params,
                          normalize_RFs=normalize_RFs)
# Define 2D iso Gaussian model fitter
gf = Iso2DGaussianFitter(data=avg_bar, model=gg, n_jobs=n_jobs, fit_css=False)


# In[77]:


# help(Iso2DGaussianModel)


# ##### Grid fit
# 
# First, conduct a quick, coarse model fitting using provided grids and predictor definitions

# Grid fit parameters

# In[125]:


grid_nr       = 30
max_ecc_size  = round(max_ecc_deg,2)

size_grid, ecc_grid, polar_grid = max_ecc_size * np.linspace(0.25,1,grid_nr)**2, \
                    max_ecc_size * np.linspace(0.1,1,grid_nr)**2, \
                        np.linspace(0, 2*np.pi, grid_nr)
verbose       = True        # boolean, optional
                            # Whether to print output. The default is False.


# Run grid fit

# In[ ]:


gf.grid_fit(ecc_grid=ecc_grid,
            polar_grid=polar_grid,
            size_grid=size_grid,
            verbose=verbose,
            n_batches=n_procs)


# Save grid fit result

# In[ ]:


f = open(grid_fit_fn, 'wb')
pickle.dump(gf, f)
f.close()


# In[1]:


#help(gf.grid_fit)


# ##### Iterative fit
# 

# Iterative fit parameters

# In[2]:


# Iterative fit parameters (2D iso Gaussian model)
rsq_thresh_itfit = 0.0005      # float
                            # Rsq threshold for iterative fitting. Must be between 0 and 1.
verbose          = True     # boolean, optional
                            # Whether to print output. The default is False.


# Run iterative fit

# In[ ]:


gf.iterative_fit(rsq_threshold=rsq_thresh_itfit, verbose=verbose)


# Save iterative fit result

# In[ ]:


f = open(iterative_fit_fn, 'wb')
pickle.dump(gf, f)
f.close()


# ### PRF parameter estimates

# Extract pRF parameter estimates from iterative fit result

# In[ ]:


x=gf.iterative_search_params[:,0]
y=gf.iterative_search_params[:,1]
sigma=gf.iterative_search_params[:,2]
total_rsq = gf.iterative_search_params[:,-1]

#Calculate polar angle and eccentricity maps
polar = np.angle(x + 1j*y)
ecc = np.abs(x + 1j*y)


# Save pRF parameters

# In[ ]:


f = open(pRF_param_fn, 'wb')
pickle.dump([x, y, sigma, total_rsq, polar, ecc], f)
f.close()


# In[13]:


# == Epilog Slurmctld ==================================================

# Job ID: 12785524
# Array Job ID: 12785523_0
# Cluster: curta
# User/Group: mayaaj90/agcichy
# State: COMPLETED (exit code 0)
# Nodes: 1
# Cores per node: 20
# CPU Utilized: 1-09:38:59
# CPU Efficiency: 94.11% of 1-11:45:20 core-walltime
# Job Wall-clock time: 01:47:16
# Memory Utilized: 18.89 GB
# Memory Efficiency: 94.46% of 20.00 GB

# ======================================================================


# ## PRF mapping results

# In[79]:


# f = open(pRF_param_fn,'rb')
# x, y, sigma, total_rsq, polar, ecc = pickle.load(f)
# f.close()


# In[80]:


# x.shape


# In[ ]:




