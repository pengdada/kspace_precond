#%matplotlib notebook

import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import numpy as np

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

#%% md

## Set parameters and load dataset

#%%

max_iter = 30
lamda = 0.01

brain_dir='data/brain/'
phantom_dir='/mnt/dskE/ResearchMRI/data/Tess_20200917/radial_data/radial_data/'


ksp_file = phantom_dir + 'ksp.npy'
coord_file = phantom_dir + 'coord.npy'

# Choose computing device.
# Device(-1) specifies CPU, while others specify GPUs.
# GPU requires installing cupy.
try:
    device = sp.Device(0)
except:
    device = sp.Device(-1)

xp = device.xp
device.use()

# Load datasets.
coord = xp.load(coord_file)
ksp = xp.load(ksp_file)

#mps = mr.app.JsenseRecon(ksp, coord=coord, device=device).run()
mps = mr.app.JsenseRecon(ksp, device=device).run()

cg_app = mr.app.SenseRecon(
    ksp, mps, coord=coord, device=device, lamda=lamda,
    max_iter=max_iter, save_objective_values=True)
cg_img = cg_app.run()

pl.ImagePlot(cg_img)