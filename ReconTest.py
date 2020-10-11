#%matplotlib notebook

import sigpy as sp
import sigpy.mri as mr
import sigpy.plot as pl
import matplotlib.pyplot as plt
import numpy as np

def estimate_shape(coord):
    """Estimate array shape from coordinates.

    Shape is estimated by the different between maximum and minimum of
    coordinates in each axis.

    Args:
        coord (array): Coordinates.
    """
    ndim = coord.shape[-1]
    shape = [int(coord[..., i].max() - coord[..., i].min())
             for i in range(ndim)]

    return shape

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

#dir='data/brain/'
dir='../../data/Tess_20200917/radial_data/radial_data/'


ksp_file = dir + 'ksp.npy'
coord_file = dir + 'coord.npy'

# Choose computing device.
# Device(-1) specifies CPU, while others specify GPUs.
# GPU requires installing cupy.
#try:
#    device = sp.Device(0)
#except:
#    device = sp.Device(-1)

device = sp.Device(0)

xp = device.xp
device.use()

# Load datasets.
def show_data_info(data, name):
  print("{}: shape={}, dtype={}".format(name, data.shape, data.dtype))

#ksp = np.load(dir+'ksp.npy').transpose((2,1,0))
#coord = np.load(dir+'coord.npy').transpose((1,0,2))

ksp = np.load(dir+'ksp.npy')
coord = np.load(dir+'coord.npy')

ksp = ksp.transpose((2,1,0))
coord = coord.transpose((1,0,2))*96

print("estimate shape=", estimate_shape(coord))

#dcf = (coord[..., 0]**2 + coord[..., 1]**2+ coord[..., 2]**2)**0.5

show_data_info(ksp, "ksp")
show_data_info(coord, "coord")
#show_data_info(dcf, "dcf")

#ksp = np.stack((ksp.real, ksp.imag), axis=-1)
#ksp = np.stack((ksp.real, ksp.imag), axis=-1).astype(np.double)

mps = mr.app.JsenseRecon(ksp, coord=coord, device=device).run()
#mps = mr.app.JsenseRecon(ksp, device=device).run()

cg_app = mr.app.SenseRecon(
    ksp, mps, coord=coord, device=device, lamda=lamda,
    max_iter=max_iter, save_objective_values=True)
cg_img = cg_app.run()

np.save("cg_img.npy", cg_img)

pl.ImagePlot(cg_img)