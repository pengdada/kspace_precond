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
max_cg_iter = 5
lamda = 0.001

ksp_file = 'data/liver/ksp.npy'
coord_file = 'data/liver/coord.npy'

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
ksp = xp.load(ksp_file)
coord = xp.load(coord_file)

#%% md

## Estimate sensitivity maps using JSENSE

# Here we use [JSENSE](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21245) to estimate sensitivity maps.

#%%

mps = mr.app.JsenseRecon(ksp, coord=coord, device=device).run()

#%% md

## ADMM

#%%

admm_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, max_iter=max_iter // max_cg_iter,
        solver='ADMM', max_cg_iter=max_cg_iter, device=device, save_objective_values=True)
admm_img = admm_app.run()

pl.ImagePlot(admm_img)

#%% md

## ADMM with circulant preconditioner

#%%

rho = 1
circ_precond = mr.circulant_precond(mps, coord=coord, device=device, lamda=rho)

img_shape = mps.shape[1:]
G = sp.linop.FiniteDifference(img_shape)
g = G.H * G * sp.dirac(img_shape)
g = sp.fft(g)
g = sp.to_device(g, device=device)
circ_precond = 1 / (1 / circ_precond + lamda * g)

img_shape = mps.shape[1:]
D = sp.linop.Multiply(img_shape, circ_precond)
P = sp.linop.IFFT(img_shape) * D * sp.linop.FFT(img_shape)

admm_cp_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, max_iter=max_iter // max_cg_iter,
        P=P, rho=rho,
        solver='ADMM', max_cg_iter=max_cg_iter, device=device, save_objective_values=True)
admm_cp_img = admm_cp_app.run()

pl.ImagePlot(admm_cp_img)

#%% md

## Primal dual hybrid gradient reconstruction

#%%

pdhg_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, max_iter=max_iter,
        solver='PrimalDualHybridGradient', device=device, save_objective_values=True)
pdhg_img = pdhg_app.run()

pl.ImagePlot(pdhg_img)

#%% md

## PDHG with dcf

#%%

# Compute preconditioner
precond_dcf = mr.pipe_menon_dcf(coord, device=device)
precond_dcf = xp.tile(precond_dcf, [len(mps)] + [1] * (mps.ndim - 1))
img_shape = mps.shape[1:]
G = sp.linop.FiniteDifference(img_shape)
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_dcf.ravel(), sigma2.ravel()])

pdhg_dcf_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, sigma=sigma, max_iter=max_iter,
        solver='PrimalDualHybridGradient', device=device, save_objective_values=True)
pdhg_dcf_img = pdhg_dcf_app.run()

pl.ImagePlot(pdhg_dcf_img)

#%% md

## PDHG with single-channel precond.

#%%

# Compute preconditioner
ones = np.ones_like(mps)
ones /= len(mps)**0.5
precond_sc = mr.kspace_precond(ones, coord=coord, device=device)
img_shape = mps.shape[1:]
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_sc.ravel(), sigma2.ravel()]) / 2

pdhg_sc_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, sigma=sigma, max_iter=max_iter,
        solver='PrimalDualHybridGradient', device=device, save_objective_values=True)
pdhg_sc_img = pdhg_sc_app.run()

pl.ImagePlot(pdhg_sc_img)

#%% md

## PDHG with multi-channel precond.

#%%

# Compute preconditioner
precond_mc = mr.kspace_precond(mps, coord=coord, device=device)
img_shape = mps.shape[1:]
max_eig_G = sp.app.MaxEig(G.H * G).run()
sigma2 = xp.ones([sp.prod(img_shape) * len(img_shape)],
                 dtype=ksp.dtype) / max_eig_G
sigma = xp.concatenate([precond_mc.ravel(), sigma2.ravel()])

pdhg_mc_app = mr.app.TotalVariationRecon(
        ksp, mps, lamda=lamda, coord=coord, sigma=sigma, max_iter=max_iter,
        solver='PrimalDualHybridGradient', device=device, save_objective_values=True)
pdhg_mc_img = pdhg_mc_app.run()

pl.ImagePlot(pdhg_mc_img)

#%% md

## Convergence curves

#%%

plt.figure(figsize=(8, 3))
plt.semilogy(admm_app.time, admm_app.objective_values,
               marker='v', color='C1')
plt.semilogy(admm_cp_app.time, admm_cp_app.objective_values,
               marker='^', color='C2')
plt.semilogy(pdhg_app.time, pdhg_app.objective_values,
               marker='+', color='C3')
plt.semilogy(pdhg_dcf_app.time, pdhg_dcf_app.objective_values,
               marker='s', color='C4')
plt.semilogy(pdhg_sc_app.time, pdhg_sc_app.objective_values,
               marker='*', color='C5')
plt.semilogy(pdhg_mc_app.time, pdhg_mc_app.objective_values,
               marker='x', color='C6')
plt.legend(['ADMM',
            'ADMM w/ circulant precond.',
            'PDHG',
            'PDHG w/ density comp.',
            'PDHG w/ SC k-space precond.',
            'PDHG w/ MC k-space precond.'])
plt.ylabel('Objective Value [a.u.]')
plt.xlabel('Time [s]')
plt.title(r"Total Variation Regularized Reconstruction")
plt.tight_layout()
plt.show()
