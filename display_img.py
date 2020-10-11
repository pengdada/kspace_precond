import numpy as np
import sigpy.plot as pl

name="../data/cg_img.npy"
name="./cg_img.npy"
img=np.load(name)

print(img.shape)

for t in range(len(img)):
    s = img[t, ..., ::-1]
    pl.ImagePlot(img[t, ..., ::-1], interpolation='lanczos')