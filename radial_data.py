# -*- coding: utf-8 -*-

import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy.linalg as la
import os
from os import path

if (path.exists('radial_data') == False):
    os.makedirs('radial_data')

# Function to generate radial trajectory (scaled between -0.5 and 0.5) given number of columns/spokes
# Returns variable traj [nCol nSpokes 3]
def generate_golden_radial_trajectory(nCol, nSpokes):
    rho = np.linspace(-0.5, 0.5, nCol)
    Mfib = [[0, 1, 0], [0, 0, 1], [1, 0, 1]]
    D, V = la.eig(Mfib)
    v = V[:, 0] / V[2, 0]
    phi1 = np.real(v[0])
    phi2 = np.real(v[1])
    traj = np.zeros((nCol, nSpokes, 3))
    for ii in range(nSpokes):
        if ii == 0:
            m1 = 0
            m2 = 0
        else:
            m1 = np.mod((m1_prev + phi1), 1)
            m2 = np.mod((m2_prev + phi2), 1)
        polarAngle = np.pi / 2 * (1 + m1)
        azAngle = 2 * np.pi * m2
        xA = np.cos(azAngle) * np.sin(polarAngle)
        yA = np.sin(azAngle) * np.sin(polarAngle)
        zA = np.cos(polarAngle)
        traj[:, ii, 0] = rho * xA
        traj[:, ii, 1] = rho * yA
        traj[:, ii, 2] = rho * zA
        m1_prev = m1
        m2_prev = m2
        return traj

# Main function to read radial data from ismrmrd format
filename = 'radial_data.h5'

dset = ismrmrd.Dataset(filename,'dataset', create_if_needed=False)

hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
enc = hdr.encoding[0]

rNx = enc.reconSpace.matrixSize.x
rNy = enc.reconSpace.matrixSize.y
rNz = enc.reconSpace.matrixSize.z
print("Reconstructed matrix size:", rNx, rNy, rNz, sep=" ")

rFOVx = enc.reconSpace.fieldOfView_mm.x
print("FOV:", rFOVx, "mm", sep=" ")

nCoils = hdr.acquisitionSystemInformation.receiverChannels
nCols = rNx*2
nSpokes = dset.number_of_acquisitions()
print("Raw data size = [", nCols, ", " , nSpokes, ", ", nCoils, "], (COL SPOKE CHA)")

print("Generating radial trajectory co-ordinates...")
traj = generate_golden_radial_trajectory(nCols,nSpokes)
traj = traj.astype(np.single)

print("Loading radial data from ISMRMRD format")
raw_data = np.zeros((nCoils, nCols, nSpokes), dtype=np.complex64)
for acqnum in range(nSpokes):
    if (np.mod(acqnum,1000)==0):
        print("Loading radial data: "+"{:.2f}".format(100*acqnum/nSpokes)+"% complete")
    acq = dset.read_acquisition(acqnum)
    raw_data[:,:,acqnum]=acq.data

kspace_data = raw_data.transpose((1,2,0))
print(kspace_data.shape)

# add by chenpeng
ksp_file = 'radial_data/ksp.npy'
coord_file = 'radial_data/coord.npy'

np.save(ksp_file, kspace_data)
np.save(coord_file, traj)
