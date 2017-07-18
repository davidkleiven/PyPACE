import sys
sys.path.append("pypace")
import missingData as md
import numpy as np
from matplotlib import pyplot as plt
import pickle as pck

def main():
    useCubicMask = False
    ds = 1
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"
    scattered = np.load( kspace )
    realsp = np.load( reconstruct )
    realspPadded = np.zeros(scattered.shape)
    start = int( scattered.shape[0]/4 )
    end = int( 3*scattered.shape[0]/4 )

    realspPadded[start:end,start:end,start:end] = realsp
    mask = np.zeros( scattered.shape, dtype=np.uint8 )
    mask[scattered>1E-16*scattered.max()] = 1

    support = np.zeros( realspPadded.shape, dtype=np.uint8 )
    support[realspPadded>1E-6*realspPadded.max()] = 1

    del realsp, realspPadded, kspace

    if ( useCubicMask ):
        mask[:,:,:] = 1
        w = mask.shape[0]/8
        center = mask.shape[0]/2
        start = int(center-w/2)
        end = int(center+w/2)
        mask[start:end,start:end,start:end] = 0
    support = support[::ds,::ds,::ds]
    mask = mask[::ds,::ds,::ds]
    mdata = md.MissingDataAnalyzer( mask, support )
    mdata.solve( niter=3000 )

if __name__ == "__main__":
    main()
