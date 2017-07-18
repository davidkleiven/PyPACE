import sys
sys.path.append("pypace")
import missingData as md
import numpy as np
from matplotlib import pyplot as plt
import pickle as pck

def main():
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

    mdata = md.MissingDataAnalyzer( mask, support )
    mdata.solve()

if __name__ == "__main__":
    main()
