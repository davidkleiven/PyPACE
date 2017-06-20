import sys
sys.path.append("../pypace")
sys.path.append("pypace")
import densityCorrector as dc
from matplotlib import pyplot as plt

def main():
    reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
    kspace = "data/NiAu_sample1_3D.npy"

    dCorr = dc.DensityCorrector( reconstruct, kspace )
    dCorr.plotRec()
    dCorr.plotKspace()
    plt.show()

if __name__ == "__main__":
    main()
