import sys
sys.path.append("pypace")
import numpy as np
import removeUncoveredRegion as rur

def main():
    fname = "data/Run5pFull/unconstrainedModes2017_07_22_11_33_40.h5"
    remover = rur.RemoveUncovered( None, fname )
    remover.plot()

if __name__ == "__main__":
    main()
