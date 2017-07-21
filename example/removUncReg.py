import sys
sys.path.append("pypace")
import numpy as np
import removeUncoveredRegion as rur

def main():
    fname = "data/Run2p5/unconstrainedModes2017_07_20_15_04_12.h5"
    remover = rur.RemoveUncovered( None, fname )
    remover.plot()

if __name__ == "__main__":
    main()
