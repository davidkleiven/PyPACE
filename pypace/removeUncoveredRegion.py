import numpy as np
import h5py as h5
import missingData as mdata
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

class RemoveUncovered( object ):
    def __init__( self, reconstructed, fname ):
        self.realspace = reconstructed

        self.modes = []
        with h5.File( fname, 'r' ) as hf:
            self.mask = np.array( hf.get("mask") )
            self.support = np.array( hf.get("support") )
            for key in hf.keys():
                group = hf.get(key)
                if ( isinstance(group,h5.Group) ):
                    self.modes.append( np.array( group.get("img")) )

        self.makeOrthogonal()

    def makeOrthogonal( self ):
        if ( len(self.modes) <= 1 ):
            print ("Less than one mode. Nothing to do.")
            return

        # Normalize the modes
        for mode in self.modes:
            mode /= np.sqrt( np.sum(mode**2) )

        # Perform Gram-Schmidt
        for i in range(1,len(self.modes)):
            for j in range(0,i):
                projIJ = np.sum( self.modes[i]*self.modes[j] )
                self.modes[i] -= projIJ*self.modes[j]
            self.modes[i] /= np.sqrt( np.sum(self.modes[i]**2) )
        return self.modes

    def removeUncoveredFeatures( self ):
        for mode in self.modes:
            proj = np.sum( self.realspace*mode )
            self.realspace -= proj*mode
        return self.realspace

    def plot( self ):
        #mask = np.load("maskTest.npy")
        #support = np.load("supportTest.npy")
        self.mask = np.fft.fftshift(mask)
        md = mdata.MissingDataAnalyzer( self.mask, self.support )
        counter = 0
        for mode in self.modes:
            print( md.computeConstrainedPower(mode) )
            fig = md.plot( mode )
            fig.savefig("data/orthogMode%d.svg"%(counter))
            counter += 1
            plt.show()
