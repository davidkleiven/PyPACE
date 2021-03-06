import numpy as np
import h5py as h5
import missingData as mdata
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

class RemoveUncovered( object ):
    """
    Class for removing the projection of the reconstructed object that scatters into the region of missing data

    reconstructed: ndarray
        3D array containing the reconstructed object

    fname: str
        Filename to a HDF5 file containing the unconstrained modes
    """
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

        # Assert that the shape of the objects are the same
        self.realspace = np.zeros(self.mask.shape)
        start = int(self.mask.shape[0]/4)
        end = int(3*self.mask.shape[0]/4)
        self.realspace[start:end,start:end,start:end] = reconstructed

    def makeOrthogonal( self ):
        """
        Run the Gram-Schmidt orthogonalization procedure on the orthogonal modes
        """
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

    def projectToScattered( self, asint8=False ):
        """
        Subtract the projection of the reconstructed that scatters into the region of missing data

        asint8: bool
            If True the resulting object will be converted to np.int8

        Returns: ndarray
            3D array representing the corrected object
        """
        if ( self.realspace is None ):
            raise TypeError("No realspace object given")
        for mode in self.modes:
            proj = np.sum( mode*self.realspace )
            self.realspace -= proj*mode

        if ( asint8 ):
            return self.toInt8(self.realspace)
        return self.realspace

    def toInt8( self, data ):
        """
        Convert array to np.int8

        data: float, ndarray
            Array to be converted

        Returns: ndarray
            The converted array
        """
        upper = np.abs(data).max()
        data *= 127/upper
        return data.astype(np.int8)

    def removeUncoveredFeatures( self ):
        """
        Remove features that scatters into the region of missing data
        """
        for mode in self.modes:
            proj = np.sum( self.realspace*mode )
            self.realspace -= proj*mode
        return self.realspace

    def plot( self ):
        """
        Plot the orthogonal unconstrained modes
        """
        #mask = np.load("maskTest.npy")
        #support = np.load("supportTest.npy")
        self.mask = np.fft.fftshift(self.mask)
        md = mdata.MissingDataAnalyzer( self.mask, self.support )
        counter = 0
        for mode in self.modes:
            print( md.computeConstrainedPower(mode) )
            fig = md.plot( mode )
            fig.savefig("data/orthogMode%d.svg"%(counter))
            counter += 1
            plt.show()
