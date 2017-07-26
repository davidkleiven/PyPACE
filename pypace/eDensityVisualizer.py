from __future__ import print_function
import h5py as hf
import numpy as np
try:
    from mayavi import mlab
    haveMayavi = True
except ImportError as exc:
    print (str(exc))
    haveMayavi = False

import h5py as h5
import segmentor as seg
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import categorize as catg
from scipy import ndimage as ndimg
from scipy import misc as msc

class EDensityVisualizer( object ):
    """
    Class for visualizing the fitted real part of the refractive index

    fname: str
        Filename to the HDF5 file containing the fits. Only the best fit from this file is used
    """
    def __init__( self, fname="" ):
        self.fname = fname
        self.ff = None
        self.sliceK = None
        self.mask = None

        if ( fname != "" ):
            with h5.File( fname, 'r' ) as hf:
                if ( not "clusters" in hf.keys() ):
                    raise RuntimeError("No dataset named clusters in the given hdf5 file")
                self.clusters = np.array( hf.get("clusters") )

                if ( "bestFarField" in hf.keys() ):
                    self.ff = np.array( hf.get("bestFarField") )
                if ( "sliceK" in hf.keys() ):
                    self.sliceK = np.array( hf.get("sliceK") )
                if ( "mask" in hf.keys() ):
                    self.mask = np.array( hf.get("mask") )

            self.segdata = np.zeros(self.clusters.shape)
            self.segmentor = seg.Segmentor( self.segdata )
            self.segmentor.clusters = self.clusters

    def getBest( self ):
        """
        Extract the dataset with that best fits the experimental data
        """
        bestVal = np.inf
        bestDset = None
        with h5.File( self.fname, 'r' ) as hf:
            for gname in hf.keys():
                group = hf.get(gname)

                if ( not isinstance(group,h5.Group) ):
                    continue

                for dsetname in group.keys():
                    dset = group.get(dsetname)
                    if ( dset.attrs["cost"] < bestVal ):
                        bestVal = dset.attrs["cost"]
                        bestDset = np.array(dset)
        if ( bestDset is None ):
            raise ValueError("Best dataset is None, meaning that an error occured when searching for the best dataset")
        return bestDset

    def getEdensity( self, dset ):
        """
        Get the real part of the refractive index

        dset: 1D array
            Array containing the mean value in each cluster

        Return: ndarray
            3D array where the value in each cluster is replaced by its mean
        """
        self.segmentor.means = dset

        self.segmentor.replaceDataWithMeans()
        return self.segmentor.data

    def plotOutline( self, data=None ):
        """
        Plot the surface of the fitted electron density

        data: ndarray
            If given this dataset is used. Otherwise the best dataset from the fitting is used
        """
        if ( data is None ):
            bestDset = self.getBest()
            edensity = self.getEdensity(bestDset)
        else:
            edensity = data

        if ( haveMayavi ):
            fig = mlab.figure( bgcolor=(0,0,0) )
            src = mlab.pipeline.scalar_field(edensity)
            #vol = mlab.pipeline.volume( src )
            mlab.pipeline.contour_surface( src )

    def plotBest( self, data=None ):
        """
        Create a 3D plot data

        data: ndarray
            If given this data is used. Otherwise the best dataset from the fitting is used
        """
        if ( data is None ):
            bestDset = self.getBest()
            edensity = self.getEdensity(bestDset)
        else:
            edensity = data

        if ( haveMayavi ):
            fig = mlab.figure( bgcolor=(0,0,0) )
            # Create 3D plot using Mayavi
            src = mlab.pipeline.scalar_field(edensity)
            #mlab.pipeline.iso_surface( src, opacity=0.1 )
            mlab.pipeline.scalar_cut_plane( src, colormap="plasma" )
            mlab.pipeline.scalar_cut_plane( src, plane_orientation="y_axes", colormap="plasma" )
            mlab.pipeline.scalar_cut_plane( src, plane_orientation="z_axes", colormap="plasma" )

    def plotCluster( self, id, downsample=4 ):
        """
        Plot clusters

        id: int
            ID of the cluster to plot

        downsample: int
            Factor to downsample the array in each direction.
            This makes Mayavi faster.
        """
        if ( haveMayavi ):
            fig = mlab.figure(bgcolor=(0,0,0))
            mask = np.zeros(self.clusters.shape, dtype=np.uint8)
            mask[self.clusters==id] = 1
            mask = mask[::downsample,::downsample,::downsample]
            src = mlab.pipeline.scalar_field(mask)
            vol = mlab.pipeline.volume( src )
            mlab.pipeline.threshold( vol, low=0.5 )

    def plotBestRadialAveragedDensity( self ):
        """
        Plot the radial averaged density

        Returns: Matplotlib figure
            Figure where the plot is
        """
        bestDset = self.getBest()
        edensity = self.getEdensity(bestDset)
        Nbins=300
        rbins = np.linspace(0.0, 1.0, Nbins)
        radialMean = catg.radialMean( edensity, Nbins )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(rbins, radialMean )
        ax.set_xlabel("Radial position")
        return fig

    def plot1DAngles( self, anglesDeg, data=None ):
        """
        Plot the density along different lines in the XY plane

        anglesDeg: list
            List if angles at which to plot the electron density along

        data: ndarray
            If given this dataset is used. Otherwise the best dataset from the fit is used

        Returns: Matplotlib figure
            Figure of the plot
        """
        if ( data is None ):
            bestDset = self.getBest()
            edensity = self.getEdensity(bestDset)
        else:
            edensity = data

        N = edensity.shape[0]
        x = np.linspace( -N/2, N/2, N )
        y = np.linspace( -N/2, N/2, N )
        xy = np.vstack( (x,y) )
        rotMatrix = np.eye(2)
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        angles = np.array( anglesDeg )*np.pi/180.0
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf", "#999999"]
        counter = 0
        for angle in angles:
            # XY-plane
            z = 0.0
            rotMatrix[0,0] = np.cos(angle)
            rotMatrix[0,1] = np.sin(angle)
            rotMatrix[1,0] = -np.sin(angle)
            rotMatrix[1,1] = np.cos(angle)
            xyprime = rotMatrix.dot(xy)
            data = edensity[:,:,N/2]
            lineData = ndimg.map_coordinates(data,N/2+xyprime)*1E6
            ax1.plot( lineData, color=colors[counter%len(colors)], label="%d"%(angle*180/np.pi))
            counter += 1
        ax1.legend(loc="best", frameon=False, labelspacing=0.05)
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Density (a.u.)")
        return fig

    def plotFit( self ):
        """
        Compare the simulated and the measured scattering pattern

        Returns: Matplotlib figure
            Figure of the plots
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        if ( not self.sliceK is None ):
            ax1.imshow( self.sliceK, cmap="nipy_spectral", interpolation="none" )
        ax2 = fig.add_subplot(2,3,2)
        if ( not self.ff is None ):
            ax2.imshow( self.ff, cmap="nipy_spectral", interpolation="none")
        ax3 = fig.add_subplot(2,3,3)

        hasMeasuredAndSimulated = not self.sliceK is None and not self.ff is None
        if ( hasMeasuredAndSimulated ):
            diff = np.abs( self.sliceK - self.ff )
            if ( not self.mask is None ):
                diff[self.mask==0] = np.nan
            ax3.imshow( diff, cmap="nipy_spectral", interpolation="none")

        ax4 = fig.add_subplot(2,3,4)
        center = int( self.sliceK.shape[0]/2 )
        if ( hasMeasuredAndSimulated ):
            ax4.plot( self.sliceK[:,center], color="#e41a1c", label="Meas" )
            ax4.plot( self.ff[:,center], color="#377eb8", label="Fit" )
            ax4.legend( loc="best", frameon=False, labelspacing=0.05)

        ax5 = fig.add_subplot(2,3,5)
        if ( hasMeasuredAndSimulated ):
            ax5.plot( self.sliceK[center,:], color="#e41a1c", label="Meas" )
            ax5.plot( self.ff[center,:], color="#377eb8", label="Fit" )
            ax5.legend( loc="best", frameon=False, labelspacing=0.05)
        ax6 = fig.add_subplot(2,3,6)
        if ( not self.mask is None ):
            ax6.imshow( self.mask, cmap="bone", interpolation="none" )
        return fig
