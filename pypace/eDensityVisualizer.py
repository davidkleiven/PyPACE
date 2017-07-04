from __future__ import print_function
import h5py as hf
import numpy as np
from mayavi import mlab
import h5py as h5
import segmentor as seg
from matplotlib import pyplot as plt
import categorize as catg
from scipy import ndimage as ndimg

class EDensityVisualizer( object ):
    def __init__( self, fname ):
        self.fname = fname
        with h5.File( fname, 'r' ) as hf:
            if ( not "clusters" in hf.keys() ):
                raise RuntimeError("No dataset named clusters in the given hdf5 file")
            self.clusters = np.array( hf.get("clusters") )

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
        """
        self.segmentor.means = dset

        self.segmentor.replaceDataWithMeans()
        return self.segmentor.data

    def plotBest( self ):
        """
        Create a 3D plot of the best fit
        """
        bestDset = self.getBest()
        edensity = self.getEdensity(bestDset)

        fig = mlab.figure( bgcolor=(0,0,0) )
        # Create 3D plot using Mayavi
        src = mlab.pipeline.scalar_field(edensity)
        #mlab.pipeline.iso_surface( src, opacity=0.1 )
        mlab.pipeline.scalar_cut_plane( src, colormap="plasma" )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="y_axes", colormap="plasma" )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="z_axes", colormap="plasma" )

    def plotCluster( self, id ):
        """
        Plot clusters
        """
        fig = mlab.figure(bgcolor=(0,0,0))
        mask = np.zeros(self.clusters.shape, dtype=np.uint8)
        mask[self.clusters==id] = 1
        mask = mask[::4,::4,::4]
        src = mlab.pipeline.scalar_field(mask)
        vol = mlab.pipeline.volume( src )
        mlab.pipeline.threshold( vol, low=0.5 )

    def plotBestRadialAveragedDensity( self ):
        """
        Plot the radial averaged density
        """
        bestDset = self.getBest()
        edensity = self.getEdensity(bestDset)
        Nbins=80
        rbins = np.linspace(0.0, 1.0, Nbins)
        radialMean = catg.radialMean( edensity, Nbins )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(rbins, radialMean )
        return fig

    def plot1DAngles( self, anglesDeg, data=None ):
        """
        Plot the density along different lines in the XY plane
        """
        if ( data is None ):
            bestDset = self.getBest()
            edensity = self.getEdensity(bestDset)
        else:
            edensity = data

        N = edensity.shape[0]
        x = np.linspace(-N/2, N/2, N )
        y = np.linspace(-N/2, N/2, N)
        xy = np.vstack((x,y))
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
        return fig
