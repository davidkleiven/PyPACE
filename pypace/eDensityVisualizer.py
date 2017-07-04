from __future__ import print_function
import h5py as hf
import numpy as np
from mayavi import mlab
import h5py as h5
import segmentor as seg
from matplotlib import pyplot as plt
import categorize as catg

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
        self.segmentor.means = dset

        self.segmentor.replaceDataWithMeans()
        return self.segmentor.data

    def plotBest( self ):
        bestDset = self.getBest()
        edensity = self.getEdensity(bestDset)

        # Create 3D plot using Mayavi
        src = mlab.pipeline.scalar_field(edensity)
        #mlab.pipeline.iso_surface( src, opacity=0.1 )
        mlab.pipeline.scalar_cut_plane( src )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="y_axes" )
        mlab.pipeline.scalar_cut_plane( src, plane_orientation="z_axes" )

    def plotCluster( self, id ):
        mask = np.zeros(self.clusters.shape, dtype=np.uint8)
        mask[self.clusters==id] = 1
        mask = mask[::4,::4,::4]
        src = mlab.pipeline.scalar_field(mask)
        vol = mlab.pipeline.volume( src )
        mlab.pipeline.threshold( vol, low=0.5 )

    def plotBestRadialAveragedDensity( self ):
        bestDset = self.getBest()
        edensity = self.getEdensity(bestDset)
        Nbins=80
        rbins = np.linspace(0.0, 1.0, Nbins)
        radialMean = catg.radialMean( edensity, Nbins )

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(rbins, radialMean )
        return fig
