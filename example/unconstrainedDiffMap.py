import sys
sys.path.append("pypace")
import missingData as md
import numpy as np
import matplotlib as mpl
import config
if ( not config.enableShow ):
    mpl.use("Agg")
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import pickle as pck
import copy
import missingDataInitGenerator as mdig
import h5py as h5
import time

NUMBER_OF_ITERATIONS=1000
RELATIVE_ERROR = 5E-2
reconstruct = "data/average_NiAu_sample1_3D_50_1.npy"
kspace = "data/NiAu_sample1_3D.npy"

def main():
    timestamp = str( time.strftime("%Y_%m_%d_%H_%M_%S") )
    print ("Started on %s"%(timestamp))
    useCubicMask = False
    ds = 1
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

    del realsp, realspPadded

    if ( useCubicMask ):
        mask[:,:,:] = 1
        w = mask.shape[0]/4
        center = mask.shape[0]/2
        start = int(center-w/2)
        end = int(center+w/2)
        mask[start:end,start:end,start:end] = 0
    support = support[::ds,::ds,::ds]
    mask = mask[::ds,::ds,::ds]
    #mdata = md.MissingDataAnalyzer( mask, support )
    #constraints = [md.FourierConstraint(mdata),md.RealSpaceConstraint(mdata)]#,md.NormalizationConstraint(mdata)]

    nfunctions = 1
    initGenerator = mdig.InitialGenerator( mask.shape )
    initconditions = initGenerator.generate()
    images = []
    orthogonalConstraints = []
    for i in range( len(initconditions) ):
        mdata = md.MissingDataAnalyzer( mask, support )
        constraints = [md.FourierConstraint(mdata),md.RealSpaceConstraint(mdata)]#,md.NormalizationConstraint(mdata)]
        #constraints += orthogonalConstraints

        result = mdata.solve( constraints, niter=NUMBER_OF_ITERATIONS, relerror=RELATIVE_ERROR, show=False, initial=initconditions[i].astype(np.float64),
        zeroLimit=1E-4 )

        fig = mdata.plot( mdata.getImg() )
        fig.savefig( "data/unconstrained%d_%s.svg"%(i,timestamp) )
        #orthogonalConstraints.append( md.OrthogonalConstraint( mdata, copy.deepcopy(mdata.getImg()) ) )
        images.append( copy.deepcopy(result) )
        mdata.allErrors = []
        mdata.bestError = 1E30

    # Save the unconstrained modes
    with h5.File( "data/unconstrainedModes%s.h5"%(timestamp), 'w' ) as hf:
        hf.create_dataset("support", data=support)
        hf.create_dataset("mask", data=mask)
        for i in range( len(images) ):
            group = hf.create_group("mode%d"%(i) )
            group.create_dataset( "img", data=images[i]["image"] )
            group.create_dataset( "error", data=images[i]["error"] )
            group.attrs["status"] = images[i]["status"]
            group.attrs["message"] = images[i]["message"]
            group.attrs["cnstpow"] = images[i]["constrainedPower"]
            group.attrs["besterr"] = images[i]["bestError"]

if __name__ == "__main__":
    main()
