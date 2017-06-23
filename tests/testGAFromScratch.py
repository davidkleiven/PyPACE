import unittest
import sys
sys.path.append("pypace")
import densityCorrector as dc
from mpi4py import MPI
import geneticAlgorithm as ga

class TestRunFromScratch(unittest.TestCase):
    def testNoThrow( self ):
        exceptionRaised = False
        msg = ""
        try:
            comm = MPI.COMM_WORLD
            reconstruct = "testData/realspace.npy"
            kspace = "testData/kspace.npy"
            dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm, debug=False )
            dCorr.fit( 2, nGAgenerations=2, printStatusMessage=False )
        except Exception as exc:
            msg = str(exc)
            exceptionRaised = True
        self.assertFalse( exceptionRaised, msg )

if __name__ == "__main__":
    unittest.main()
