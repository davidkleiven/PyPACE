import unittest
import sys
sys.path.append("pypace")
import densityCorrector as dc
from mpi4py import MPI
import geneticAlgorithm as ga

class TestRunFromScratch(unittest.TestCase):
    def testNoThrow( self ):
        exceptionRaised = False
        try:
            comm = MPI.COMM_WORLD
            reconstruct = "testData/realspace.npy"
            kspace = "testData/kspace.npy"
            dCorr = dc.DensityCorrector( reconstruct, kspace, 0.17, 55.2, comm=comm, debug=False )
            dCorr.fit( 2 )
        except ga.GACouldNotFindParentsError as exc:
            pass
        except Exception as exc:
            exceptionRaised = True
        self.assertFalse( exceptionRaised, str(exc) )

if __name__ == "__main__":
    unittest.main()
