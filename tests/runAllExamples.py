import sys
sys.path.append("example")
import unittest
import os

class AllExamples( unittest.TestCase ):
    def test_azmExample( self ):
        print ("Testing azmExample")
        try:
            import azmExample as azm
            azm.reconstruct = "testData/realspace.npy"
            azm.kspace = "testData/kspace.npy"
            azm.haveMayavi = False # Gets stuck if this is Ture, appears to only happen in the test case
            azm.main( ["example/azmTestParams.json"] )
        except Exception as exc:
            print (str(exc))
            self.fail(str(exc))

    def test_plotKspace( self ):
        print ("Testing plotKspace")
        try:
            import plotKpace as pkp
            pkp.reconstruct = "testData/realspace.npy"
            pkp.kspace = "testData/kspace.npy"
            pkp.main()
        except Exception as exc:
            print (str(exc))
            self.fail(str(exc))

    def test_removeUncRegion( self ):
        print ("Testing removeUncRegion")
        try:
            import removUncReg as rur
            rur.main( ["--testImports"] )
        except Exception as exc:
            print (str(exc))
            self.fail(str(exc))

    def test_showCorrected( self ):
        print ("Testing showCorrected")
        try:
            import showCorrected as sc
            sc.main( ["--testImports"] )
        except Exception as exc:
            self.fail(str(exc))

    def test_ugelstadSphere( self ):
        print ("Testing ugelstad sphere")
        try:
            import ugelstadSphere as us
            us.reconstruct = "testData/realspace.npy"
            us.kspace = "testData/kspace.npy"
            us.main(["--testFit"])
        except Exception as exc:
            print (str(exc))
            self.fail(str(exc))

    def test_unconstrainedDiffMap( self ):
        print ("Testing uncosntrainedDiffMap")
        try:
            import unconstrainedDiffMap as udm
            udm.NUMBER_OF_ITERATIONS = 1
            udm.RELATIVE_ERROR = 5E2
            udm.reconstruct = "testData/realspace.npy"
            udm.kspace = "testData/kspace.npy"
            udm.main()
        except Exception as exc:
            print (str(exc))
            self.fail(str(exc))

if __name__ == "__main__":
    unittest.main()
