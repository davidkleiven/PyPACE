import unittest
import sys
sys.path.append("../pyREC")
sys.path.append("pyREC")
import numpy as np
import objectToScatteredTransformer as otst

class TestRytovBackForw( unittest.TestCase ):
    def testUnmodifiedBF( self ):
        noThrow = True
        msg = ""
        try:
            ksp = np.load( "pyREC/kspaceCoatedSphere3D.npy" )
            rytov = otst.Rytov( ksp, 1.0 )
            rytov.backward()
            rytov.forward()
            rytov.backward()
            rytov.forward()
            rytov.backward()
            rytov.forward()
            diff = np.sqrt( np.sum( (ksp-np.real(rytov.scatteredData))**2 ) )
        except Exception as exc:
            noThrow = False
            msg = str(exc)
        self.assertTrue( noThrow, msg)
        self.assertAlmostEqual( diff, 0.0, places=3 )

if __name__ == "__main__":
    unittest.main()
