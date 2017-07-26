import numpy as np

class InitialGenerator( object ):
    """
    Generates initial conditions for the :class:'MissingDataAnalyzer'

    shape: list
        List of the number of elements along each dimension. Typically [N,N,N] where
        N is the number of elememts in each direction
    """
    def __init__( self, shape ):
        self.shape = shape
        self.initConditions = []

    def generate( self ):
        """
        Generates initial conditions

        Returns: list
            List of 3D arrays representing the inital conditions. Each array has data type np.int8
        """
        newarray = np.zeros( self.shape, dtype=np.int8 )
        center = int( self.shape[0]/2 )
        newarray[:,:,:] = 1
        self.initConditions.append(newarray)

        newarray = np.zeros( self.shape, dtype=np.int8 )
        newarray[:center,:,:] = 1
        newarray[center:,:,:] = -1
        self.initConditions.append(newarray)

        newarray = np.zeros( self.shape, dtype=np.int8 )
        newarray[:,:center,:] = 1
        newarray[:,center:,:] = -1
        self.initConditions.append(newarray)

        newarray = np.zeros( self.shape, dtype=np.int8 )
        newarray[:,:,:center] = 1
        newarray[:,:,center:] = -1
        self.initConditions.append(newarray)
        self.addQuads()
        self.addOct()
        return self.initConditions

    def addQuads( self ):
        """
        Add quadrupole contributions
        """
        newarray = np.zeros( self.shape, dtype=np.int8 )
        center = int( self.shape[0]/2 )
        newarray[:center,:center,:] = 1
        newarray[:center,center:,:] = -1
        newarray[center:,:center,:] = -1
        newarray[center:,center:,:] = 1
        self.initConditions.append(newarray)

        newarray = np.zeros( self.shape, dtype=np.int8 )
        center = int( self.shape[0]/2 )
        newarray[:center,:,:center] = 1
        newarray[:center,:,center:] = -1
        newarray[center:,:,:center] = -1
        newarray[center:,:,center:] = 1
        self.initConditions.append(newarray)

        newarray = np.zeros( self.shape, dtype=np.int8 )
        center = int( self.shape[0]/2 )
        newarray[:,:center,:center] = 1
        newarray[:,:center,center:] = -1
        newarray[:,center:,:center] = -1
        newarray[:,center:,center:] = 1
        self.initConditions.append(newarray)

    def addOct( self ):
        """
        Adds octupole contributions
        """
        newarray = np.zeros( self.shape, dtype=np.int8 )
        center = int( self.shape[0]/2 )
        newarray[:center,:center,:center] = 1
        newarray[center:,:center,:center] = -1
        newarray[:center,center:,:center] = -1
        newarray[center:,center:,:center] = 1

        newarray[:center,:center,center:] = -1
        newarray[center:,:center,center:] = 1
        newarray[:center,center:,center:] = 1
        newarray[center:,center:,center:] = -1
        self.initConditions.append( newarray )
