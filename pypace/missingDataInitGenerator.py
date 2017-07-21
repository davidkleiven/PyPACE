import numpy as np

class InitialGenerator( object ):
    def __init__( self, shape ):
        self.shape = shape
        self.initConditions = []

    def generate( self ):
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
