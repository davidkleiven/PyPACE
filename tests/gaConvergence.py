import sys
sys.path.append("pypace")
import unittest
import geneticAlgorithm as ga
from mpi4py import MPI
import densityCorrector as dc
import numpy as np
from matplotlib import pyplot as plt

class DCTester( dc.DensityCorrector ):
    def __init__( self ):
        super().__init__( "testData/realspace.npy", "testData/kspace.npy", 0.17, 55.2 )
        self.x = np.linspace(-1.0,1.0,101)
        self.target = np.sin(self.x) + np.sin(2*self.x) + np.sin(3*self.x) + np.sin(4*self.x)
        self.legendreCoeff = np.zeros(10)

    def costFunction( self ):
        leg = np.polynomial.legendre.Legendre(self.legendreCoeff)
        return np.sum( (leg(self.x) - self.target)**2 )

class GATest( ga.GeneticAlgorithm ):
    def __init__(self, dc, maxValue, comm ):
        super().__init__( dc, maxValue, comm, 10000, saveInterval=100 )
        self.maxValue = maxValue
        self.nPopulations = 10*len(dc.legendreCoeff )
        self.nGenes = len(dc.legendreCoeff)
        self.population = np.random.rand(self.nPopulations,self.nGenes)*maxValue*2-maxValue
        self.fitness = np.zeros( self.nPopulations )
        self.bestIndividuals = np.zeros((self.nGenerations,self.nGenes))
        self.parentHistogram = np.zeros( self.nPopulations )

    def computeFitness( self, step ):
        for i in range(0,self.nPopulations):
            self.dc.legendreCoeff = self.population[i,:]
            self.fitness[i] = 1.0/self.dc.costFunction()

    def plot( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.dc.x, self.dc.target, '.')
        leg = np.polynomial.legendre.Legendre(self.dc.legendreCoeff)
        ax.plot( self.dc.x, leg(self.dc.x), 'b' )

        # Fit legendre
        legFit = leg.fit( self.dc.x, self.dc.target, len(self.dc.legendreCoeff) )
        ax.plot( self.dc.x, legFit(self.dc.x), 'r' )


def main():
    comm = MPI.COMM_WORLD
    dc = DCTester()
    dc.segment(4)
    ga = GATest( dc, 20.0, comm, )
    ga.run(1.0)
    print ("Legendre coefficients:")
    print (ga.dc.legendreCoeff)
    ga.plot()
    ga.plotParentHistogram()
    ga.plotDiversity()
    plt.show()

if __name__ == "__main__":
    main()
