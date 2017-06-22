from __future__ import print_function
import numpy as np
import densityCorrector as dc
from mpi4py import MPI
import pickle as pck
import copy

class GeneticAlgorithm(object):
    def __init__( self, densCorr, maxValue, comm, nGenerations, debug=False ):
        if ( not isinstance(densCorr,dc.DensityCorrector) ):
            raise TypeError("Genetic Algorithm requires a DensityCorrector object")
        self.dc = densCorr
        self.maxValue = 0.0
        self.nPopulations = 10*len(self.dc.segmentor.means)
        # Round this number to be an integer number of the number of processes
        self.nPopulations = int(1+self.nPopulations/comm.size)*comm.size
        self.nGenes = len(self.dc.segmentor.means)-1 # -1: The cluster corresponding to the outer is forced to be zero
        self.population = np.random.rand(self.nPopulations,self.nGenes)*maxValue
        self.numberOfGenesToMutate = int(self.nGenes*self.nPopulations*0.001)
        self.nGenerations = nGenerations
        if ( self.numberOfGenesToMutate == 0 ):
            self.numberOfGenesToMutate = 1
        self.fitness = np.zeros(self.nPopulations)
        self.comm = comm
        self.bestIndividuals = np.zeros((self.nGenerations,self.nGenes))
        self.currentGeneration = 0
        self.debug = debug
        self.printStatusMessage = True

    def computeFitness( self, angleStepDeg ):
        """
        Compute the fitness factor for all the populations
        """
        self.comm.Barrier() # All process wait until all reach this point
        self.population = self.comm.bcast( self.population, root=0 )
        self.fitness = np.zeros(len(self.fitness))
        nPopPerProc = int(self.nPopulations/self.comm.size)
        start = self.comm.Get_rank()*nPopPerProc
        end = self.comm.Get_rank()*nPopPerProc+nPopPerProc
        for i in range(start,end):
            if ( self.debug ):
                print ("Rank %d: Computing fitness factor for individual %d"%(self.comm.Get_rank(),i))
            if ( self.comm.Get_rank() == 0 and self.printStatusMessage ):
                print ("Generation %d, %.1f"%(self.currentGeneration,i*100/end))
            # Insert the means in to the clusters
            self.dc.segmentor.means[0] = 0.0 # Force the region surrounding the scatterer to have delta = 0
            self.dc.segmentor.means[1:] = self.population[i,:]
            self.dc.segmentor.replaceDataWithMeans()
            self.dc.buildKspace( angleStepDeg )
            self.fitness[i] = 1.0/self.dc.costFunction()

        # Collect the fitness from the other processes
        dest = np.zeros(self.fitness.shape)
        self.comm.Reduce( self.fitness, dest, op=MPI.SUM, root=0 )
        self.fitness = dest

    def getParents( self ):
        """
        Select parents using roulette search
        """
        S = np.sum(self.fitness)
        cumsum = 0.0
        randnum = np.random.rand()*S
        for i in range(0,len(self.fitness)):
            cumsum += self.fitness[i]
            if ( cumsum > randnum ):
                parent1 = i
                oldFitParam = self.fitness[parent1]
                self.fitness[parent1] = 0.0
                break

        cumsum = 0.0
        S = np.sum( self.fitness )
        randnum = np.random.rand()*S
        for i in range(0,len(self.fitness)):
            cumsum += self.fitness[i]
            if ( cumsum > randnum ):
                parent2 = i
                self.fitness[parent1] = oldFitParam
                return parent1, parent2
        raise Exception("An unexpected error occured: Could not find parents!")

    def reproduce( self ):
        """
        Create a new generation by single point cross over from two parents
        """
        copyPop = copy.deepcopy(self.population)
        for i in range(self.nPopulations):
            parent1,parent2 = self.getParents()
            crossPoint = np.random.randint(low=0,high=self.nGenes)
            self.population[i,:crossPoint] = copyPop[parent1,:crossPoint]
            self.population[i,crossPoint:] = copyPop[parent2,crossPoint:]

    def mutate( self ):
        """
        Mutate some genes by assigning a random value
        """
        genesToMutate = np.random.randint(low=0,high=self.nPopulations*self.nGenes, size=self.numberGenesOfToMutate)
        for num in genesToMutate:
            individual = int( num/self.nPopulations )
            gene = num%self.nGenes
            self.population[individual,gene] = np.random.rand()*self.maxValue

    def getBestIndividual( self ):
        """
        Returns the individual with the best fitness factor
        """
        indx = np.argmax(self.fitness)
        return self.population[indx,:]

    def evolveOneGeneration( self, angleStepKspace ):
        """
        Evolves the GA by one generation
        """
        self.computeFitness( angleStepKspace )
        if ( self.comm.Get_rank() == 0 ):
            print ("Best fitness: %.2E"%(np.max(self.fitness)))
            print ("Worst fitness: %.2E"%(np.min(self.fitness)))
            self.bestIndividuals[self.currentGeneration,:] = self.getBestIndividual()
            self.reproduce()
            self.mutate()
            self.currentGeneration += 1
            fname = "bestIndividuals.csv"
            np.savetxt( fname , self.bestIndividuals[:self.currentGeneration,:], delimiter="," )
            print ("Best individual in each generation written to %s"%(fname))

    def run( self, angleStepKspace ):
        """
        Runs GA for a given number of generations
        """
        if ( self.comm.Get_rank() == 0 ):
            print ("Starting the Genetic Algorithm...")
        for i in range(self.nGenerations):
            self.evolveOneGeneration( angleStepKspace )
