from __future__ import print_function
import numpy as np
import densityCorrector as dc
from mpi4py import MPI
import pickle as pck
import copy
import config
import matplotlib as mpl
if ( not config.enableShow ):
    mpl.use("Agg")
from matplotlib import pyplot as plt

class GACouldNotFindParentsError(Exception):
    pass

class GeneticAlgorithm(object):
    def __init__( self, densCorr, maxValue, comm, nGenerations, debug=False, saveInterval=1 ):
        if ( not isinstance(densCorr,dc.DensityCorrector) ):
            raise TypeError("Genetic Algorithm requires a DensityCorrector object")
        self.dc = densCorr
        self.maxValue = 0.0
        self.nPopulations = 10*len(self.dc.segmentor.means)
        # Round this number to be an integer number of the number of processes
        self.nPopulations = int(1+self.nPopulations/comm.size)*comm.size
        self.nGenes = len(self.dc.segmentor.means)-1 # -1: The cluster corresponding to the outer is forced to be zero
        self.population = np.random.rand(self.nPopulations,self.nGenes)*maxValue
        self.numberOfGenesToMutate = int(self.nGenes*self.nPopulations*0.01)
        self.nGenerations = nGenerations
        if ( self.numberOfGenesToMutate == 0 ):
            self.numberOfGenesToMutate = 1
        self.fitness = np.zeros(self.nPopulations)
        self.comm = comm
        self.bestIndividuals = np.zeros((self.nGenerations,self.nGenes))
        self.currentGeneration = 0
        self.debug = debug
        self.printStatusMessage = True
        self.saveInterval = saveInterval
        self.parentHistogram = np.zeros(self.nPopulations)
        self.diversity = np.zeros(self.nGenerations)

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
            self.fitness[i] = 1.0/self.dc.costFunction() # Use the negative of the cross cost function --> value with smallest cost gets the largest fitness

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
            if ( cumsum >= randnum ):
                parent1 = i
                oldFitnessParam = self.fitness[parent1]
                self.fitness[parent1] = self.fitness.min()
                break

        cumsum = 0.0
        S = np.sum( self.fitness )
        randnum = np.random.rand()*S
        for i in range(0,len(self.fitness)):
            cumsum += self.fitness[i]
            if ( cumsum >= randnum ):
                parent2 = i
                self.fitness[parent1] = oldFitnessParam
                return parent1, parent2
        raise RuntimeError("Did not manage to find parents!")
        #raise GACouldNotFindParentsError("An unexpected error occured: Could not find parents!")

    def rankSelection( self ):
        """
        Select parents with rank selection
        """
        rank = np.argsort(self.fitness)
        N = len(rank)
        maxval = N*(N+1)/2
        randnum = np.random.rand()*(maxval)
        cumsum = 0
        for i in range(0,len(rank)):
            #cumsum += (len(rank)-i)
            cumsum += (i+1)
            if ( cumsum >= randnum ):
                parent1 = rank[i]
                break

        #maxval -= rank[parent1]
        parent2 = parent1
        while ( parent2 == parent1 ):
            randnum = np.random.rand()*maxval
            cumsum = 0.0
            for i in range(0,len(rank)):
                #cumsum += (len(rank)-i)
                cumsum += (i+1)
                if ( cumsum >= randnum ):
                    parent2 = rank[i]
                    break
        return parent1,parent2

    def reproduce( self ):
        """
        Create a new generation by single point cross over from two parents
        """
        copyPop = copy.deepcopy(self.population)
        # First find the two best solutions and pass them to the next generation
        Npass = 3
        oldBestFitness = np.zeros(Npass)
        bestIndx = np.zeros(Npass,dtype=np.int32)
        # Extract the best individuals
        for i in range(0,Npass):
            best = np.argmax(self.fitness)
            bestIndx[i] = best
            oldBestFitness[i] = self.fitness[best]
            self.fitness[best] = self.fitness.min()
            self.population[i,:] = copyPop[best,:]

        # Insert the fitness back into the fitness array
        for i in range(0,Npass):
            self.fitness[bestIndx[i]] = oldBestFitness[i]

        #self.pointCrossOver( Npass, copyPop )
        self.uniformCrossOver( Npass, copyPop )

    def mutate( self ):
        """
        Mutate some genes by assigning a random value
        """
        genesToMutate = np.random.randint(low=0,high=self.nPopulations*self.nGenes, size=self.numberOfGenesToMutate)
        for num in genesToMutate:
            individual = int( num/self.nPopulations )
            gene = num%self.nGenes
            self.population[individual,gene] = np.random.rand()*self.maxValue

    #def mutate( self ):
    #    """
    #    Mutate some genes by assigning a random value, but ensure that the best solution is not altered
    #    """
    #    Ngenes = 5
    #    mutants = np.random.randint(low=0,high=self.nPopulations, size=Ngenes)
    #    best = np.argmax(self.fitness)
    #    for ind in mutants:
    #        if ( ind == best ):
    #            continue
    #        self.population[ind,:] += np.random.normal(0.0,0.2*self.maxValue,size=self.nGenes)


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
            self.updateDiversity()
            self.reproduce()
            self.mutate()
            self.currentGeneration += 1
            if ( self.currentGeneration%self.saveInterval == 0 ):
                fname = "bestIndividuals.csv"
                np.savetxt( fname , self.bestIndividuals[:self.currentGeneration,:], delimiter="," )
                np.savetxt( "data/lastGeneration%d.csv"%(self.currentGeneration), self.population, delimiter="," )
                print ("Best individual in each generation written to %s"%(fname))

    def pointCrossOver( self, start, copyPop ):
        """
        Create child by a single point cross over
        """
        # Create the rest of the generation based on parents
        for i in range(start, self.nPopulations):
            #parent1,parent2 = self.getParents()
            parent1,parent2 = self.rankSelection()
            self.parentHistogram[parent1] += 1
            self.parentHistogram[parent2] += 1
            crossPoint = np.random.randint(low=0,high=self.nGenes)
            self.population[i,:crossPoint] = copyPop[parent1,:crossPoint]
            self.population[i,crossPoint:] = copyPop[parent2,crossPoint:]

    def uniformCrossOver( self, start, copyPop ):
        """
        Create child with uniform cross over
        """
        for i in range(start,self.nPopulations ):
            parent1,parent2 = self.rankSelection()
            #print ("Parent fitness: ", self.fitness[parent1],self.fitness[parent2])
            self.parentHistogram[parent1] += 1
            self.parentHistogram[parent2] += 1
            for j in range(0,self.nGenes):
                parent = None
                if ( np.random.rand() >= 0.5 ):
                    parent = parent1
                else:
                    parent = parent2
                self.population[i,j] = copyPop[parent,j]

    def run( self, angleStepKspace ):
        """
        Runs GA for a given number of generations
        """
        if ( self.comm.Get_rank() == 0 ):
            print ("Starting the Genetic Algorithm...")
        for i in range(self.nGenerations):
            self.evolveOneGeneration( angleStepKspace )

    def updateDiversity( self ):
        self.diversity[self.currentGeneration] = self.getDiversity()

    def getDiversity( self ):
        """
        Calculate the average standard deviation in the entire population
        """
        div = 0
        for i in range( 0, self.nGenes ):
            div += np.std(self.population[:,i])
        return div/(self.nGenes*self.maxValue)

    def plotParentHistogram( self ):
        """
        Plots the histogram over parents
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.parentHistogram/np.sum(self.parentHistogram), ls="steps")

    def plotDiversity( self ):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot( self.diversity )
