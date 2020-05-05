import numpy as np
import torch.multiprocessing as mp
import torch
import time

#Uses CamelCase all around 
#Note most of the mathematical operations (if not, all) are done on pytorch, just to make sure everythig is Model compatible

#Some Notations and other Subtleties:-
#   1. We assume that the name of the class that has inherited the PTReplicaMetaBase has the name 'MyReplica' in the docstrings.
#   2. Please note the Dosctring of 'InitReplicas'!!
#

#PROCEDURE TO MANUALLY START TRAINING:
#   1. Call InitReplicas
#   2. Call Runner
#   3. Call SwapExecutor
#   4. Call CopyandSetReplicas to generate copied Processes to run again on next iteration.
#   5. Call Runner again
#   6. Repeat from step 3.

#Note that if the first replica seems to be stuck (i.e. noticing a lot of rejections), then keep in mind first replica has temperature = 1K, i.e. exact Posterior Distribution
#(not the flattened one that will appear at high temperatures), thus, there will be a lot of rejections if the starting point is the low probability regions in the posterior.


#Final Wrapper for the Parallel Tempering Class, to execute complete Langevin Gradient Parallel Tempering Algortihm.


class ParallelTempering():
    
    def __init__(self, ReplicaClass, NumReplicas, MaxSamples, SwapInterval, MaxTemp, BetaLadderMethod = 'GEO'):
        
        """
        ReplicaClass : (class) The Class which inherits PTReplicaBaseClass and implements all the needed Abstract Functions.
        NumReplicas : (int) The number of Replicas to have for the algortihm.
        Maxsamples : (int) Maximum no. of Samples from each Replica.
        SwapInterval : (float) If < 1, Then it is the fraction of MaxSamples after which Swap Condition will be checked
                                        and if it's >= 1, then it is the SwapInterval (i.e. Number of Samples before checking for a swap)
        MaxTemp : Maximum Temperature for the Ladder.
        BetaLadderMethod : (str) The method by which the BetaLadder will be constructed. Currently supports 'GEO' for Geometric, 'LIN' for Linear, 'HAR' for Harmonic.
        """
        
        self.ReplicaClass = ReplicaClass
        self.NumReplicas = NumReplicas
        self.MaxSamples = MaxSamples
        self.MaxTemp = MaxTemp
        self.BetaLadderMethod = BetaLadderMethod

        #assert ((SwapInterval <= 1) and (SwapInterval >0)), "SwapInterval should be between 0 and 1"
        if ((SwapInterval < 1) and (SwapInterval > 0)):

            self.NumReplicaSamples = int(SwapInterval * MaxSamples)   #No.of iterations to run each Replica for in each iteration.

        else:

            assert isinstance(SwapInterval, int) == True, "If SwapInterval >= 1, then it should be of type integer."
            self.NumReplicaSamples = SwapInterval

        self.SwapInterval = SwapInterval


        print("Swap Condition will be tested every {} samples.".format(self.NumReplicaSamples))

        self.Temperatures = torch.tensor([1 for _ in range(NumReplicas)], dtype = torch.float64) #Placeholder for Temperatures.
        self.ReplicaList = [None for _ in range(NumReplicas)]

        ################################################################ [DEPRECATED, NOW USES PIPES TO COMMUNICATE BACK] #########################################################
        #SamplesQueueList is an Important Variable in this Implementation, it holds (in the following order) Model's Weights(and Biases), Miscellaneous Param List
        #for all the samples for each replica.
        #self.SamplesQueueList = [mp.Queue() for _ in range(NumReplicas)] 

        #Stores the Samples collected from the Last iteration where each chain collected NumReplicaSamples amount of samples
        self.LastRunSamplesAllReplicas = [ [] for _ in range(NumReplicas) ]


        self.SwapHistory = []

        #Pipes to transfer the Likelihood and Prior Probabilities back ######################### USE PIPES LATER ON TO TRANSFER EVERYTHING TO MAIN PROCESS
        self.PipeList = [mp.Pipe() for _ in range(NumReplicas)]

        #Have Replicas been Initialized? 
        self.isInitReplicaCalled = False
    

        #Final Samples Placeholder, it stores ALL MaxSamples amount of samples collected. It's updated only when Run
        self.AllSamples = [ [] for _ in range(NumReplicas) ]


    def TempLadderInitializer(self):
        
        """
        Creates the Beta Ladder as per the specifications, i.e. BetaLadderMethod, MaxTemp and NumReplicas.
        It updates the Temperature directly.
        
        
        RETURNS : A list of length NumReplicas containing the Temperature assignments.
        """
        
        if self.BetaLadderMethod == 'GEO':
            #Create a Geometric Ladder, with common ratio: r = (1/Tmax)**(1/(nChains - 1))
            # The Betas will be 1,r,r**2, ... , r**(nChains - 1)
            
            if self.NumReplicas == 1:
                self.Temperatures = [1]
            else:
                r = torch.tensor((1/self.MaxTemp) ** (1/(self.NumReplicas - 1)) , dtype = torch.float64 )
                for i in range(self.NumReplicas):
                    self.Temperatures[i] = 1/torch.pow(r,i)
                    
        
        elif self.BetaLadderMethod == 'LIN':
            # Create a Linear Ladder (in A.P.), with common difference: d = (1 - 1/Tmax) * (1 / (nChains - 1))
            # The Betas will be beta_1, beta_2 = beta_1 + d, ... , beta_R
            
            if self.NumReplicas == 1:
                self.Temperatures = [1]
            else:
                d = torch.tensor( (1 - 1/self.MaxTemp) * (1 / (self.NumReplicas - 1)), dtype = torch.float64 )
                for i in range(self.NumReplicas):
                    self.Temperatures[i] = 1/(1 - (i) * d)
            
        elif self.BetaLadderMethod == 'HAR':
            # Create a Harmonic Ladder (Betas in H.P.), which implies that Temperatures themselves are in A.P.
            # with d = (TMax - 1) / (nChains - 1)
            # The Betas will be T_1, T_2 = T_1 + d, ... , T_R
            if self.NumReplicas == 1:
                self.Temperatures = [1]
            else:
                Temp_d = torch.tensor( (self.MaxTemp - 1) / (self.NumReplicas - 1) , dtype = torch.float64) 
                for i in range(self.NumReplicas):
                    self.Temperatures[i] = (1 + (i) * Temp_d)
                    
        else:
            raise ValueError("'{}' is not a valid Method for creating the Beta Ladder, valid forms are 'GEO' for Geometric, 'LIN' for Linear, 'HAR' for Harmonic.".format(self.BetaLadderMethod))

        
        
    def InitReplicas(self, *args):
        
        """
        Call this function to initialize the Replicas by giving it the exactly the same arguments as you would to instantiate an object from 'MyReplica'.
        
        NOTE that, of-course, your temperature assignment and NumReplicas will NOT be used, as the BetaLadderInitializer and PT Class init will take care of that. However, this Temperature assignment has been nonetheless kept as it is in the
        PTReplicaMetaBase, just in the case that you might have the thought of using the individual Replica for testing/other purposes.
        """
        
        self.args = args #Just to store the args for Making more Replicas/Copying.
        
        #Step 1. Calculte and Set the Temperatures, will be needed to alter the Replicas Temperature.
        self.TempLadderInitializer()
        
        
        #Step 2. Set the Replicas in the ReplicaList 
        for i in range(self.NumReplicas):

            self.ReplicaList[i] = self.ReplicaClass(*args)

            self.ReplicaList[i].name = "CHAIN - {}".format(i)

            self.ReplicaList[i].Temperature = self.Temperatures[i]

            #DEPRECATED  #self.ReplicaList[i].QueueSamples = self.SamplesQueueList[i]               Now uses Pipes...

            self.ReplicaList[i].NumSamples = self.NumReplicaSamples
            self.ReplicaList[i].ChildConn = self.PipeList[i][1] #Sending in the child connection   
        


        self.isInitReplicaCalled = True
    

        return True




    def CopyandSetReplicas(self):

        """
        This function copies All the Replicas along with Model and other parameters to make new Processes AND ASSIGNS them inplace of previous ones.

        This is called after every Replica (i.e. Process) being run for total of Swap Interval amount of iterations
        """

        NewReplicas = [None for _ in range(self.NumReplicas)]

        NewPipes = [mp.Pipe() for _ in range(self.NumReplicas)]


        for i in range(self.NumReplicas):

            NewReplicas[i] = self.ReplicaClass(*self.args)

            NewReplicas[i].name = self.ReplicaList[i].name
            NewReplicas[i].Model = self.ReplicaList[i].Model
            NewReplicas[i].Temperature = self.ReplicaList[i].Temperature
            NewReplicas[i].NumSamples = self.NumReplicaSamples
            NewReplicas[i].ChildConn = NewPipes[i][1]

            #Transferring Meta Parameters
            NewReplicas[i].CurrentPriorProb = self.ReplicaList[i].CurrentPriorProb
            NewReplicas[i].CurrentLikelihoodProb = self.ReplicaList[i].CurrentLikelihoodProb
            NewReplicas[i].MiscParamList = self.ReplicaList[i].MiscParamList

            #Other Parameters for Logging purposes
            NewReplicas[i].Swaps = self.ReplicaList[i].Swaps

        #Assigning New Replicas And Connections
        self.ReplicaList = NewReplicas
        self.PipeList = NewPipes

        print("New Replicas ready to run....")

        return True





    def __NPList_TensorList(self, npList):
        """
        Converts a list of Numpy arrays to a list of tensors
        """

        result = []
        for array in npList:
            result.append(torch.tensor(array, dtype = torch.float64))

        return result


    def Run(self):

        """
        Runs the Replicas PARALLELY and collects the samples in NumReplicas amount of Lists.

        This function only completes one segment of parallel run, i.e. all Replicas run for SwapInterval amount of iters
        and then return the corresponding samples

        NOTE: This function assumes that the Replicas have been Initialized with corresponding Temperatures have been assigned.

        In short, this function runs all the replicas and then sets all relevant Model Parameters to the current one (i.e. those which are achieved after the 'run').
        """

        #Step 1. Starting the Replicas

        for replica in self.ReplicaList:
            replica.start()

        print("All Processes have been started!")
        #time.sleep(0) #Just to make sure other processes always stay ahead of main process

        #Step 2. Collecting samples out of the Queue, because if NumReplicaSamples is too high, the OS Pipes will face a deadlock, and the processes WILL HANG.
        print("Collecting Samples...")

        for i,replica in enumerate(self.ReplicaList):

            self.LastRunSamplesAllReplicas[i], replica.CurrentLikelihoodProb, replica.CurrentPriorProb = self.PipeList[i][0].recv()
            
                                # for i in range(self.NumReplicaSamples):
                                #     for j in range(self.NumReplicas):
                                #         self.LastRunSamplesAllReplicas[j].append(self.SamplesQueueList[j].get())

        # Closing All the child Pipes!
        for i in range(self.NumReplicas):

            self.PipeList[i][1].close()


        #Step 3. Joining all the child processes.
        for replica in self.ReplicaList:
            replica.join()


        #SINCE THE MODEL'S WEIGHTS HASN'T BEEN CHANGED AS THE PROCESSES WORK ON A COPY OF THE CLASS INSTANCES, WE NEED TO LOAD THE FOLLOWING INTO THE ORIGINAL CLASS INSTANCE:
        #   1. Model's Weight 
        #   2. Misc Param List
        #   3. Current Likelihood Prob
        #   4. Current Prior Prob
        #   5. Optimizer's State (for the time when we'll introduce the Optimizer training.)

        for i,replica in enumerate(self.ReplicaList):  
                                                      #                                                                                                   ^ Setting last sample from the previous run as the model weight
            replica.Model.load_state_dict(dict(zip( list(replica.Model.state_dict().keys()), self.__NPList_TensorList( self.LastRunSamplesAllReplicas[i][-1][0] ))))
            replica.MiscParamList = self.LastRunSamplesAllReplicas[i][-1][1]
            
            #replica.CurrentLikelihoodProb, replica.CurrentPriorProb =  ProbList[i]



        return True


    def SwapExecutor(self, ):

        """
        Applies Replica Swap Mechanism after collecting SwapInterval (i.e. NumReplicaSamples) samples

        Things this function will sequentially do:
            1.Gather the state_dict of model and MiscParamList of each replica
            2.Using those state dict and MiscParamList, calculate the prior and the likelihood for each replica.
            3.Loop over each replica and check if it can swap with it's next neighbour.
            4.Swap the temperature of the replicas, if 3. satisfies.
        """

        likelihoods = [None for _ in range(self.NumReplicas)]
        priors = [None for _ in range(self.NumReplicas)]

        for i in range(self.NumReplicas):

            MiscParamList = self.ReplicaList[i].MiscParamList
            ModelParamList = list(self.ReplicaList[i]._ParamClonetoDict().values())

            likelihoods[i] = self.ReplicaList[i].Likelihood(MiscParamList, ModelParamList)[0]
            priors[i] = self.ReplicaList[i].PriorLikelihood(MiscParamList, ModelParamList)[0]

        
        #Looping over Each Replica to check if swap compatible

        #Place to store Swap history:
        history = []

        for i in range(self.NumReplicas - 1):

            DeltaProposal = (1/self.ReplicaList[i+1].Temperature) * (likelihoods[i+1] * self.ReplicaList[i+1].Temperature + priors[i+1])
            DeltaCurent = (1/self.ReplicaList[i].Temperature) * (likelihoods[i] * self.ReplicaList[i].Temperature + priors[i])

            alpha_candidate = np.exp(DeltaProposal - DeltaCurent)

            alpha = min(1, alpha_candidate)

            #Draw u \sim Unif(0,1)

            u = np.random.uniform(0,1)

            if u<alpha:

                print("Swap Accepted between {} and {} !!".format(self.ReplicaList[i].name, self.ReplicaList[i+1].name))

                #Swap the Replicas, i.e. swapping the Temperatures
                self.ReplicaList[i].Temperature, self.ReplicaList[i+1].Temperature = self.ReplicaList[i+1].Temperature, self.ReplicaList[i].Temperature 

                self.ReplicaList[i].Swaps += 1
                self.ReplicaList[i+1].Swaps += 1

                history.append([i,i+1])


        self.SwapHistory.append(history)



    def RunChains(self, *args):

        """
        Runs all the chains to collect MaxSamples samples from each replica.

        Each Replica collects self.NumReplicaSamples (SwapInterval) amout of samples in each run, therefore the Swap checks will be done int(MaxSamples/NumReplicaSamples) times
        """

        t1 = time.time()

        if self.isInitReplicaCalled == False:

            if (len(args)==0):

                raise ValueError("It seems you have not Initialized the Replicas by calling InitReplicas. You can call InitReplicas, OR, pass the arguments to this function itself to initialize it within this function.")

            self.InitReplicas(*args)
            print("Replicas ready to run...")



        NumSwapChecks = int(self.MaxSamples / self.NumReplicaSamples) #Check for Swaps this many times.

        for i in range(NumSwapChecks - 1): # -1 because you don't need to check for swaps when you have already sampled maximum amount of times.

            t2 = time.time()
            #Run the replicas to collect NumReplicaSamples amount of samples from each replica
            self.Run()

            #Collect those replicas into self.AllSamples
            for j in range(self.NumReplicas):
                self.AllSamples[j].extend(self.LastRunSamplesAllReplicas[j])

            #Checking for Swap Conditions on all Replicas
            self.SwapExecutor()

            #Creating New Process while preserving the Replicas' Information to run again.
            self.CopyandSetReplicas()

            t3 = time.time()
            print('\n')
            print("------------------------------ Run Number {} took {} seconds to complete ------------------------------".format(i+1, t3-t2))
            print('\n')
        #Run the replicas to collect NumReplicaSamples amount of samples from each replica, for the last time.
        t2 = time.time()
        self.Run()

        #Collect those replicas into self.AllSamples, for the last time.
        for j in range(self.NumReplicas):
            self.AllSamples[j].extend(self.LastRunSamplesAllReplicas[j])

        t3 = time.time()
        print("------------------------------ Run Number {} took {} seconds to complete ------------------------------".format(NumSwapChecks, t3-t2))

        t4 = time.time()
        print('\n\n')


        print("######################  All Runs Completed in {} seconds, saving samples now as 'Samples.npy' ###################### ".format(t4-t1))
        np.save('Samples.npy',np.array(self.AllSamples), allow_pickle = True)

        print("ALL DONE!")

        return True
