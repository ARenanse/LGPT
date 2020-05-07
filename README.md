[![Build Status](https://travis-ci.org/WhenDustSettles/LGPT.svg?branch=master)](https://travis-ci.org/WhenDustSettles/LGPT)         ![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/WhenDustSettles/LGPT?color=j&include_prereleases)



# Langevin Gradient Parallel Tempering for Bayesian Neural Learning
This repository is a very tangible and general purpose implementation of the Langevin Gradient Parallel Tempering Algorithm as explained in Chandra et al in < arXiv:1811.04343v1 > using mutliprocessing.

Using this, one can sample from the Posterior Distribution in weights of any\* PyTorch based model by inheriting the *PTReplicaMetaBase* class from *ReplicaABC.py* and defining the other model parameters, like the Prior Distribution over weights, the Likelihood and other problem dependent parameters (look over to the following sections on how to declare them correctly).

# 1. Requirements

Built on:
1. Numpy v1.17.2
2. Torch v1.15.0

# 2. A Brief Explaination of the Parallel Tempering Algorithm
Parallel Tempering has been long used in Monte Carlo simulations, dating back from Hansmann's use of it in 1997 in simulations of a biomolecule to Sugita's formulation of a molecular dynamics version of parallel tempering in 1999.

In Parallel Tempering, one simulates some (say, *M*) number of replicas of the system in interest where each replica is in a canonical ensemble, albeit in different temperatures.
The idea behind keeping multiple samplers at different temperatures, instead of just one sampler, is due to the fact that when a single sampler with low temperature (say, 1) is presented to sample from a multi-modal distributions where distance between the modes is somewhat large (this means there is a *big valley* where probability density is very low between them), then the single sampler usually gets *stuck* in the low probability regions between the modes.

To tackle that, Parallel Tempering uses multiple replicas at different temperatures, where high temperature effectively *flattens the modes* so as to reduce the chances of getting *stuck* in a local minima. The replicas at lower temperature doesn't have this effect and thus can effectively sample from the original distribution.
The main idea in Parallel Tempering is that of swapping the configuration of these replicas from high temperature to lower temperature depending on the likelihood that the system sampling at the higher temperature happens to be in a region of phase space that is important to the replicas at lower temperature.

# 3. How to use this Package

For a given well defined Bayesian problem in Neural setting, one will have a function approximator, in this case, a Neural Network defined as a PyTorch model,
the Prior Distribution over the parameters (weights & biases), and the Likelihood of the data given the parameters.

Using this package, you can define any type of Prior, Likelihood and a PyTorch Model and then collect samples from it's Posterior Distribution over parameters.

To do this, first, you will need to create a class which inherits the **PTReplicaMetaBase** class from **ReplicaABC.py**.
After inheriting it, you can define your Model, Training/Testing set and anything else as you wish. However, there are few functions and Class Variables with a certain name that are **NECESSARY** to be defined by your class, and it is these Functions and Variables that defines the main problem that you're trying to tackle.


## Class Members you need to have 

#### *NOTE: All the below 3 parameters should be Initialized in an Abstract method called InitializeMetaParameters, which is explained below*

### Class Variable 1 : 
    **self.MiscParamList**
    
This is a list of those parameters that are used in Prior or Likelihood distributions, other than Model results itself.
    
For example, someone might define the Prior distribution over weights as the following : 
    
![p(\theta) \propto \frac{1}{{(2 \pi \tau^2)}^{n/2}} \times exp \left( -\frac{1}{2\sigma^2} \sum_{\forall w \in \mathbb{W} } w^2 \right) \times \tau^{2(1 + \nu_1)} exp(-\frac{-\nu_2}{\tau^2})](https://render.githubusercontent.com/render/math?math=p(%5Ctheta)%20%5Cpropto%20%5Cfrac%7B1%7D%7B%7B(2%20%5Cpi%20%5Ctau%5E2)%7D%5E%7Bn%2F2%7D%7D%20%5Ctimes%20exp%20%5Cleft(%20-%5Cfrac%7B1%7D%7B2%5Csigma%5E2%7D%20%5Csum_%7B%5Cforall%20w%20%5Cin%20%5Cmathbb%7BW%7D%20%7D%20w%5E2%20%5Cright)%20%5Ctimes%20%5Ctau%5E%7B2(1%20%2B%20%5Cnu_1)%7D%20exp(-%5Cfrac%7B-%5Cnu_2%7D%7B%5Ctau%5E2%7D))


The above equation describes the Prior distribution over weights for a particular problem (see arXiv:1811.04343v1, Chandra et al).
    
In this equation, the Miscellaneous Parameters are: ![\tau, \sigma, \nu_1, \nu_2](https://render.githubusercontent.com/render/math?math=%5Ctau%2C%20%5Csigma%2C%20%5Cnu_1%2C%20%5Cnu_2).
    
Therefore, the self.MiscParamList will be a length 4 list of these parameters, or it can increase if Likelihood introduces more parameters.
    
### Class Variable 2 :
    **self.CurrentPriorProb**
    
This is a placeholder for the current Prior Probability, used in the core LGPT algorithm for determining the acceptance probability.

### Class Variable 3 : 
    **self.CurrentLikelihoodProb**
    
This is a placeholder for the current Likelihood Probability, used in the core LGPT algorithm for determining the acceptance probability.

    

### Function 1 : 
    *InitializeMiscParameters(self)*

Define this function so as to initialize the three Meta Variables defined above: *self.MiscParamList*, *self.CurrentPriorProb*, *self.CurrentLikelihoodProb*
And then Call this at the bottom of your Custom Model Class's (which inherited PTReplicaMetaBase) *__init__*.

You can be as flexible as possible in initializing the Meta Parameters, using even the Model Predictions (though with torch.no_grad() namespace) to guide the initial values of them.

Please Note that how you initialize the Parameters partially governs the Chains performance. So in case if you see undesired samples or chains doesn't explore enough, you can try to change the initialization scheme for the Meta Parameters. 

### Function 2 :
    *ProposeMiscParameters(self)*

Takes No arguments, except self.

This function returns a list of exact same orientation and length as of the self.MiscParamList, albeit the values are the proposals for new values of each the Miscellaneous Parameters in the self.MiscParamList.
This is used in LGPT Algorithm for gathering new Proposed values of the Miscellaneous Parameters which, in-turn, will be used for proposing new Likelihood and Prior Probabilities.


Returns a List where each element of the list corresponds to the new proposed value of the respective Miscellaneous Parameter, and make sure they are of type numpy.ndarray or Python native types, not torch .tensor or other types as they are not communicable between processes.

### Function 3 :
	*ReturnLoss(self)*

Takes No arguments.

This function is supposed to calculate the loss on the Training set that you have passed with the Model and the Loss function of your choice and setting.

It can be as customized as possible, at the end of it, it should just return the loss of (strictly) type torch.tensor, computed with the help of Model you have defined in your class and training set.


### Function 4 :
	*PriorLikelihood(self, ProposedMiscParamList, ModelParametersProposed)*

Takes two arguments:

1. ProposedMiscParamList : A list of newly proposed Miscellaneous Parameters.

2. ModelParametersProposed : A list of Tensors of Model's Parameters, that is proposed in a given iteration of the LGPT algorithm.

Note that above two arguments are already presented to the function, your task is to implement this function using these two (sufficient) arguments to return the Log Prior Probability over the Model's Parameters (that is, Weights and Biases).

Also, make sure to return a second variable(list) of info regarding the Prior Probability, otherwise make this an Empty list and make it return alongside the Prior Probability.

This function can be implemented as flexibly as possible and is supplemented with ModelParametersProposed, which is a list of Model's Parameters that has been proposed to be accepted in the given iteration of LGPT algorithm (by one iteration, I mean the innermost loop of the algorithm). This list of Proposed Model parameters is only there for use in the case when your Problem's Prior actually depends on the Model's Parameters. However, it's not compulsory to use them, it just depends on your problem setting.

Finally, the function should return two arguments, first one is the Prior probability itself (strictly of type torch.tensor), the other is a list of inormation regarding the computation, if not needed, just pass a [None] or [].


### Function 5 :
	*Likelihood(self, ProposedMiscParamList, ModelParametersProposed)*

Takes two arguments:

1. ProposedMiscParamList : A list of newly proposed Miscellaneous Parameters.

2. ModelParametersProposed : A list of Tensors of Model's Parameters, that is proposed in a given iteration of the LGPT algorithm.

Note that above two arguments are already presented to the function, your task is to implement this function using these two (sufficient) arguments to return the Log Likelihood Probability of the data given the Model's Parameters (that is, Weights and Biases).

Also, make sure to return a second variable(list) of info regarding the Likelihood Probability WHERE FIRST ELEMENT OF THAT LIST SHOULD BE THE LOSS INCURRED BY THE MODEL ON TRAINING/OTHER SET. If your problem does not require calculation of Loss or Model's predictions at that iteration (which, I think is highly unlikely), then just return an Singular None List [None] or an Empty List [].

Again, this function can be implemented as flexibly as possible and is supplemented with ModelParametersProposed, which is a list of Model's Parameters that has been proposed to be accepted in the given iteration of LGPT algorithm (by one iteration, I mean the innermost loop of the algorithm). This list of Proposed Model parameters is only there for use in the case when your Problem's Likelihood distribution actually depends on the Model's Parameters or the precition of the model with parameters set as the proposed parameters. However, it's not compulsory to use them, it just depends on your problem setting.

Finally, the function should return two arguments, first one is the Likelihood probability itself (strictly of type torch.tensor), the other is a list of inormation regarding the computation as discussed above.


## 4. How to use this class

In your class, now you must have *NumSamples, GlobalFraction, Temperature, UseLG, LGProb, TrainData, TestData, lr, RWStepSize, ChildConn* as arguments on the *__init__* apart from other stuff that is specefic fro your need. 

Note that you might have some conflicting doubts what ChildConn actually is. For now, just pass anything to it, it's irrelevant on high level abstractions and the code itself ignores anything that you pass to it. But it's presence is there just in the off-chance that you want to create more complex versions of this replica.

After defining your class with all the above Methods and Members that we discussed, you might want to sample from the Posterior distribution of your Model's parameters. To do that, just do the following.

import *ParallelTempering* class from *LGPT.py*.

Initialize a member of this class by passing the following arguments:

      1. ReplicaClass : The class of the Model that you just made above, which inherits the PTReplicaMetaBase.

      2. NumReplicas : How many Replicas/Samplers you want for your task.

      3. MaxSamples : The Maximum amount of samples that you want to colllect (Note that this may wary a lot, as you might need to sample a considerable amount to overcome Burn-In)

      4. SwapInterval : if <1, then the fraction of MaxSamples after which all Replicas will be checked for the Swap Criterion for the Replicas. Else if >1, then the actual number of Samples after which Swap Criterion will be checked.

      5. MaxTemp : Maximum Temperature of the system.

      6. BetaLadderMethod : This is the mode switcher for the Beta (inverse of Temperature) Ladder Initialization method. For example, if your MaxTemp is 5000 and BetaLadderMethod is 'GEO', then each Replica will be assigned temperature on the basis of Geometric Progression of Beta from 1/MaxTemp to 1.
                            Currently, three Methods are implemented: 'GEO' for Geometric, 'LIN' for Linear, 'HAR' for Harmonic spacing. 'GEO' is the default.


After making an instance of ParallelTempering, you need to call it's Method called *InitReplicas(\*args)* which initializes all the Chains based on your Model that you just made above. The arguments that you need to pass it are important. You have to pass it the exact same type of arguments that you would pass to **YOUR MODEL** that you've just made by inheriting PTReplicaMetaBase. Make sure to pass it the **exactly same arguments** otherwise it may fail.

But Note that the Replicas' Temperature, NumSamples, ChildConn that you will pass to InitReplicas will be ignored and will be rather assigned correclty, internally. Note I chose to keep those parameters available while writing the base class because there are lot of freedom of further development in this regime, like Hamiltonian Parallel Tempering and so on, and if anyone is interested, they would require these parameters to be easily achievable, that (hopefully) justifies their *openness* to the user.

After this, you can just call *RunChains()* and it will run the chains with important info being verbosed.
The samples collected from all the chains will be available as a Numpy file with name *Samples.npy*.


## 5. Example Usage

We have Implemented a basic Model as explained in arXiv:1811.04343v1 in the file *PTReplica.py*.
That Model is trained by the script in *PT_Test.py*, take a look there on how to Run Replicas.

The results of training this basic model on a basic ![\frac{sin(x)} {x}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bsin(x)%7D%20%7Bx%7D) Regression task is shown in the Notebook named *Comparison.ipynb*.
It also carries a comparison with a basic frequentist model trained by backpropagation.



**Finally, I hope the Internal Documentation of the code is clear enough for those who are interested in the implementation itself.**





    
    
