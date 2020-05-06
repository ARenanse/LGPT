from ReplicaABC import PTReplicaMetaBase
import numpy as np
import torch

# TESTING ABC PTReplicaMetaBase to Train a basic Model on a basic Regression task.



#NOTE ONE CAN DEFINE TRAINING AND TESTING TENSOR SETS INSIDE THE CLASS ITSELF. 
class BasicModel(PTReplicaMetaBase):
	
	def __init__(self, D_in, H, D_out, NumSamples, GlobalFraction, Temperature, UseLG, LGProb, TrainData, TestData, lr, RWStepSize, ChildConn, LossFunc = torch.nn.MSELoss,):
		
		
		#Custom Parameters for the model
		self.D_in = D_in
		self.H = H
		self.D_out = D_out
		self.Step_eta = 0.2  #For proposing New Misc Parameters
		

		#Defining the Model.
		self.Model = torch.nn.Sequential( torch.nn.Linear(D_in, H), torch.nn.Sigmoid(), torch.nn.Linear(H, D_out) )
		

		#Initialize the Base Class.
		super().__init__(self.Model, NumSamples, GlobalFraction, Temperature, UseLG, LGProb, TrainData, TestData, lr, RWStepSize, ChildConn, LossFunc = torch.nn.MSELoss)
		
		
		"""
		Specifications:
		
		TrainData = Shape will be [BatchSize, D_in + D_out], D_in + D_out because 1 columns is needed for labels
		TestData = Shape will be [BatchSize, D_out]
		
		"""

		#Initializing Miscellaneous Parameters, at the extreme bottom of __init__.
		self.InitializeMiscParameters()
		
	
	def GiveMeTheLoss(self):
		
		"""
		This function is supposed to do the following things:-
				1. Calculate y_pred (shape = [BatchSize, D_out]) on the entire batch of data by calling Train Data on Model.
				2. Calculate the loss using the self.LossFunc with y_true (shape = [BatchSize, D_out]) and y_pred.
				3. Return the loss [torch.tensor]
		"""
		
		y_pred = self.Model(self.TrainData[:,:self.D_in])
		y_true = self.TrainData[:, self.D_in:]
		
		loss = self.LossFunc(y_pred, y_true)
		
		return loss
	
	
	
	def ProposeMiscParameters(self):
		"""
		Propose new parameters from the current values of self.MiscParamList
		"""
		#For and tau:-
		etaProposal = self.MiscParamList[0] + np.random.normal(0,self.Step_eta,1)
		tauProposal = np.exp(etaProposal)
		
		#For sigma2, nu_1, nu_2
		sigma2Proposal = 25
		nu_1Proposal = 0
		nu_2Proposal = 0
		
		#Returning in the original Order
		NewMiscProposals = [etaProposal, tauProposal, sigma2Proposal, nu_1Proposal, nu_2Proposal]
		
		return NewMiscProposals
	
	
	def rmse(self, y_true, y_pred):
		
		return torch.mean( torch.sqrt( torch.mean((y_true - y_pred)**2, axis = 1,dtype = torch.float) ) )
	
	
	def Likelihood(self, MiscProposalList, Theta_proposal):
		
		"""
		Calculate and return the value of log likelihood as per the decided model.
		
		The Likelihood here is same as that in the paper.
		
		MiscProposalList = The list containing the values of newly proposed Misc Parameters
		
		Theta_proposal = The list containing proposed Parameters for the model.(it's a list not a dict!!)
		"""
		
		with torch.no_grad():
			
			#Setting the model weights as Theta_Proposal
			InitParams = self._ParamClonetoDict()
			
			theta_dict = dict(zip(list(self.Model.state_dict().keys()), Theta_proposal))
			self.Model.load_state_dict(theta_dict)

			#Calculating Model Results on the Training Set.
			fx = self.Model(self.TrainData[:,:self.D_in])
			y_true = self.TrainData[:,self.D_in:]
			
			rmseloss = self.rmse(y_true, fx)
			
		#Load the model parameters back
		self.Model.load_state_dict(InitParams)
		
		#Calculating log probability   
		logprob1 = torch.tensor(-1 * (self.D_out/2) * np.log(2 * np.pi * (MiscProposalList[1])))
		logprob2 = torch.tensor(-1 / (2*MiscProposalList[1]))  *   torch.sum((y_true - fx)**2, axis = 1) 
		logprob = logprob1 + logprob2
		
		#Since we assume that the each row in the TrainData is independent, we calculate the product of each probability, that is, sum all individual log prob.
		return torch.sum(logprob), [rmseloss]
	


	def InitializeMiscParameters(self):

		"""
		Initializes the self.CurrentLikelihoodProb, self.CurrentPriorProb and self.MiscParamList
		"""


		#To set Meta Parameters, as done in the paper.
		#Note:- 
		#	self.MiscParamList == [eta, tau_squared, sigma2, nu_1, nu_2]


		with torch.no_grad():

			#For MiscParamList
			train_pred = self.Model(self.TrainData[:,:self.D_in])
			train_truth = self.TrainData[:,self.D_in:]
			eta = np.log( np.mean(np.var( np.array(train_pred - train_truth) )) )
			tau_squared = np.exp(eta)
			sigma_squared = 25
			nu_1 = 0
			nu_2 = 0

			self.MiscParamList = [eta, tau_squared, sigma_squared, nu_1, nu_2]

			#For CurrentPriorProb, Note that we entered the list of current model weights.
			self.CurrentPriorProb, _  =  self.PriorLikelihood(self.MiscParamList, list(self.Model.state_dict().values()) )

			#For CurrentLikelihoodProb
			self.CurrentLikelihoodProb, _ = self.Likelihood(self.MiscParamList, list(self.Model.state_dict().values())  )


	
	
	def SumTheSquareWeights(self, Theta):

		"""
		Sums up each Weight's square.
		
		Theta is a list of weights and biases.
		"""
		with torch.no_grad():
		
			result = 0

			for param in Theta:

				result += torch.sum(torch.square(param))

		#print("Sum of squares of the weights: ", result)
		return result

		
	
	
	
	def PriorLikelihood(self, MiscProposalList, Theta_proposal):
		
		"""
		Calculate and return the value of log Prior likelihood as per the decided model.
		
		The Prior Likelihood here is same as that in the paper.
		
		MiscProposalList = The list containing the values of newly proposed Misc Parameters
		
		Theta_proposal = The list containing proposed Parameters for the model.(it's a list not a dict!!)
		"""
		
		with torch.no_grad():                                     #      ^ * ((self.D_in * self.H + self.H + 2)/2)
			logprob_part1 =  -1 * np.log(2 * np.pi * MiscProposalList[2]) * ((self.D_in * self.H + self.H + 2)/2)  - (1/(2*MiscProposalList[2])) * self.SumTheSquareWeights(Theta_proposal) 
			logprob_part2 = (1 + MiscProposalList[3]) * np.log(MiscProposalList[1]) - (MiscProposalList[4]/MiscProposalList[1])
		
		return logprob_part1 - logprob_part2, [None]
	
