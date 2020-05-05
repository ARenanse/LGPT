from ReplicaABC import PTReplicaMetaBase
import LGPT as pt

# For importing custom model.
from PTReplica import BasicModel
import torch
import time

import unittest

#DNM : Does Not Matter

class Tests(unittest.TestCase):


	def Runner(self, NumChains, MaxSamples, TempLadderMethod, SwapInt, MaxTemp, H, GlobalFraction, UseLG, LGProb, lr, RWStepSize):

		Model = pt.ParallelTempering(BasicModel, NumChains, MaxSamples, SwapInt, 5000, TempLadderMethod)

		time.sleep(2)

		train = torch.tensor([[1,2,3,10],[4,5,6,10],[7,8,9,10],[11,12,13,10],[14,15,16,10]], dtype = torch.float)
		test = torch.tensor([[17,18,19,10],[20,21,22,10],[23,24,25,10]], dtype = torch.float)

		#                         DNM                DNM                                        DNM
		Model.InitReplicas(3,H,1, 200,GlobalFraction,700,UseLG,LGProb,train,test,lr,RWStepSize,"DNM")

		x = Model.RunChains()

		return x


	def test_Basic1(self):

		NumChains = 10
		MaxSamples = 100000
		TempLadderMethod = 'GEO'
		MaxTemp = 5000
		SwapInt = 0.2
		H = 5
		GlobalFraction = 0.6
		UseLG = True
		LGProb = 0.5
		lr = 0.001
		RWStepSize = 0.025

		result = self.Runner(NumChains, MaxSamples, TempLadderMethod, SwapInt, MaxTemp, H, GlobalFraction, UseLG, LGProb, lr, RWStepSize)

		self.assertTrue(result)
		#Model, NumSamples, GlobalFraction, Temperature, UseLG, LGProb, TrainData, TestData, lr, RWStepSize, ChildConn


		


	def test_Basic2(self):

		NumChains = 15
		MaxSamples = 100000
		TempLadderMethod = 'LIN'
		MaxTemp = 50000
		SwapInt = 0.74
		H = 25
		GlobalFraction = 0.1
		UseLG = True
		LGProb = 0.8
		lr = 0.00001
		RWStepSize = 0.0025

		result = self.Runner(NumChains, MaxSamples, TempLadderMethod, SwapInt, MaxTemp, H, GlobalFraction, UseLG, LGProb, lr, RWStepSize)

		self.assertTrue(result)

		


	def test_Basic3(self):

		NumChains = 20
		MaxSamples = 120000
		TempLadderMethod = 'HAR'
		MaxTemp = 500000
		SwapInt = 0.8
		H = 100
		GlobalFraction = 0.02
		UseLG = True
		LGProb = 0.9999
		lr = 0.000001
		RWStepSize = 1.25

		result = self.Runner(NumChains, MaxSamples, TempLadderMethod, SwapInt, MaxTemp, H, GlobalFraction, UseLG, LGProb, lr, RWStepSize)

		self.assertTrue(result)



if __name__ == '__main__':

	unittest.main()
