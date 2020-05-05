from ReplicaABC import PTReplicaMetaBase
import LGPT as pt

# For importing custom model.
from PTReplica import BasicModel
import torch
import time


#DNM:- Does Not Matter.

if __name__ == "__main__":

	Model = pt.ParallelTempering(BasicModel, 5, 100000, 0.2, 5000, 'GEO')

	time.sleep(2)

	train = torch.tensor([[1,2,3,10],[4,5,6,10],[7,8,9,10],[11,12,13,10],[14,15,16,10]], dtype = torch.float)
	test = torch.tensor([[17,18,19,10],[20,21,22,10],[23,24,25,10]], dtype = torch.float)

	#                         DNM     DNM                                    
	Model.InitReplicas(3,5,1, 200,0.6,700,True,0.5,train,test,0.001,0.025, "ANYTHING")

	Model.RunChains()