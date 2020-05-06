from ReplicaABC import PTReplicaMetaBase
import LGPT as pt

# For importing custom model.
from PTReplica import BasicModel
import numpy as np
import torch
import time


#DNM:- Does Not Matter.

if __name__ == "__main__":

	x = np.random.uniform(-50,50,size = [1000,1])
	y = np.sin(x) /x
	train = torch.tensor(np.concatenate( (x,y) ,axis = 1), dtype = torch.float)

	x_test = np.random.uniform(-50,50,size = [100,1])
	y_test = np.sin(x_test) / x_test
	test = torch.tensor(np.concatenate((x_test,y_test),axis = 1), dtype = torch.float)



	Model = pt.ParallelTempering(BasicModel, 15, 100000, 200, 5000, 'GEO')

	time.sleep(2)

	# train = torch.tensor([[1,2,3,10],[4,5,6,10],[7,8,9,10],[11,12,13,10],[14,15,16,10]], dtype = torch.float)
	# test = torch.tensor([[17,18,19,10],[20,21,22,10],[23,24,25,10]], dtype = torch.float)

	#                         DNM     DNM                                    
	Model.InitReplicas(1,5,1, 200,0.6,700,True,0.5,train,test,0.001,0.05, "ANYTHING")

	np.save('test_set.npy', np.array(test), allow_pickle = True)

	Model.RunChains(SamplesFileName = 'Samples2.npy')
