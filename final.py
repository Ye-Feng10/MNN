#Final
import pandas as pd
import numpy as np
from scipy.stats import zscore 
import time
import matplotlib.pyplot as plt


#import data
df = pd.read_csv('adultdata.csv', header = None, prefix='X')
df_test = pd.read_csv('adulttest.csv', header = None, prefix='X')

#data clean
df = df[np.abs(df.X0 - df.X0.mean() <= (3*df.X0.std()) )]
df = df[np.abs(df.X2 - df.X0.mean() <= (3*df.X2.std()) )]
df = df[np.abs(df.X4 - df.X0.mean() <= (3*df.X4.std()) )]
df = df[np.abs(df.X10 - df.X0.mean() <= (3*df.X10.std()) )]
df = df[np.abs(df.X11 - df.X0.mean() <= (3*df.X11.std()) )]
df = df[np.abs(df.X12 - df.X0.mean() <= (3*df.X12.std()) )]



data_use = df.values
data_test = df_test.values
ls_num = [14,2]
Epsi = 0.0001


#code test
x_test = np.array([0.5,0.9])
ls_layerW_test = [np.array([[3,-2,4],[-4,1,-1],[0,0,0]]),np.array([[-3],[5],[0],[0]]),np.array([[0.5,-1],[0,0]])]
y_test = np.array([0,1])
Epsi_test = 0.1

data_x_0_13 = np.random.random([30000,14])
data_x = np.random.randint(0,2,[30000,15])
data_x = data_x.astype(float)
data_x[:,:-1] = data_x_0_13
Epsi_x = 0.01


#Neural Network
def Sigmoid(z):
	ans = 1/(1+np.exp(-z))
	return(ans)

def feed_step_forward(a,b):
	return(Sigmoid(np.matmul(np.append(a,[1]),b)))
	
#generate W: ls_num includes number of attributes and each layers' nodes number
def gen_layerW(ls_num):
	ls_matrix = list(zip([x + 1 for x in ls_num[:-1]],ls_num[1:]))
	ls_layerW = [np.random.uniform(-2.45,2.45,x) for x in ls_matrix] #sample a Uniform(-r, r) with r=4root(6/(fan-in+fan-out)) #Glorot and Bengio (2010) 
	return(ls_layerW)
	
def feedforward(x,ls_layerW):
	ls_out = []
	ls_out.append(x)
	out1 = feed_step_forward(x, ls_layerW[0])
	ls_out.append(out1)
	former_out = out1
	for w in ls_layerW[1:]:
		out = feed_step_forward(former_out,w)
		ls_out.append(out)
		former_out = out	
	return(ls_out)

def backprop(out_matrix,y,ls_layerW,Epsi):
	out_top = out_matrix[-1]
	delta_top = np.multiply(np.multiply((y - out_top),(1 - out_top)),out_top) ###(y-r or r-y
	delta = delta_top
	i = 1
	ls_W_update = []
	while i < len(out_matrix):
		out = out_matrix[-i-1]
		ls_W_update.append(Epsi*np.outer(np.append(out,1),delta))
		ThisLayerW = ls_layerW[-i]
		try:
			Error_correction = np.matmul(ThisLayerW[:-1,:].T,delta)
		except:
			Error_correction = np.dot(ThisLayerW[:-1,:],delta)
		delta = np.multiply(np.multiply(Error_correction,(1-out)),out)
		i += 1
	return(ls_W_update)

def UpdateW(ls_layerW, ls_W_update):
	ls_W_update = ls_W_update[::-1]
	ls_layerW_new = [np.add(x[0],x[1]) for x in list(zip(ls_layerW,ls_W_update))]
	return(ls_layerW_new)

def generateY(datapts):
	n = len(datapts)
	outTypes = np.unique(datapts[:,-1])
	outTypesNum = len(outTypes)
	Y_pool = np.identity(outTypesNum)
	Y = np.zeros(shape = [n,outTypesNum])
	i = 0
	while i < n:
		typeLoc = np.where( outTypes == datapts[:,-1][i] )[0][0]
		Y[i] = Y_pool[typeLoc]
		i += 1
	return(Y)

def Evaluate(ls_top_out, Y):
	length = len(Y)
	correct_num = 0
	i = 0
	while i < length:
		if np.where(ls_top_out[i] == ls_top_out[i].max())[0][0] == np.where(Y[i] == Y[i].max())[0][0]:
			correct_num += 1
		i += 1
	accuracy = correct_num/length
	TotalError = Y - ls_top_out
	Costfunction = sum(sum(TotalError**2))
	return(accuracy,Costfunction)
	
	
def NNmainFunction(data, ls_num_or_W, Epsi):
	ls_result = []
	Y_out = generateY(data)
	try:
		W = gen_layerW(ls_num_or_W)
	except:
		W = ls_num_or_W
	k_times = 0
	Cost_former = 99999999
	data_use = data[:,:-1]
	while True:
		top_out = []
		i = 0
		W_new = W
		while i < len(Y_out):
			x = data_use[i]
			Y = Y_out[i]
			out_matrix = feedforward(x,W)
			top_out.append(out_matrix[-1])
			W_update = backprop(out_matrix,Y,W,Epsi)
			W_new = UpdateW(W_new,W_update)
			i += 1
		error_test = [np.all(x[0] == x[-1]) for x in list(zip(W,W_new))]
		W = W_new

		
		k_times += 1
		accuracy,cost = Evaluate(top_out,Y_out)
		ls_result.append([accuracy,cost,k_times,W])
		print(accuracy,cost,k_times)
		if cost == Cost_former:
			break
		else:	
			Cost_former = cost
		if k_times > 1000:
			break
	return(ls_result)
	
def NNmodelTest(data_test,W_trained):
	out_test = []
	Y_test = generateY(data_test)
	length_test = len(data_test)
	data_test = data_test[:,:-1]
	for x in data_test:
		#print(x)
		top_out_test = feedforward(x,W_trained)
		out_test.append(top_out_test[-1])
	
	accuracy_test,cost_test = Evaluate(out_test,Y_test)
	results = [accuracy_test,cost_test,length_test]
	return(results)


	
def testfun():
	out_mat_test = feedforward(x_test,ls_layerW_test)
	ls_num = UpdateW(ls_layerW_test,backprop(out_mat_test,y_test,ls_layerW_test,Epsi_test))
	while True:
		out_mat_test = feedforward(x_test,ls_num)
		print(out_mat_test[-1])
		ls_num = UpdateW(ls_num,backprop(out_mat_test,y_test,ls_num,Epsi_test))
		print(y_test)
		time.sleep(1)
	

	

	
if __name__ == "__main__":
	NN_results = NNmainFunction(data_use, ls_num, Epsi)
	W_trained = NN_results[-1][-1]
	NN_test_results = NNmodelTest(data_test,W_trained)
	x = [x[2] for x in NN_results]
	y = [x[1] for x in NN_results]
	z = [x[0] for x in NN_results]
	w = [x[3] for x in NN_results]
	e = [NNmodelTest(data_test,w) for w in w ]
	e1 = [x[1] for x in e]
	e2 = [x[0] for x in e]

	
	
	