# model twi-esn 
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as slinalg

from sklearn.linear_model import Ridge

import gc

from utils.utils import calculate_metrics
from utils.utils import create_directory
from sklearn.model_selection import train_test_split

import time

class Classifier_TWIESN:

	def __init__(self, output_directory,verbose): 
		self.output_directory = output_directory
		self.verbose = verbose

		first_config = {'N_x':250,'connect':0.5,'scaleW_in':1.0,'lamda':0.0}
		second_config = {'N_x':250,'connect':0.5,'scaleW_in':2.0,'lamda':0.05}
		third_config = {'N_x':500,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		fourth_config = {'N_x':800,'connect':0.1,'scaleW_in':2.0,'lamda':0.05}
		self.configs = [first_config,second_config,third_config,fourth_config]
		self.rho_s = [0.55,0.9,2.0,5.0]
		self.alpha = 0.1 # leaky rate


	def init_matrices(self):
		self.W_in = (2.0*np.random.rand(self.N_x,self.num_dim)-1.0)/(2.0*self.scaleW_in)

		converged = False

		i =0 

		while(not converged):
			i+=1

			self.W = sparse.rand(self.N_x,self.N_x,density=self.connect).todense()

			self.W[np.where(self.W>0)] -= 0.5

			try:
				eig, _ = slinalg.eigs(self.W,k=1,which='LM')
				converged = True
			except: 
				print('not converged ',i)
				continue

		self.W /= np.abs(eig)/self.rho

	def compute_state_matrix(self, x_in):
		n = x_in.shape[0]
		X_t = np.zeros((n, self.T, self.N_x),dtype=np.float64) 
		X_t_1 = np.zeros((n, self.N_x),dtype=np.float64) 
		for t in range(self.T): 
			curr_in = x_in[:,t,:]
			curr_state = np.tanh(self.W_in.dot(curr_in.T)+self.W.dot(X_t_1.T)).T
			curr_state = (1-self.alpha)*X_t_1 + self.alpha*curr_state
			X_t_1 = curr_state
			X_t[:,t,:] = curr_state
				
		return X_t

	def reshape_prediction(self,y_pred, num_instances,length_series): 
		new_y_pred = y_pred.reshape(num_instances,length_series,y_pred.shape[-1])
		new_y_pred = np.average(new_y_pred, axis=1)
		new_y_pred = np.argmax(new_y_pred,axis=1)
		return new_y_pred

	def train(self):
		start_time = time.time()
        
		self.init_matrices()

		state_matrix = self.compute_state_matrix(self.x_train)
		new_x_train = np.concatenate((self.x_train,state_matrix), axis=2).reshape(
			self.N * self.T , self.num_dim+self.N_x)

		state_matrix = None 
		gc.collect()

		new_labels = np.repeat(self.y_train,self.T,axis=0)
		ridge_classifier = Ridge(alpha=self.lamda)
		ridge_classifier.fit(new_x_train,new_labels)


		state_matrix = self.compute_state_matrix(self.x_val)
		new_x_val = np.concatenate((self.x_val,state_matrix), axis=2).reshape(
			self.x_val.shape[0] * self.T , self.num_dim+self.N_x)

		y_pred_val = ridge_classifier.predict(new_x_val)
		y_pred_val = self.reshape_prediction(y_pred_val,self.x_val.shape[0],self.T)

		df_val_metrics = calculate_metrics(np.argmax(self.y_val, axis=1),y_pred_val,0.0)

		train_acc = df_val_metrics['accuracy'][0]
        
		state_matrix = self.compute_state_matrix(self.x_test)

		new_x_test = np.concatenate((self.x_test,state_matrix), axis=2).reshape(self.x_test.shape[0] * self.T , self.num_dim+self.N_x)
		state_matrix = None 
		gc.collect()

		y_pred = ridge_classifier.predict(new_x_test)

		y_pred = self.reshape_prediction(y_pred,self.x_test.shape[0],self.T)

		duration = time.time() - start_time

		df_metrics = calculate_metrics(self.y_true,y_pred,duration)


		self.W_out = ridge_classifier.coef_
		ridge_classifier = None 
		gc.collect()


		df_metrics.to_csv(self.output_directory+'df_metrics.csv', index=False)

		return df_metrics , train_acc


	def fit(self,x_train,y_train,x_test,y_test, y_true):
		best_train_acc = -1

		self.num_dim = x_train.shape[2]
		self.T = x_train.shape[1]
		self.x_test = x_test
		self.y_true = y_true
		self.y_test = y_test


		self.x_train, self.x_val, self.y_train,self.y_val = \
			train_test_split(x_train,y_train, test_size=0.2)
		self.N = self.x_train.shape[0]


		if self.x_train.shape[0] > 1000 or self.x_test.shape[0] > 1000 : 
			for config in self.configs: 
				config['N_x'] = 100
			self.configs = [self.configs[0],self.configs[1],self.configs[2]]

		output_directory_root = self.output_directory 

		for idx_config in range(len(self.configs)): 
			for rho in self.rho_s:
				self.rho = rho 
				self.N_x = self.configs[idx_config]['N_x']
				self.connect = self.configs[idx_config]['connect']
				self.scaleW_in = self.configs[idx_config]['scaleW_in']
				self.lamda = self.configs[idx_config]['lamda']
				self.output_directory = output_directory_root+'/hyper_param_search/'+\
					'/config_'+str(idx_config)+'/'+'rho_'+str(rho)+'/'
				create_directory(self.output_directory)
				df_metrics, train_acc = self.train() 

				if best_train_acc < train_acc : 
					best_train_acc = train_acc
					df_metrics.to_csv(output_directory_root+'df_metrics.csv', index=False)
					np.savetxt(output_directory_root+'W_in.txt', self.W_in)
					np.savetxt(output_directory_root+'W.txt', self.W)
					np.savetxt(output_directory_root+'W_out.txt', self.W_out)

				gc.collect()