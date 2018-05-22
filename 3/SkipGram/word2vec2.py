import zipfile
import collections
import numpy as np

import math
import random


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time
import pickle

from inputdata import Options, scorefunction
from model import skipgram



class word2vec:
	def __init__(self, inputfile, vocabulary_size=100000, embedding_dim=300, epoch_num=50, batch_size=16, windows_size=2,neg_sample_num=10):
		self.op = Options(inputfile, vocabulary_size)
		self.embedding_dim = embedding_dim
		self.windows_size = windows_size
		self.vocabulary_size = vocabulary_size
		self.batch_size = batch_size
		self.epoch_num = epoch_num
		self.neg_sample_num = neg_sample_num


	def train(self):
		model = skipgram(self.vocabulary_size, self.embedding_dim)
		
		if torch.cuda.is_available():
			model.cuda()
			
		optimizer = optim.SGD(model.parameters(),lr=0.2)
		
		for epoch in range(self.epoch_num):
			print ('Starting with epoch',epoch)
			start = time.time()     
			self.op.process = True
			batch_num = 0	  
			batch_new = 0

			while self.op.process:
				pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)
	
				pos_u = Variable(torch.LongTensor(pos_u))
				pos_v = Variable(torch.LongTensor(pos_v))
				neg_v = Variable(torch.LongTensor(neg_v))
	
				if torch.cuda.is_available():
					pos_u = pos_u.cuda()
					pos_v = pos_v.cuda()
					neg_v = neg_v.cuda()
	
				optimizer.zero_grad()
				loss = model(pos_u, pos_v, neg_v,self.batch_size)
	
				loss.backward()
				
				optimizer.step()
				
				batch_num = batch_num + 1 
				
			self.save_embedding(model,self.op.i2w,self.op.w2i)
			print ('Epoch',epoch,'took',time.time()-start,'seconds')
		print("Optimization Finished!")
		
	def save_embedding(self, model,idx2word=None, word2idx=None):
		word2embedW1 = {}
		word2embedW2 = {}
        
		for key, value in word2idx.items():
			idx = word2idx[key]
			word2embedW1[key] = model.u_embeddings.weight.data[idx].numpy()
			word2embedW2[key] = model.v_embeddings.weight.data[idx].numpy()
        
		
		with open('SG/idx2word.pickle','wb') as f:
			pickle.dump(dict(idx2word),f)
        
		time.sleep(1)
        
		with open('SG/word2idx.pickle','wb') as f:
			pickle.dump(dict(word2idx),f)
		
		
		time.sleep(1)
        
		with open('SG/word2embed_W1.pickle','wb') as f:
			pickle.dump(word2embedW1,f)
        
		time.sleep(1)
        
		with open('SG/word2embed_W2.pickle','wb') as f:
			pickle.dump(word2embedW2,f)
  
if __name__ == '__main__':
	wc= word2vec('training.en',epoch_num=10)
	wc.train()