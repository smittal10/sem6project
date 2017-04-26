from __future__ import division
from pprint import pprint
import random
import numpy as np
import copy
import os.path
batch_size=1
def load_caption_2_id(file_path):
    caption_2_id = {}
    with open(file_path,'r') as f:
       for line in f:
            entry = line.split('\t')
            if len(entry) == 2:
                caption_2_id[(entry[1])] = int(entry[0])
    return caption_2_id

def load_id_2_imageid(file_path):
    id_2_imageid = {}
    with open(file_path,'r') as f:
       for line in f:
            entry = line.split('\t')
            if len(entry) == 2:
                id_2_imageid[int(entry[0])] = int(entry[1])

    return id_2_imageid
def load_token_to_id(token_to_id_file_path):
    token_to_id = {}
    with open(token_to_id_file_path,'r') as f:
       for line in f:
            entry = line.split('\t')
            if len(entry) == 2:
                token_to_id[entry[0]] = int(entry[1])

    return token_to_id


def get_captions(file_path):
    caption_set = {}
    
    with open(file_path,'r') as f:
       for line in f:
            sent_length=len(line.split())
            #print(sent_length)
            if sent_length in caption_set :
                caption_set[sent_length].append(line)
            else:
                caption_set[sent_length]=[line]

    return caption_set
    
class DataReader(object):
	def __init__(self,token_to_id_path,segment_sepparator_token):
	    self.token_to_id_path = token_to_id_path
	    self.token_to_id = load_token_to_id(token_to_id_path)
	    self.vocab_dim = len(self.token_to_id)
	    if not segment_sepparator_token in self.token_to_id:
	    	print ("ERROR: separator token '%s' has no id:" % (segment_sepparator_token))
	    	sys.exit()
	    self.segment_sepparator_id = self.token_to_id[segment_sepparator_token]

	def get_batches(self,train_captions,batch_size):
		train_batches = []
		for sent_length, caption_set in train_captions.items():
			caption_set = list(caption_set)
			random.shuffle(caption_set)
			num_captions = len(caption_set)
			num_batches = num_captions // batch_size
			for i in range(num_batches+1):
				end_idx = min((i+1)*batch_size, num_captions)
				new_batch = caption_set[(i*batch_size):end_idx]
				if len(new_batch) == batch_size:
					train_batches.append((new_batch, sent_length))
		random.shuffle(train_batches)
		return train_batches
	def create_batch(self,train_captions,token_2_id,caption_2_id,id_2_image,img_file):
		train_batches = self.get_batches(train_captions, batch_size)
		print(len(train_batches))
		for batch_item in train_batches:
			(batch,sent_length)=batch_item
			x=[]
			y=[]
			img_features=[]
			token_count=0
			for caption in batch:
				tokens=caption.split();
				tok_id=[]
				img_feat=[]
				id=caption_2_id[caption]
				img_id=id_2_image[id]
				img_id=str(img_id)
				n=copy.copy(img_id)
				n=n.zfill(12)
				img_path=img_file+n+'.txt'
				if not os.path.exists(img_path):
					continue

				img=np.loadtxt(img_path)				
				for i in range(0,sent_length):
					print ("hello")
					if tokens[i] not in token_2_id:
						continue				
					tok_id.append(token_2_id[tokens[i]])
					img_feat.append(img)
				x.append((tok_id[: -1]))
				y.append(tok_id[1:])
				img_features.append(img_feat)
				token_count+=sent_length
			yield x,y,img_features,token_count
	
#id_2_imageid=load_id_2_imageid('Toy/train/idtoimgid.txt')
#print(id_2_imageid[1450])
			



				
			
		
	
	
	
	
	

