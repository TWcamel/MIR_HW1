#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Example code for key detection preprocessing. You may start with filling the 
'?'s below. There're also some description and hint within comment. However, 
please feel free to modify anything as you like!

@author: selly
"""
from glob import glob
from collections import defaultdict
# Below are packages and functions that you might need. Uncomment by remove the
# "#" in front of each line.
# from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
# from scipy.stats import pearsonr
# from mir_eval.key import weighted_score
# from sklearn.metrics import accuracy_score
# import numpy as np # np.log10()

#%%
import utils # self-defined utils.py file
DB   = 'GTZAN'
if DB=='GTZAN': # dataset with genre label classify at parent directory
	FILES = glob(DB+'/wav/*/*.wav')
else:
	FILES = glob(DB+'/wav/*.wav')
#print(FILES)

GENRE = [g.replace('\\','/').split('\\')[2] for g in glob(DB+'/wav/*')]
n_fft = 100	# (ms)
hop_length = 25	# (ms)

#%% Q1
if DB=='GTZAN':
	label, pred = defaultdict(list), defaultdict(list)
else:
	label, pred = list(), list()
chromagram = list()
gens       = list()
for f in FILES:
	f = f.replace('\\','/')
	content = utils.read_keyfile(f,'*.lerch.txt')
	if (int(content)<0): continue # skip saving if key not found
	if DB=='GTZAN':
		gen = f.split('/')[2]
		label[gen].append(utils.LABEL[int(content)]) 
		gens.append(gen)
	else:
		label.append(utils.LABEL[content]) 
	
	sr, y = utils.read_wav(f)
	##########
	# TODO: Follow task1 description to give each audio file a key prediction.
	
	# compute the chromagram of audio data `y`
	cxx = ?
	chromagram.append(cxx) # store into list for further use
	# summing up all the chroma features into chroma vector
	chroma_vector = ?
	# finding the maximal value in the chroma vector and considering the note 
	# name corresponding to the maximal value as the tonic pitch
	key_ind = ?
	# finding the correlation coefficient between the summed chroma vectors and 
	# the mode templates
	# Hint: utils.rotate(ar,n) may help you find different key mode template
	mode = ?
	if DB=='GTZAN':
		pred[gen].append(?)
	else: 
		pred.append('?') # you may ignore this when starting with GTZAN dataset
	##########

print("***** Q1 *****")
if DB=='GTZAN':
	label_list, pred_list = list(), list()
	print("Genre    \taccuracy")
	for g in GENRE:
		##########
		# TODO: Calculate the accuracy for each genre
		# Hint: Use label[g] and pred[g]
		acc = ?
		##########
		print("{:9s}\t{:8.2f}%".format(g,acc))
		label_list += label[g]
		pred_list  += pred[g]
else:
	label_list = label
	pred_list  = pred

##########
# TODO: Calculate the accuracy for all file.
# Hint1: Use label_list and pred_list.
acc_all = ?
##########
print("----------")
print("Overall accuracy:\t{:.2f}%".format(acc_all))



# %%
