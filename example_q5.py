#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example code for local key detection preprocessing. You may start with filling 
the '?'s below. There're also some description and hint within comment. However, 
please feel free to modify anything as you like!

@author: selly
"""
import numpy as np
from glob import glob

import utils # self-defined utils.py file
DB = 'BPS-FH'

chromagram, label, index = list(), list(), list()
for f in glob(DB+'/wav/*.wav'):
	f = f.replace('\\','/')
	key = utils.parse_key([line.split('\t')[1] for line in utils.read_keyfile(f,'REF_key_*.txt').split('\n')])
	label.extend(key)
	ind = int(f.split('/')[-1].strip('.txt'))
	if ind in utils.DATA_SPLIT[DB]['train']:
		index.extend(['train']*len(key))
	elif ind in utils.DATA_SPLIT[DB]['valid']:
		index.extend(['valid']*len(key))
	elif ind in utils.DATA_SPLIT[DB]['test']:
		index.extend(['test']*len(key))

#	chromagram.append(?)

chromagram, label, index = map(np.array, [chromagram, label, index])
valid_x = chromagram[index=='valid']
test_x  = chromagram[index=='test' ]
train_x = chromagram[index=='train']

valid_y = label[index=='valid']
test_y  = label[index=='test' ]
train_y = label[index=='train']

del chromagram, label # clean to free memory

#%%