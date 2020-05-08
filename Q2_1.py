#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import utils  # self-defined utils.py file

DB = 'GTZAN'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB + '/wav/*/*.wav')
    # print(FILES)
else:
    FILES = glob(DB + '/wav/*.wav')
    # print(FILES)

GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]
GENRE.remove('classical')
n_fft = 100  # (ms)
hop_length = 25  # (ms)

if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()

idx =0
gamma = 1
while idx<4 and gamma<1001:
    print('gamma',gamma)
    for f in tqdm(FILES):
        f = f.replace('\\', '/')
        # print("file: ", f)
        content = utils.read_keyfile(f, '*.lerch.txt')
        if (int(content) < 0): continue  # skip saving if key not found
        if DB == 'GTZAN':
            gen = f.split('/')[2]
            label[gen].append(utils.LABEL[int(content)])
            gens.append(gen)
        else:
            label.append(utils.LABEL[content])
    
        sr, y = utils.read_wav(f)
    
        cxx = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=y, sr=sr)))
        chromagram.append(cxx)  # store into list for further use
        chroma_vector = np.sum(cxx, axis=1)
        key_ind = np.where(chroma_vector == np.amax(chroma_vector))
        key_ind = int(key_ind[0])
        mode = {"cMajor": [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
                "cMinor": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]}
        mode['cMajor'] = utils.rotate(mode['cMajor'],  key_ind)
        mode['cMinor'] = utils.rotate(mode['cMinor'],  key_ind)
    
        cMajorCoefficient = pearsonr(chroma_vector, mode['cMajor'])
        cMinorCoefficient = pearsonr(chroma_vector, mode['cMinor'])
    
        mode = ''
    
        a = (key_ind+3)%12
        if (cMajorCoefficient[0] > cMinorCoefficient[0]):
            modePred = a
        else:
            modePred = a+12
    
        modePred = utils.lerch_to_str(modePred)
    
        if DB == 'GTZAN':
            pred[gen].append(modePred)
        else:
            pred.append('?')  # you may ignore this when starting with GTZAN dataset
        # print(pred[gen])
    
    print("***** Q2 *****")
    if DB == 'GTZAN':
        label_list, pred_list = list(), list()
        print("Genre    \taccuracy")
        for g in GENRE:
            # TODO: Calculate the accuracy for each genre
            # Hint: Use label[g] and pred[g]
            correct = 0
            for acc_len in range(len(label[g])):
                if label[g][acc_len] == pred[g][acc_len]:
                    correct += 1
            try:
                acc = correct / len(label[g])
            except ZeroDivisionError:
                acc = 0
            print("{:9s}\t{:8.2%}".format(g, acc))
            label_list += label[g]
            pred_list += pred[g]
    else:
        label_list = label
        pred_list = pred
    
    # TODO: Calculate the accuracy for all file.
    # Hint1: Use label_list and pred_list.
    correct_all = 0
    for acc_len in range(len(label_list)):
        if label_list[acc_len] == pred_list[acc_len]:
            correct_all += 1
    try:
        acc_all = correct_all / len(label_list)
    except ZeroDivisionError:
        acc_all = 0
    ##########
    print("----------")
    print("Overall accuracy:\t{:.2%}".format(acc_all))
    idx+=1
    gamma*=10
