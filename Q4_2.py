#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Example code for key detection preprocessing. You may start with filling the
'?'s below. There're also some description and hint within comment. However,
please feel free to modify anything as you like!

@author: selly
"""
# %%
from glob import glob
from collections import defaultdict

# Below are packages and functions that you might need. Uncomment by remove the
# "#" in front of each line.
import librosa
# from librosa.feature import chroma_stft, chroma_cqt, chroma_cens
from scipy.stats import pearsonr
# from mir_eval.key import weighted_score
from sklearn.metrics import accuracy_score
import numpy as np  # np.log10()

import utils  # self-defined utils.py file
DB = 'GTZAN'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB+'/wav/*/*.wav')
else:
    FILES = glob(DB+'/wav/*.wav')

GENRE = [g.split('/')[2]
         for g in glob(DB+'/wav/*')]
GENRE.remove('classical')
print(GENRE)
n_fft = 100  # (ms)
hop_length = 25  # (ms)

# %% Q1
if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()
for f in FILES:
    f = f.replace('\\', '/')
    # print("file: ", f)
    content = utils.read_keyfile(f, '*.lerch.txt')
    if (int(content) < 0):
        continue  # skip saving if key not found
    if DB == 'GTZAN':
        gen = f.split('/')[2]

        label[gen].append(utils.LABEL[int(content)])
        gens.append(gen)
    else:
        label.append(utils.LABEL[content])

    sr, y = utils.read_wav(f)

    cxx = librosa.feature.chroma_stft(sr=sr, y=y)
    chromagram.append(cxx)  # store into list for further use

    chroma_vector = np.sum(cxx, 1)
    key_ind = np.where(chroma_vector == np.amax(chroma_vector))[0][0]

    mode = dict({'majorProfile': [4, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 'cMinor': [
                4, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]})
    mode['cMajor'] = utils.rotate(mode['cMajor'],  key_ind)
    mode['cMinor'] = utils.rotate(mode['cMinor'],  key_ind)

#    print('key_ind',key_ind)
#    print('cMajor',mode['cMajor'])
#    print('cMinor',mode['cMinor'])
#    print('chroma_vector',chroma_vector)

    cMajorCoefficient = pearsonr(chroma_vector, mode['cMajor'])
    cMinorCoefficient = pearsonr(chroma_vector, mode['cMinor'])
    modePred = ''
    a = (key_ind+3)%12
    if (cMajorCoefficient[0] > cMinorCoefficient[0]):
        modePred = a
    else:
        modePred = a+12

    modePred = utils.lerch_to_str(modePred)

    #print(utils.lerch_to_str(modePred))
#    print('cMajorKey',key_ind, 'cMajorMode',modePred)
#    print('key_ind',key_ind)
#    print('cMajor',mode['cMajor'])
#    print('cMinor',mode['cMinor'])
#    print('chroma_vector',chroma_vector)
#    print('chroma_vector_ro',chroma_vector_ro,'\n')
#
    if DB == 'GTZAN':
        pred[gen].append(modePred)
    else:
        # you may ignore this when starting with GTZAN dataset
        pred.append('?')
        ##########


print("***** Q1-GTZAN *****")
if DB == 'GTZAN':
    label_list, pred_list = list(), list()
    print("Genre\taccuracy")
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
        print("{:9s}\t{:.2%}".format(g, acc))
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
p
