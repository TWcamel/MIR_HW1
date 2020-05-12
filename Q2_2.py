#!/usr/bin/python
# -*- coding:utf-8 -*-
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

import utils  # self-defined utils.py file

DB = 'Giantsteps'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB+'/wav/*/*.wav')
else:
    FILES = glob(DB+'/wav/*')

n_fft = 100  # (ms)
hop_length = 25  # (ms)

if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()

ind = 0
gamma = 1
while ind<4 and gamma<1001:
    print('gamma',gamma)
    for f in tqdm(FILES):
        f = f.replace('\\', '/')
        # print("file: ", f)
        content = utils.read_keyfile(f, '*.key')
        # print("key: ", content,"\t")
        if (len(content) < 0):
            continue  # skip saving if key not found
        if DB == 'GTZAN':
            gen = f.split('/')[2]
            label[gen].append(utils.LABEL[int(content)])
            gens.append(gen)
        else:
            label.append(content)

        sr, y = utils.read_wav(f)

        cxx = np.log(1 + gamma * np.abs(librosa.feature.chroma_stft(y=y, sr=sr)))
        chromagram.append(cxx)  # store into list for further use
        chroma_vector = np.sum(cxx, 1)
        key_ind = np.where(chroma_vector == np.amax(chroma_vector))
        key_ind = int(key_ind[0])

        mode = dict({'cMajor': [4, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 'cMinor': [
                    4, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]})
        mode['cMajor'] = utils.rotate(mode['cMajor'],  key_ind)
        mode['cMinor'] = utils.rotate(mode['cMinor'],  key_ind)

        cMajorCoefficient = pearsonr(chroma_vector, mode['cMajor'])
        cMinorCoefficient = pearsonr(chroma_vector, mode['cMinor'])
        modePred = ''

        if DB == 'Giantsteps':
            aMajorTo_cMajor = (key_ind+3)%12
            if (cMajorCoefficient[0] > cMinorCoefficient[0]):
                modePred = aMajorTo_cMajor
            else:
                modePred = aMajorTo_cMajor+12
            modePred = utils.lerch_to_str(modePred)
        else:
            if (cMajorCoefficient[0] > cMinorCoefficient[0]):
                modePred = key_ind
            else:
                modePred = key_ind+12
            modePred = utils.lerch_to_str(modePred)

        if DB == 'Giantsteps':
            pred.append(modePred)
        else:
            pred.append('?')

        label_list = label
        pred_list = pred

        for idx, item in enumerate(label_list):
            a = ' '.join(item.split()[0]).split()
            if (len(a)>1):
                temp = item.split()[0] = a[0]+'#'
                label_list[idx] = temp + ' ' + item.split()[1]

    print("***** Q1-GiantSteps *****")
    if DB == 'Giantsteps':
        correct_all = 0
        for acc_len in range(len(label_list)):
            if label_list[acc_len] == pred_list[acc_len]:
                correct_all += 1
        try:
            acc_all = correct_all / len(label_list)
        except ZeroDivisionError:
            acc_all = 0
    print("----------")
    print("Overall accuracy:\t{:.2%}".format(acc_all))
    ind+=1
    gamma*=10
