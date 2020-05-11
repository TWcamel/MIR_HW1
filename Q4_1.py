# -*- coding:utf-8 -*-
from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr

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
print(GENRE)
n_fft = 100  # (ms)
hop_length = 25  # (ms)

# %% Q4
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
    if (int(content) < 0): continue  # skip saving if key not found
    if DB == 'GTZAN':
        gen = f.split('/')[2]
        label[gen].append(utils.LABEL[int(content)])
        gens.append(gen)
    else:
        label.append(utils.LABEL[content])

    sr, y = utils.read_wav(f)

    cxx = librosa.feature.chroma_stft(y=y, sr=sr)
    chromagram.append(cxx)  # store into list for further use
    chroma_vector = np.sum(cxx, axis=1)
    key_ind = np.where(chroma_vector == np.amax(chroma_vector))
    key_ind = int(key_ind[0])
    # print('key index: ', key_ind)
    # print('chroma_vector: ', chroma_vector)
    KS = {"cMajor": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
          "cMinor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]}
    KS['cMajor'] = utils.rotate(KS['cMajor'],  key_ind)
    KS['cMinor'] = utils.rotate(KS['cMinor'],  key_ind)

    cMajorCoefficient = pearsonr(chroma_vector, mode['cMajor'])
    cMinorCoefficient = pearsonr(chroma_vector, mode['cMinor'])
    # print(r_co_major[0])
    # print(r_co_minor[0])
    modePred = ''
    a = (key_ind+3)%12
    if (cMajorCoefficient[0] > cMinorCoefficient[0]):
        modePred = a
    else:
        modePred = a+12

    modePred = utils.lerch_to_str(modePred)
    # print('mode', mode)
    if DB == 'GTZAN':
        pred[gen].append(modePred)
    else:
        pred.append('?')  # you may ignore this when starting with GTZAN dataset
    # print(pred[gen])

print("***** Q4 *****")
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
        print("{:9s}\t{:8.2f}%".format(g, acc))
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
print("Overall accuracy:\t{:.2f}%".format(acc_all))
