from glob import glob
from collections import defaultdict
import librosa
import numpy as np
from scipy.stats import pearsonr
import mir_eval 
from tqdm import tqdm 
import utils  # self-defined utils.py file

DB = 'Giantsteps'
if DB == 'GTZAN':  # dataset with genre label classify at parent directory
    FILES = glob(DB + '/wav/*/*.wav')
    # print(FILES)
else:
    FILES = glob(DB + '/wav/*.wav')
    # print(FILES)

GENRE = [g.split('/')[2] for g in glob(DB + '/wav/*')]
n_fft = 100  # (ms)
hop_length = 25  # (ms)

if DB == 'GTZAN':
    label, pred = defaultdict(list), defaultdict(list)
else:
    label, pred = list(), list()
chromagram = list()
gens = list()

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

    cxx = librosa.feature.chroma_stft(sr=sr, y=y)
    chromagram.append(cxx)  # store into list for further use

    chroma_vector = np.sum(cxx, 1)
    key_ind = np.where(chroma_vector == np.amax(chroma_vector))[0][0]

    mode = dict({'cMajor': [4, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], 'cMinor': [
                4, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]})
    mode['cMajor'] = utils.rotate(mode['cMajor'],  key_ind)
    mode['cMinor'] = utils.rotate(mode['cMinor'],  key_ind)

    cMajorCoefficient = pearsonr(chroma_vector, mode['cMajor'])
    cMinorCoefficient = pearsonr(chroma_vector, mode['cMinor'])
    modePred = ''

    if DB == 'Giantsteps':
        aMajorTo_cMajor = (key_ind+3) % 12
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

print("***** Q3 *****")
# TODO: Calculate the accuracy for all file.
# Hint1: Use label_list and pred_list.
sum_all = 0
for acc_len in range(len(label_list)):
    score_all = mir_eval.key.weighted_score(
            '{:9s}'.format(label_list[acc_len]),
            pred_list[acc_len]
            )
    sum_all += score_all
try:
    acc_all = sum_all / len(label_list)
except ZeroDivisionError:
    acc_all = 0
##########
print("----------")
print("Overall accuracy:\t{:.2%}".format(acc_all))
