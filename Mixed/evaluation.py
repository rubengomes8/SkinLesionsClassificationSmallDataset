import pandas as pd
import numpy

ben_empty = False
mal_empty = False
mel_empty = False
flat_empty = False

if not ben_empty:
    path_ben_csv = '/home/ruben/Desktop/teste_ben_indeta.csv'
    df_ben = pd.read_csv(path_ben_csv)
    ben_pred = df_ben['BEN_PRED'].to_numpy()
    ben = df_ben['BEN_TRUTH'].to_numpy()
    for i in range(0, len(ben_pred)):
        ben_pred[i] += 4

if not mal_empty:
    path_mal_csv = '/home/ruben/Desktop/teste_mal_indeta.csv'
    df_mal = pd.read_csv(path_mal_csv)
    mal_pred = df_mal['MAL_PRED'].to_numpy()
    mal = df_mal['MAL_TRUTH'].to_numpy()
    for i in range(0, len(mal_pred)):
        mal_pred[i] += 2


path_mel_csv = '/home/ruben/Desktop/teste_mel_indeta.csv'
path_flat_csv = '/home/ruben/Desktop/teste_flat_indeta.csv'

df_mel = pd.read_csv(path_mel_csv)

df_flat = pd.read_csv(path_flat_csv)


mel_pred = df_mel['MEL_PRED'].to_numpy()
mel = df_mel['MEL_TRUTH'].to_numpy()
flat_pred = df_flat['FLAT_PRED'].to_numpy()
flat = df_flat['FLAT_TRUTH'].to_numpy()




'''
print("MAL:")
print(mal_pred)
print(mal)
print("BEN:")
print(ben_pred)
print(ben)
print("MEL:")
print(mel_pred)
print(mel)
'''

counter_truth = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
counter_pred = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}

conf = numpy.zeros(shape=(7, 7), dtype=int)

total = 0
# MEL
if not mel_empty:
    for i in range(0, len(mel)):
        total += 1
        conf[mel[i]][mel_pred[i]] += 1
        counter_truth[str(mel[i])] += 1
        counter_pred[str(mel_pred[i])] += 1

# BEN
if not ben_empty:
    for i in range(0, len(ben)):
        total += 1
        conf[ben[i]][ben_pred[i]] += 1
        counter_truth[str(ben[i])] += 1
        counter_pred[str(ben_pred[i])] += 1

# MAL
if not mal_empty:
    for i in range(0, len(mal)):
        total += 1
        conf[mal[i]][mal_pred[i]] += 1
        counter_truth[str(mal[i])] += 1
        counter_pred[str(mal_pred[i])] += 1

# FLAT
if not flat_empty:
    for i in range(0, len(flat)):
        total += 1
        conf[flat[i]][flat_pred[i]] += 1
        counter_truth[str(flat[i])] += 1
        counter_pred[str(flat_pred[i])] += 1

print(conf)

print("TOTAL: ", total)