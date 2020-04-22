import pandas as pd
import numpy

path_mel_csv = '/home/ruben/Desktop/teste_mel3.csv'
path_mal_csv = '/home/ruben/Desktop/teste_mal3.csv'
path_ben_csv = '/home/ruben/Desktop/teste_ben3.csv'

df_mel = pd.read_csv(path_mel_csv)
df_mal = pd.read_csv(path_mal_csv)
df_ben = pd.read_csv(path_ben_csv)

print(df_mal)

mal_pred = df_mal['MAL_PRED'].to_numpy()
mal = df_mal['MAL_TRUTH'].to_numpy()
ben_pred = df_ben['BEN_PRED'].to_numpy()
ben = df_ben['BEN_TRUTH'].to_numpy()
mel_pred = df_mel['MEL_PRED'].to_numpy()
mel = df_mel['MEL_TRUTH'].to_numpy()

for i in range(0, len(mal_pred)):
    mal_pred[i] += 2

for i in range(0, len(ben_pred)):
    ben_pred[i] += 4

counter_truth = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}
counter_pred = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0}

conf = numpy.zeros(shape=(7, 7), dtype=int)

# MEL
for i in range(0, len(mel)):
    conf[mel[i]][mel_pred[i]] += 1
    counter_truth[str(mel[i])] += 1
    counter_pred[str(mel_pred[i])] += 1

# BEN
for i in range(0, len(ben)):
    conf[ben[i]][ben_pred[i]] += 1
    counter_truth[str(ben[i])] += 1
    counter_pred[str(ben_pred[i])] += 1

# MAL
for i in range(0, len(mal)):
    conf[mal[i]][mal_pred[i]] += 1
    counter_truth[str(mal[i])] += 1
    counter_pred[str(mal_pred[i])] += 1

print(conf)