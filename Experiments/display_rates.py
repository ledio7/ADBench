import pandas as pd
import multiprocessing as mp
import pathlib
from evaluate import aggregate_dataframe, test_then_train
from main_plots import create_main_plots, create_curves
from tqdm import tqdm
import os
import sys
from ast import literal_eval
import numpy as np
import csv
import ast


path_rate = os.path.join('Results', 'Rates_temp.csv')

df = pd.read_csv(path_rate)
column_names = ['model', 'dataset', 'seed', 'fpr', 'tpr', 'recall', 'precision' ]
print(column_names[3:])
df['recall'] = df['recall'].str.strip('[]').str.split()
df['precision'] = df['precision'].str.strip('[]').str.split()
df['tpr'] = df['tpr'].str.strip('[]').str.split()
df['fpr'] = df['fpr'].str.strip('[]').str.split()

df['recall'] = df['recall'].apply(lambda x: np.array(x, dtype=float))
df['precision'] = df['precision'].apply(lambda x: np.array(x, dtype=float))
df['tpr'] = df['tpr'].apply(lambda x: np.array(x, dtype=float))
df['fpr'] = df['fpr'].apply(lambda x: np.array(x, dtype=float))
# pd.set_option('display.width', None)
# data = df.iloc[1, 3]
# print(data)
# df= pd.read_csv(path_rate, index_col=0)
# df = df.drop(df.columns[0], axis=1)

# first_row = df.head(1)


# print(df.columns)
# print(df.head(1))
# create_curves(df)
# print(df_rates.head())  # Print the first few rows of the DataFrame
# print(df.info())
# print(df.dtypes)