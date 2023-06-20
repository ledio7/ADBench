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


path_rate = os.path.join('Results', 'temp.csv')

df = pd.read_csv(path_rate, quotechar='"')

# Apply ast.literal_eval to the 'fpr' column
df['fpr'] = df['fpr'].apply(ast.literal_eval)
df['tpr'] = df['tpr'].apply(ast.literal_eval)
df['recall'] = df['recall'].apply(ast.literal_eval)
df['precision'] = df['precision'].apply(ast.literal_eval)
# df['fpr'] = df['fpr'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
# df['tpr'] = df['tpr'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
# df['recall'] = df['recall'].apply(lambda x: [float(val) for val in x.strip('[]').split()])
# df['precision'] = df['precision'].apply(lambda x: [float(val) for val in x.strip('[]').split()])

# df['fpr'] = df['fpr'].apply(ast.literal_eval).astype(float)
# df['tpr'] = df['tpr'].apply(ast.literal_eval).astype(float)
# df['recall'] = df['recall'].apply(ast.literal_eval).astype(float)
# df['precision'] = df['precision'].apply(ast.literal_eval).astype(float)


# df= pd.read_csv(path_rate, index_col=0)
# df = df.drop(df.columns[0], axis=1)

# first_row = df.head(1)


# print(df.columns)
# print(df.head(1))
# create_curves(df)
# print(df_rates.head())  # Print the first few rows of the DataFrame
# print(df.info())
print(df.dtypes)