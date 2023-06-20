import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IncrementalTorch.datasets import Covertype, Shuttle
from river.anomaly import *
from river.datasets import *
import sys
import matplotlib.colors as mcolors


def adjust_heatmaps(df):
    annot_font_props = {'size': 8}
    # subset_df = df[['dataset', 'model', 'ROC-AUC', 'anomalies', 'samples']]
    # subset_df['anomaly_ratio'] = subset_df['anomalies'] / subset_df['samples']
    # sorted_df = subset_df.sort_values(by='anomaly_ratio')
    # sorted_datasets = sorted_df['dataset'].unique()
    # dataset_order_mapping = {dataset: order for order, dataset in enumerate(sorted_datasets)}

    # # Add a new column to reROCesent the order of the datasets
    # sorted_df['dataset_order'] = sorted_df['dataset'].map(dataset_order_mapping)

    # # ROCint(sorted_df)
    # heatmap_data_geo = sorted_df.pivot(index='dataset', columns='model', values='ROC-AUC')
    # sorted_heatmap_data_geo = heatmap_data_geo.reindex(sorted_datasets)
    # # print(heatmap_data_geo) 
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(sorted_heatmap_data_geo, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, annot_kws = annot_font_props)
    # plt.title('Heatmap Geometric Mean')
    # plt.xlabel('Model')
    # plt.ylabel('Dataset')
    # # save_dir = os.path.join('Results', 'Main_Plots')
    # # plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), dpi=200)
    # plt.show()
    ######

    # Runtime
    # heatmap_data_runtime = df.pivot(index='dataset', columns='model', values='runtime')
    # plt.figure(figsize=(10, 8))
    # flattened_data = heatmap_data_runtime.values.flatten() 
    # vmin = np.min(flattened_data)
    # vmax = np.max(flattened_data)
    # sns.heatmap(heatmap_data_runtime, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, annot_kws = annot_font_props, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), fmt='.1f')
    # plt.title('Heatmap Runtime (s)')
    # plt.xlabel('Model')
    # plt.ylabel('Dataset')
    # plt.show()
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime.png'), dpi=200)
    ####

    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    
    # heatmap_data_pr = df.pivot(index='dataset', columns='model', values='F1_0')

    # sns.heatmap(heatmap_data_pr, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[0].set_title('Heatmap F1 Class 0')
    # axs[0].set_xlabel('')
    # axs[0].set_ylabel('Datasets')

    # heatmap_data_roc = df.pivot(index='dataset', columns='model', values='F1_1')

    # sns.heatmap(heatmap_data_roc, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[1].set_title('Heatmap F1 Class 1')
    # axs[1].set_xlabel('')
    # axs[1].set_ylabel('')
    # plt.tight_layout()
    # # plt.show()
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_F1.png'), dpi=200)
    ########

csv_file = os.path.join("Benchmark_agg.csv")

if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    adjust_heatmaps(df)
    # adjust_formats(df)
else:
    print(f"File '{csv_file}' doesn't exist. You must run benchmark.py in order to have a valid file to plot results. ")