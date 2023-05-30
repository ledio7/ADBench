import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns
import pandas as pd

save_dir = os.path.join('Results', 'Main_Plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
   

def create_main_plots(df):
    
    # For multiple heatmaps: 

    #ROC - PR
    heatmap_data_pr =  df['PR-AUC'].unstack('model')
    heatmap_data_roc =  df['ROC-AUC'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_pr, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap PR-AUC')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_roc, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap ROC-AUC')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_PR+ROC.png'), bbox_inches='tight')


    #Runtime - Geomean
    heatmap_data_runtime =  df['runtime'].unstack('model')
    heatmap_data_geo =  df['Geo-Mean'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_runtime, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[0])
    axs[0].set_title('Heatmap Runtime')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_geo, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1])
    axs[1].set_title('Heatmap Geo Mean')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime+GeoMean.png'), bbox_inches='tight')


    # F1 Class 0 - F1 Class 1
    heatmap_data_F10 =  df['F1_0'].unstack('model')
    heatmap_data_F11 =  df['F1_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_F10, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap F1 Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_F11, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap F1 Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_F1.png'), bbox_inches='tight')


    # Recall Class 0 - Recall Class 1
    heatmap_data_Recall0 =  df['Recall_0'].unstack('model')
    heatmap_data_Recall1 =  df['Recall_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_Recall0, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap Recall Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_Recall1, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap Recall Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Recall.png'), bbox_inches='tight')


    # heatmap_data_geo =  df['Geo-Mean'].unstack('model')
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(heatmap_data_geo, annot=True, cmap='RdYlGn', cbar=True, vmin=0.0, vmax=1.0)
    # plt.title('Heatmap Geometric Mean')
    # plt.xlabel('Model')
    # plt.ylabel('Dataset')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), bbox_inches='tight')