import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors


save_dir = os.path.join('Results', 'Main_Plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
   

def create_main_plots(df):
    
    # For multiple heatmaps: 
    annot_font_props = {'size': 8}
    #ROC - PR
    heatmap_data_pr =  df['PR-AUC'].unstack('model')
    heatmap_data_roc =  df['ROC-AUC'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

    sns.heatmap(heatmap_data_pr, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0, fmt='.3f')
    axs[0].set_title('Heatmap PR-AUC')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_roc, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0, fmt='.3f')
    axs[1].set_title('Heatmap ROC-AUC')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_PR+ROC.png'), dpi=200)


    # F1 Class 0 - F1 Class 1
    heatmap_data_F10 =  df['F1_0'].unstack('model')
    heatmap_data_F11 =  df['F1_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

    sns.heatmap(heatmap_data_F10, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap F1 Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_F11, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap F1 Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_F1.png'), dpi=200)


    # Recall Class 0 - Recall Class 1
    heatmap_data_Recall0 =  df['Recall_0'].unstack('model')
    heatmap_data_Recall1 =  df['Recall_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))

    sns.heatmap(heatmap_data_Recall0, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap Recall Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_Recall1, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap Recall Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Recall.png'), dpi=200)

    # Geo-Mean
    heatmap_data_geo =  df['Geo-Mean'].unstack('model')
    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data_geo, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, annot_kws = annot_font_props)
    plt.title('Heatmap Geometric Mean')
    plt.xlabel('Model')
    plt.ylabel('Dataset')
    plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), dpi=200)

    # Runtime 
    heatmap_data_runtime =  df['runtime'].unstack('model')
    flattened_data = heatmap_data_runtime.values.flatten() 
    vmin = np.min(flattened_data)
    vmax = np.max(flattened_data)

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap_data_runtime, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, annot_kws = annot_font_props, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), fmt='.1f')
    plt.title('Heatmap Runtime (s)')
    plt.xlabel('Model')
    plt.ylabel('Dataset')
    plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime.png'), dpi=200)