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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

outliers_percentage = {
    "covertype": 0.96,
    "creditcard": 0.17,
    "shuttle": 7.15,
    "smtp": 0.03,
    "http": 0.4,
    "satimage-2": 1.2,
    "annthyroid": 7.42,
    "letter": 6.25,
    "mammography": 2.32,
    "musk": 3.2,
    "optdigits": 3,
    "pendigits": 2.27,
    "thyroid": 2.5,
    "vowels": 3.4,
    "cardio": 9.6,
    "mnist": 9.2,
    "speech": 1.65,
    "wbc": 5.6,
    "breastw": 35,
    "arrhythmia": 15,
}

features_number = {
    "covertype": 10,
    "creditcard": 30,
    "shuttle": 9,
    "smtp": 3,
    "http": 3,
    "satimage-2": 36,
    "annthyroid": 6,
    "letter": 32,
    "mammography": 6,
    "musk": 166,
    "optdigits": 64,
    "pendigits": 16,
    "thyroid": 6,
    "vowels": 12,
    "cardio": 21,
    "mnist": 100,
    "speech": 400,
    "wbc": 30,
    "breastw": 9,
    "arrhythmia": 274,
}

# final_score = {
#     "covertype": 10*286048/0.96,
#     "creditcard": 30*284807/0.17,
#     "shuttle": 9*49097/7.15,
#     "smtp": 3*95156/0.03,
#     "http": 3*567497/0.4,
#     "satimage-2": 36*5803/1.2,
#     "annthyroid": 6*7200/7.42,
#     "letter": 32*1600/6.25,
#     "mammography": 6*11183/2.32,
#     "musk": 166*3062/3.2,
#     "optdigits": 64*5216/3,
#     "pendigits": 16*6870/2.27,
#     "thyroid": 6*3772/2.5,
#     "vowels": 12*1456/3.4,
#     "cardio": 21*1831/9.6,
#     "mnist": 100*7603/9.2,
#     "speech": 400*3686/1.65,
#     "wbc": 30*278/5.6,
#     "breastw": 9*683/35,
#     "arrhythmia": 274*452/15,
# }
samples = {
    "covertype":286048,
    "creditcard":284807,
    "shuttle":49097,
    "smtp":95156,
    "http":567497,
    "satimage-2":5803,
    "annthyroid":7200,
    "letter":1600,
    "mammography":11183,
    "musk": 3062,
    "optdigits":5216,
    "pendigits":6870,
    "thyroid":3772,
    "vowels":1456,
    "cardio":1831,
    "mnist": 7603,
    "speech": 3686,
    "wbc":278,
    "breastw":683,
    "arrhythmia": 452,
}


def adjust_heatmaps(df): # Ordering
    annot_font_props = {'size': 10}

    ## Outlier percentage order
    # sorted_outliers_percentage = {k: v for k, v in sorted(outliers_percentage.items(), key=lambda item: item[1], reverse=False)}
    # desired_order = list(sorted_outliers_percentage.keys())

    ## Features number order
    # sorted_features_number = {k: v for k, v in sorted(features_number.items(), key=lambda item: item[1], reverse=False)}
    # desired_order = list(sorted_features_number.keys())

    # Samples order
    sorted_samples = {k: v for k, v in sorted(samples.items(), key=lambda item: item[1], reverse=False)}
    desired_order = list(sorted_samples.keys())
    
    df['dataset'] = pd.Categorical(df['dataset'], categories=desired_order, ordered=True)
    metrics = ['PR-AUC', 'ROC-AUC', 'F1_1', '']

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 1, 0.1]})
    
    for i, metric in enumerate(metrics):
        if i<3:
            heatmap_data = df.pivot(index='dataset', columns='model', values=metric)
                
            sns.heatmap(heatmap_data, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=False, ax=axs[i], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
                
            axs[i].set_title(f'{metric} (ordered by #samples)')
            # axs[i].set_xlabel('Modello')
            if i ==1:
                axs[i].set_xlabel('Models')
            else:
                axs[i].set_xlabel('')

            if i != 0:
                axs[i].set_yticks([])

            axs[i].set_ylabel('')
        else: 
            cbar = fig.colorbar(axs[2].collections[0], cax=axs[3], ticks=[0, 0.5, 1])
            cbar.ax.set_aspect(40)
            # cbar.set_label('Colorbar Title', rotation=270, labelpad=15)
    plt.subplots_adjust(wspace=10)    
    plt.tight_layout()
    save_dir = os.path.join('Results', 'Main_Plots')
    plt.savefig(os.path.join(save_dir, f'Heatmap_Samples.png'), dpi=200)
    


# def adjust_formats(df): # Formats, style, colors
    # annot_font_props = {'size': 8}
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    
    # heatmap_data_runtime = df.pivot(index='dataset', columns='model', values='runtime')

    # flattened_data = heatmap_data_runtime.values.flatten() 
    # vmin = np.min(flattened_data)
    # vmax = np.max(flattened_data)

    # sns.heatmap(heatmap_data_runtime, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[0], annot_kws = annot_font_props, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), fmt='.1f')
    # axs[0].set_title('Heatmap Runtime (s)')
    # axs[0].set_xlabel('')
    # axs[0].set_ylabel('Datasets')

    # heatmap_data_ram = df.pivot(index='dataset', columns='model', values='RAMHours')

    # flattened_data = heatmap_data_ram.values.flatten() 
    # vmin = np.min(flattened_data)
    # vmax = np.max(flattened_data)
    # sns.heatmap(heatmap_data_ram, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[1], annot_kws = annot_font_props, norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
    # axs[1].set_title('Heatmap RAMHours (Gb)')
    # axs[1].set_xlabel('')
    # axs[1].set_ylabel('')


    # plt.tight_layout()
    # # plt.show()
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime+RAMHours.png'), dpi=200)

    #######################################à

    # annot_font_props = {'size': 8}
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    
    # heatmap_data_pr = df.pivot(index='dataset', columns='model', values='PR-AUC')

    # sns.heatmap(heatmap_data_pr, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[0].set_title('Heatmap PR-AUC')
    # axs[0].set_xlabel('')
    # axs[0].set_ylabel('Datasets')

    # heatmap_data_roc = df.pivot(index='dataset', columns='model', values='ROC-AUC')

    # sns.heatmap(heatmap_data_roc, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[1].set_title('Heatmap ROC-AUC')
    # axs[1].set_xlabel('')
    # axs[1].set_ylabel('')


    # plt.tight_layout()
    # # plt.show()
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_PR+ROC.png'), dpi=200)

    ###############################################

    # annot_font_props = {'size': 8}
    # fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 8))
    
    # heatmap_data_pr = df.pivot(index='dataset', columns='model', values='Recall_0')

    # sns.heatmap(heatmap_data_pr, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[0].set_title('Heatmap Recall Class 0')
    # axs[0].set_xlabel('')
    # axs[0].set_ylabel('Datasets')

    # heatmap_data_roc = df.pivot(index='dataset', columns='model', values='Recall_1')

    # sns.heatmap(heatmap_data_roc, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], annot_kws = annot_font_props, vmin=0.0, vmax=1.0)
    # axs[1].set_title('Heatmap Recall Class 1')
    # axs[1].set_xlabel('')
    # axs[1].set_ylabel('')


    # plt.tight_layout()
    # # plt.show()
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_Recall.png'), dpi=200)

    # #######################################################

    # annot_font_props = {'size': 8}
    # heatmap_data_geo = df.pivot(index='dataset', columns='model', values='Geo-Mean')
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(heatmap_data_geo, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, annot_kws = annot_font_props)
    # plt.title('Heatmap Geometric Mean')
    # plt.xlabel('Model')
    # plt.ylabel('Dataset')
    # save_dir = os.path.join('Results', 'Main_Plots')
    # plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), dpi=200)
    # # plt.show()



csv_file = os.path.join("Results", "Benchmark_agg.csv")

if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    adjust_heatmaps(df)
    # adjust_formats(df)
else:
    print(f"File '{csv_file}' doesn't exist. You must run benchmark.py in order to have a valid file to plot results. ")