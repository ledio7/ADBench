import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns
import pandas as pd

save_dir = os.path.join('Results', 'Main_Plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def create_curves(roc_rates, pr_rates):

    # #####
    #Multiple ROC Curves
    # grouped_roc_rates = {}
    # for rate in rates:
    #     key = (rate[0], rate[1])  # usa il modello e il dataset come chiave
    #     if key not in grouped_rates:
    #         grouped_rates[key] = {'fpr': [], 'tpr': []}
    # # aggiungi le fpr e le tpr alla lista appropriata
    #     grouped_rates[key]['fpr'].append(rate[2])
    #     grouped_rates[key]['tpr'].append(rate[3])
    # # print(f' Grouped rates: {grouped_rates}; \n Keys: {key}')
    # mean_rates = []
    # for key, value in grouped_rates.items():
    #     mean_fpr = np.mean(value['fpr'], axis=0)
    #     mean_tpr = np.mean(value['tpr'], axis=0)
    #     mean_rates.append((key[0], key[1], mean_fpr, mean_tpr))
    #
    # print(mean_rates) #mean
    
    # models = set([m for m, d, fpr, tpr in mean_rates])
    # datasets = set([d for m, d, fpr, trp in mean_rates])
    # # print(models)
    # for dataset in datasets:
        
    #     plt.figure(figsize=(8,8))
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Curva ROC per dataset {dataset}')

    #     for model in models:
    #         fpr, tpr= None, None
    #         for m, d, f, t in mean_rates:
    #             if m == model and d == dataset:
    #                 fpr, tpr = f, t
                    
    #                 #Loda rename
    #                 label = f'{model}'
    #                 if 'loda' in f'{model}':
    #                     label = 'LODA'
                    
    #                 plt.plot(fpr, tpr, label=label)
           
    #     plt.legend()
    #     plt.show()
    ###############################
    

    #Single ROC Curves
    models = set([m for m, d, fpr, tpr, seed in roc_rates])
    datasets = set([d for m, d, fpr, trp, seed in roc_rates])

    for dataset in datasets:

        plt.figure(figsize=(8,8))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for dataset {dataset}')

        for model in models:
            fpr, tpr= None, None
            for m, d, f, t, seed in roc_rates:
                if m == model and d == dataset:
                    fpr, tpr = f, t
                    
                    #Loda rename
                    label = f'{model}'
                    if 'loda' in f'{model}':
                        label = 'LODA'
                    
                    plt.plot(fpr, tpr, label=f'{label}, seed={seed}')
           
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{dataset}_ROC_Curve.png'), bbox_inches='tight')

    #Single PR Curves
    for dataset in datasets:

        plt.figure(figsize=(8,8))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curve for dataset {dataset}')

        for model in models:
            recall, precision= None, None
            for m, d, r, p, seed in pr_rates:
                if m == model and d == dataset:
                    recall, precision = r, p
                    
                    #Loda rename
                    label = f'{model}'
                    if 'loda' in f'{model}':
                        label = 'LODA'
                    
                    plt.plot(recall, precision, label=f'{label}, seed={seed}')
           
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{dataset}_PR_Curve.png'), bbox_inches='tight')

    

def create_main_plots(df):
    
    # Option to plot for all the models and all the datasets
    # df = pd.read_csv("Results\\Benchmark_agg_final.csv")
    # df = df[["dataset", "model", "PR-AUC"]]
    # heatmap_data = df.pivot(index="dataset", columns="model", values="PR-AUC")

    ################################
    # plt.figure(figsize=(8, 6))

    # sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', cbar=True)
    # plt.title('Heatmap della Metrica')
    # plt.xlabel('Dataset')
    # plt.ylabel('Modello')
    # plt.show()


    # For multiple heatmaps: 

    #ROC - PR
    heatmap_data_pr =  df['PR-AUC'].unstack('model')
    heatmap_data_roc =  df['ROC-AUC'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_pr, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0])
    axs[0].set_title('Heatmap PR-AUC')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_roc, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1])
    axs[1].set_title('Heatmap ROC-AUC')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_PR+ROC.png'), bbox_inches='tight')


    #Runtime - Geo-Mean
    heatmap_data_runtime =  df['runtime'].unstack('model')
    heatmap_data_gm =  df['Geo-Mean'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_runtime, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[0])
    axs[0].set_title('Heatmap Runtime')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_gm, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1])
    axs[1].set_title('Heatmap Geo-Mean')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime+GeoMean.png'), bbox_inches='tight')