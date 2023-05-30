import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns
import pandas as pd

save_dir = os.path.join('Results', 'Main_Plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def create_curves(rates):

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
    

    # #Single ROC Curves
    # models = set([m for m, d, fpr, tpr, recall, precision, seed in rates])
    # datasets = set([d for m, d, fpr, trp, recall, precision, seed in rates])
    models = rates['model'].unique()
    datasets = rates['dataset'].unique()
    seeds = rates['seed'].unique()
    # for dataset in datasets:

    #     plt.figure(figsize=(8,8))
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'ROC Curve for dataset {dataset}')

    #     for model in models:
    #         fpr, tpr= None, None
    #         for m, d, f, t, seed in roc_rates:
    #             if m == model and d == dataset:
    #                 fpr, tpr = f, t
                    
    #                 #Loda rename
    #                 label = f'{model}'
    #                 if 'loda' in f'{model}':
    #                     label = 'LODA'
                    
    #                 plt.plot(fpr, tpr, label=f'{label}, seed={seed}')
           
    #     plt.legend()
    #     plt.savefig(os.path.join(save_dir, f'{dataset}_ROC_Curve.png'), bbox_inches='tight')

    # #Single PR Curves
    # for dataset in datasets:

    #     plt.figure(figsize=(8,8))
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title(f'PR Curve for dataset {dataset}')

    #     for model in models:
    #         recall, precision= None, None
    #         for m, d, r, p, seed in pr_rates:
    #             if m == model and d == dataset:
    #                 recall, precision = r, p
                    
    #                 #Loda rename
    #                 label = f'{model}'
    #                 if 'loda' in f'{model}':
    #                     label = 'LODA'
                    
    #                 plt.plot(recall, precision, label=f'{label}, seed={seed}')
           
    #     plt.legend()
    #     plt.savefig(os.path.join(save_dir, f'{dataset}_PR_Curve.png'), bbox_inches='tight')
    #################################################

    # Interpolated curves
    fig, axs = plt.subplots(nrows=len(datasets), ncols=len(models), figsize=(8, 8))
    for i, dataset in enumerate(datasets):
        
        for j, model in enumerate(models):
            
            tpr_curves = []
            fpr_curves = []
            
            for seed in seeds:
                # tpr_seed = rates.loc[(rates['dataset'] == dataset) & (rates['model'] == model) &(rates['seed'] == seed), 'tpr']
                # # tpr_seed = tpr_seed.astype(float)
                # tpr_curves.append(tpr_seed)
                # fpr_seed = rates.loc[(rates['dataset'] == dataset) & (rates['model'] == model) &(rates['seed'] == seed), 'fpr']
                # # fpr_seed = fpr_seed.astype(float)
                # fpr_curves.append(fpr_seed)

                filtered_df = rates[(rates['dataset'] == dataset) & (rates['model'] == model) & (rates['seed'] == seed)]

                tpr = filtered_df['tpr'].values[0]
                fpr = filtered_df['fpr'].values[0]
                # plt.plot(fpr, tpr, 'b', alpha=0.15)
                tpr_curves.append(np.array(tpr))
                fpr_curves.append(np.array(fpr))


            # print(tpr_curves)
            max_length = max(len(tpr) for tpr in tpr_curves)
            interp_tpr_curves = []
            interp_fpr_curves = []
            for tpr, fpr in zip(tpr_curves, fpr_curves):
                interp_tpr = np.interp(np.linspace(0, 1, num=max_length), fpr, tpr)
                interp_tpr_curves.append(interp_tpr)
                interp_fpr_curves.append(np.linspace(0, 1, num=max_length))

            mean_curve = np.mean(interp_tpr_curves, axis=0)
            std_curve = np.std(interp_tpr_curves, axis=0)

            axs[i, j].plot(interp_fpr_curves[0], mean_curve, label='Mean Curve')

            axs[i, j].fill_between(interp_fpr_curves[0], mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

            # axs[i, j].plot([0, 1], [0, 1],'r--')
            # axs[i, j].xlim([-0.01, 1.01])
            # axs[i, j].ylim([-0.01, 1.01])
            
            axs[i, j].set_xlabel('False Positive Rate')
            if j == 0: 
                axs[i, j].set_ylabel(f'{dataset} \n\n True Positive Rate')
            else:
                axs[i, j].set_ylabel('True Positive Rate')
            axs[i, j].set_title(f'ROC Curve for {model}')
            axs[i, j].legend()
               
        
            
        
    fig.tight_layout()
    plt.show()


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


    #Runtime - RAMHours
    heatmap_data_runtime =  df['runtime'].unstack('model')
    heatmap_data_ram =  df['RAMHours'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_runtime, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[0])
    axs[0].set_title('Heatmap Runtime')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_ram, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[1])
    axs[1].set_title('Heatmap RAMHours')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime+RAMHours.png'), bbox_inches='tight')


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


    heatmap_data_geo =  df['Geo-Mean'].unstack('model')
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data_geo, annot=True, cmap='RdYlGn', cbar=True)
    plt.title('Heatmap Geometric Mean')
    plt.xlabel('Model')
    plt.ylabel('Dataset')
    plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), bbox_inches='tight')