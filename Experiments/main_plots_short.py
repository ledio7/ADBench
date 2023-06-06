import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os
import seaborn as sns
import pandas as pd
import math

save_dir = os.path.join('Results', 'Main_Plots')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def create_curves(rates):

    # models = set([m for m, d, fpr, tpr, recall, precision, seed in rates])
    # datasets = set([d for m, d, fpr, trp, recall, precision, seed in rates])
    models = rates['model'].unique()
    datasets = rates['dataset'].unique()
    seeds = rates['seed'].unique()

    # Interpolated curves
    # fig, axs = plt.subplots(nrows=len(datasets), ncols=len(models), figsize=(16, 10), sharey=True)
    
    # for i, dataset in enumerate(datasets):
        
    #     for j, model in enumerate(models):
            
    #         tpr_curves = []
    #         fpr_curves = []
            
    #         for seed in seeds:
                

    #             filtered_df = rates[(rates['dataset'] == dataset) & (rates['model'] == model) & (rates['seed'] == seed)]

    #             tpr = filtered_df['tpr'].values[0]
    #             fpr = filtered_df['fpr'].values[0]
    #             # plt.plot(fpr, tpr, 'b', alpha=0.15)
    #             tpr_curves.append(np.array(tpr))
    #             fpr_curves.append(np.array(fpr))


    #         # print(tpr_curves)
    #         max_length = max(len(tpr) for tpr in tpr_curves)
    #         interp_tpr_curves = []
    #         interp_fpr_curves = []
    #         for tpr, fpr in zip(tpr_curves, fpr_curves):
    #             interp_tpr = np.interp(np.linspace(0, 1, num=max_length), fpr, tpr)
    #             interp_tpr_curves.append(interp_tpr)
    #             interp_fpr_curves.append(np.linspace(0, 1, num=max_length))

    #         mean_curve = np.mean(interp_tpr_curves, axis=0)
    #         std_curve = np.std(interp_tpr_curves, axis=0)

    #         axs[i, j].plot(interp_fpr_curves[0], mean_curve, label=F'{model}')

    #         axs[i, j].fill_between(interp_fpr_curves[0], mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

    #         # # axs[i, j].plot([0, 1], [0, 1],'r--')
    #         # # axs[i, j].xlim([-0.01, 1.01])
    #         # # axs[i, j].ylim([-0.01, 1.01])
            
    #         axs[i, j].set_xlabel('FPR')
    #         if j == 0: 
    #             axs[i, j].set_ylabel(f'{dataset} \n\n TPR')
    #         else:
    #             axs[i, j].set_ylabel('TPR')
    #         axs[i, j].set_title(f'ROC Curve {model}')
    #         axs[i, j].legend()
               
    # # fig.tight_layout()
    # fig.subplots_adjust(top=0.97, bottom=0.055, left=0.045, right=0.97, hspace=0.41, wspace=0.21)
    # fig.savefig('nome_immagine.png')
    # plt.show()
    # plt.close(fig)





    # ROC-Curves
    # datasets = ["covertype", "creditcard", "http", "satimage-2", "shuttle", "smtp"]
    nrows = 2
    ncols = len(datasets) // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    for i, dataset in enumerate(datasets):
        # fig, axs = plt.subplots(figsize=(10, 6))
        # # ax = axs[i]
        row = i // ncols 
        col = i % ncols  

        ax = axs[row, col]
        for j, model in enumerate(models):
            
            tpr_curves = []
            fpr_curves = []
            
            for seed in seeds:
                

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

            ax.plot(interp_fpr_curves[0], mean_curve, label=F'{model}')

            # ax.fill_between(interp_fpr_curves[0], mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

            # ax.plot([0, 1], [0, 1],'r--')
            # ax.xlim([-0.01, 1.01])
            # ax.ylim([-0.01, 1.01])
            
            ax.set_xlabel('FPR')
            ax.set_ylabel('TPR')
            ax.set_title(f'Mean ROC Curves {dataset}')
            ax.legend()
               
    fig.tight_layout()
    # fig.subplots_adjust(top=0.97, bottom=0.055, left=0.045, right=0.97, hspace=0.41, wspace=0.21)
    fig.savefig(os.path.join(save_dir, f'ROC_Curves_Main.png'))
    # plt.show()
    # plt.close(fig)



    # PR - CURVES
    # datasets = ["covertype", "creditcard", "http", "satimage-2", "shuttle", "smtp"]
    nrows = 2
    ncols = len(datasets) // nrows
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
    for i, dataset in enumerate(datasets):
        # # fig, axs = plt.subplots(figsize=(10, 6))
        # ax = axs[i]
        row = i // ncols  
        col = i % ncols 

        ax = axs[row, col]
        for j, model in enumerate(models):
            
            precision_curves = []
            recall_curves = []
            
            for seed in seeds:
                

                filtered_df = rates[(rates['dataset'] == dataset) & (rates['model'] == model) & (rates['seed'] == seed)]

                precision = filtered_df['precision'].values[0]
                recall = filtered_df['recall'].values[0]
                # plt.plot(recall, precision, 'b', alpha=0.15)
                precision_curves.append(np.sort(np.array(precision)))
                recall_curves.append(np.sort(np.array(recall))[::-1])


            # print(precision_curves)
            max_length = max(len(precision) for precision in precision_curves)
            interp_precision_curves = []
            interp_recall_curves = []
            for precision, recall in zip(precision_curves, recall_curves):
                interp_precision = np.interp(np.linspace(0, 1, num=max_length), recall[::-1], precision[::-1])
                interp_precision_curves.append(interp_precision)
                interp_recall_curves.append(np.linspace(0, 1, num=max_length))


            mean_curve = np.mean(interp_precision_curves, axis=0)
            std_curve = np.std(interp_precision_curves, axis=0)

            ax.plot(interp_recall_curves[0], mean_curve, label=f'{model}')

            # ax.fill_between(interp_recall_curves[0], mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)

            # # ax.plot([0, 1], [0, 1],'r--')
            # # ax.xlim([-0.01, 1.01])
            # # ax.ylim([-0.01, 1.01])
            
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Mean PR Curves {dataset}')
            ax.legend()
               
    fig.tight_layout()
    # fig.subplots_adjust(top=0.97, bottom=0.055, left=0.045, right=0.97, hspace=0.41, wspace=0.21)
    fig.savefig(os.path.join(save_dir, f'PR_Curves_Main.png'))
    # # plt.show()
    # # plt.close(fig)





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

    sns.heatmap(heatmap_data_pr, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap PR-AUC')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_roc, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap ROC-AUC')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_PR+ROC.png'), bbox_inches='tight')


    #Runtime - RAMHours
    heatmap_data_runtime =  df['runtime'].unstack('model')
    heatmap_data_ram =  df['RAMHours'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_runtime, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[0])
    axs[0].set_title('Heatmap Runtime')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_ram, linewidths=0.5, annot=True, cmap='RdYlGn_r', cbar=True, ax=axs[1])
    axs[1].set_title('Heatmap RAMHours')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Runtime+RAMHours.png'), bbox_inches='tight')


    # F1 Class 0 - F1 Class 1
    heatmap_data_F10 =  df['F1_0'].unstack('model')
    heatmap_data_F11 =  df['F1_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_F10, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap F1 Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_F11, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap F1 Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_F1.png'), bbox_inches='tight')


    # Recall Class 0 - Recall Class 1
    heatmap_data_Recall0 =  df['Recall_0'].unstack('model')
    heatmap_data_Recall1 =  df['Recall_1'].unstack('model')
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.heatmap(heatmap_data_Recall0, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[0], vmin=0.0, vmax=1.0)
    axs[0].set_title('Heatmap Recall Class 0')
    axs[0].set_xlabel('')
    axs[0].set_ylabel('Datasets')

    sns.heatmap(heatmap_data_Recall1, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True, ax=axs[1], vmin=0.0, vmax=1.0)
    axs[1].set_title('Heatmap Recall Class 1')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'Heatmap_Recall.png'), bbox_inches='tight')


    heatmap_data_geo =  df['Geo-Mean'].unstack('model')
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data_geo, linewidths=0.5, annot=True, cmap='RdYlGn', cbar=True)
    plt.title('Heatmap Geometric Mean')
    plt.xlabel('Model')
    plt.ylabel('Dataset')
    plt.savefig(os.path.join(save_dir, f'Heatmap_GeoMean.png'), bbox_inches='tight')