import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def create_plots(df):

    bar_width = 0.35

    datasets = df.index.get_level_values('dataset').unique()
    models = df.index.get_level_values('model').unique() 

    
    for dataset in datasets:
        #PR-AUC, ROC-AUC, Geo-Mean
        for metric in ['PR-AUC', 'ROC-AUC', 'Geo-Mean']:
            fig = plt.figure()
            plt.bar(models, df[metric][df.index.get_level_values('dataset')==dataset], color='green')

            plt.xlabel('Algorithm')
            plt.ylabel(f'{metric}')
            plt.title(f'Comparison of {metric} values on dataset {dataset}')

            fig.savefig(os.path.join('Results', 'Plots', f'{dataset}_{metric}.png'), bbox_inches='tight')

        
        
        r1 = np.arange(len(models))
        r2 = [x + bar_width for x in r1]

        #F1_0, F1_1
        fig1, ax = plt.subplots()
        ax.bar(r1, df['F1_1'][df.index.get_level_values('dataset')==dataset], width=bar_width, label='Anomaly')
        ax.bar(r2, df['F1_0'][df.index.get_level_values('dataset')==dataset], width=bar_width, label='Normal')

        ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
        ax.set_xticklabels(models)
        ax.set_ylabel('F1 score')
        ax.set_title('F1 score for anomaly and normal class for dataset {}'.format(dataset))
        ax.legend()

        fig1.savefig(os.path.join('Results', 'Plots', f'{dataset}_F1.png'), bbox_inches='tight')


        #Recall1, Recall0
        fig2, ax = plt.subplots()
        ax.bar(r1, df['Recall_1'][df.index.get_level_values('dataset')==dataset], width=bar_width, label='Anomaly')
        ax.bar(r2, df['Recall_0'][df.index.get_level_values('dataset')==dataset], width=bar_width, label='Normal')

        
        ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
        ax.set_xticklabels(models)
        ax.set_ylabel('Recall')
        ax.set_title('Recall for anomaly and normal class for dataset {}'.format(dataset))
        ax.legend()

        fig2.savefig(os.path.join('Results', 'Plots', f'{dataset}_Recall.png'), bbox_inches='tight')


    #Runtime  
    
    fig3, ax = plt.subplots()
    for model in models:
        ax.plot(datasets, df['runtime'][df.index.get_level_values('model')==model], label=model)

    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime comparison')
    ax.legend()

    fig3.savefig(os.path.join('Results', 'Plots', 'Runtime.png'), bbox_inches='tight')    
