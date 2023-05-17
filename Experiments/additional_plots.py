import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def create_plots(df):

    bar_width = 0.35

    # datasets = df.index.get_level_values('dataset').unique()
    # models = df.index.get_level_values('model').unique() 

    datasets = df['dataset'].unique()
    models = df['model'].unique()
        
    for dataset in datasets:
        
        save_dir = os.path.join('Results', 'Additional_Plots', f'{dataset}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #PR-AUC, ROC-AUC, Geo-Mean, runtime
        for metric in ['PR-AUC', 'ROC-AUC', 'Geo-Mean', 'runtime']:
            fig = plt.figure()
            plt.bar(models, df[metric][df['dataset']==dataset], color='green')

            plt.xlabel('Algorithm')
            plt.ylabel(f'{metric}')
            plt.title(f'Comparison of {metric} values on dataset {dataset}')

            fig.savefig(os.path.join(save_dir, f'{dataset}_{metric}.png'), bbox_inches='tight')
            plt.close()
        
        
        r1 = np.arange(len(models))
        r2 = [x + bar_width for x in r1]

        #F1_0, F1_1
        fig1, ax = plt.subplots()
        ax.bar(r1, df['F1_1'][df['dataset']==dataset], width=bar_width, label='Anomaly')
        ax.bar(r2, df['F1_0'][df['dataset']==dataset], width=bar_width, label='Normal')

        ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
        ax.set_xticklabels(models)
        ax.set_ylabel('F1 score')
        ax.set_title('F1 score for anomaly and normal class for dataset {}'.format(dataset))
        ax.legend()

        fig1.savefig(os.path.join(save_dir, f'{dataset}_F1.png'), bbox_inches='tight')
        plt.close()

        #Recall1, Recall0
        fig2, ax = plt.subplots()
        ax.bar(r1, df['Recall_1'][df['dataset']==dataset], width=bar_width, label='Anomaly')
        ax.bar(r2, df['Recall_0'][df['dataset']==dataset], width=bar_width, label='Normal')

        
        ax.set_xticks([r + bar_width / 2 for r in range(len(models))])
        ax.set_xticklabels(models)
        ax.set_ylabel('Recall')
        ax.set_title('Recall for anomaly and normal class for dataset {}'.format(dataset))
        ax.legend()

        fig2.savefig(os.path.join(save_dir, f'{dataset}_Recall.png'), bbox_inches='tight')
        plt.close()


csv_file = os.path.join("Results", "Benchmark_agg.csv")

if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    create_plots(df)
else:
    # raise FileNotFoundError(f"Il file '{csv_file}' non esiste.")
    print(f"File '{csv_file}' doesn't exist. You must run benchmark.py in order to have a valid file to plot results. ")