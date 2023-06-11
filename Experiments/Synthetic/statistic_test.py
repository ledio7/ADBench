import pandas as pd
import Orange
import matplotlib.pyplot as plt
import os
import sys

METRICS = ["PR-AUC", "ROC-AUC", "F1_0", "F1_1", "Recall_0", "Recall_1", "Geo-Mean", "runtime"]

def test(df):
    if len(sys.argv) < 2:
        metric="PR-AUC"
    else: 
        metric = sys.argv[1]

    if metric not in METRICS:
        print("Wrong metric, check your input metric name")
    else:


        save_dir = os.path.join('Results', 'Statistic_Tests', f'{metric}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # df = pd.read_csv("Results\\Benchmark_agg_final.csv")#20 datasets
        # df = pd.read_csv("Results\\Temp.csv")#14 datasets
        # df = pd.read_csv("Results\\prova.csv")
        
        # Columns
        df = df[["dataset", "model", f"{metric}"]]

        # ["AE", "DAE", "PW-AE", "HST", "ILOF", "LODA"]
        # df = df[df["model"].isin(["AE", "DAE", "PW-AE", "HST", "ILOF", "LODA"])]

        # Reshape
        df = df.pivot(index="dataset", columns="model", values=f"{metric}")
        # print(df , "\n")

        # Classify
        ranks = df.transpose().rank(method='min', ascending=False)
        # print(ranks.transpose(), "\n")
        

        # from scipy.stats import friedmanchisquare
        # data = [ranks[col].values for col in ranks.columns]
        # statistic, p_value = friedmanchisquare(*data)
        # print(p_value)

        #Avg ranks for every model
        avranks = ranks.mean(axis=1)

        av = pd.DataFrame(avranks)
        av = av.rename_axis('Model').reset_index()
        av.columns = ['Model', 'Value']
        
        #plot ranks and avranks together
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

        axes[0].axis('off') 
        table = axes[0].table(cellText=ranks.transpose().values, colLabels=ranks.transpose().columns,  cellLoc='center', loc='center')
        table.set_fontsize(14)
        table.scale(1, 1.7)
        axes[0].set_title('')

        axes[1].axis('off')
        table1 = axes[1].table(cellText=av.values, colLabels=av.columns,  cellLoc='center', loc='center')
        table1.set_fontsize(14)
        table1.scale(1, 1.7)
        axes[1].set_title('Average Ranks')

        plt.savefig(os.path.join(save_dir, f'{metric}_Rankings.png'), bbox_inches='tight')

        # Last plot 
        cd = Orange.evaluation.compute_CD(avranks, 30, alpha='0.05', test='nemenyi')
        Orange.evaluation.graph_ranks(avranks, avranks.index, cd=cd, width=8, textspace=2)
        plt.title(f'CD = {cd}')
        plt.savefig(os.path.join(save_dir, f'{metric}_Nemenyi.png'), bbox_inches='tight')


csv_file = os.path.join("Benchmark_agg.csv")
if os.path.isfile(csv_file):
    df = pd.read_csv(csv_file)
    test(df)
else:
    print(f"File '{csv_file}' doesn't exist. You must run benchmark.py in order to have a valid file to plot results. ")