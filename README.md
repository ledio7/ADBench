# Anomaly Detection Benchmark
Benchmark for unsupervised learning algorithms applied to anomaly detection.

This repository contains a collection of unsupervised anomaly detection algorithms, which are useful for identifying anomalous behavior in large datasets.

The goal is to provide a benchmark of these algorithms on various datasets, allowing for easy comparison of their performance.

## Installation

Python requirement for this project: 
`python_version <= 3.9`
  
`git clone https://github.com/ledio7/simple_benchmark.git`

`cd simple_benchmark`

`python3 -m venv bench`

`source bench/bin/activate` *For mac0S/Linux*   or   `bench\Scripts\activate` *For Windows*

`pip install -r requirements.txt`
  
  
## How to obtain the results

### Main Results

`cd Experiments`

`python benchmark.py`

The results will be stored in the *./Experiment/Results* folder and they include: 

  * CSV file with metric values for all the runs
  
  * CSV file with aggregated metric values 
  
  * PR Curve and ROC Curve for each dataset (in *Main_Plots* folder)
  
  * Heatmaps (in *Main_Plots* folder)

Since there are large datasets and multiple runs for each dataset, the overall run takes several hours; if the user needs faster results, he/she can specify a smaller number of samples to process:

`python benchmark.py *number_of desired_samples*`

Bear in mind that if the number of desired samples is too small, some metric values will show 0 as result because there were not sufficient samples.

### Additional Results

In order to have further details about the results, the user can run: 

`python additional_plots.py`

which produces additional plots about each specific metric in the *additional_plots* folder for each dataset. 

### Statistical Test

In order to perform the Nemenyi Test and compare the algorithms (after the runs) on a specific metric:

`python statistic_test.py *desired_metric*`

where the metric must be chosen from one of these: *PR-AUC, ROC-AUC, F1_O, F1_1, Recall_0, Recall_1, Geo-Mean* (default: *PR-AUC*).

