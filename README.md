# Anomaly Detection Benchmark
Benchmark for unsupervised learning algorithms applied to streaming anomaly detection.

This repository contains a collection of unsupervised anomaly detection algorithms, which are useful for identifying anomalous behavior in large datasets.

The goal is to provide a benchmark of these algorithms on various streaming datasets, allowing to easily compare their performance.

## Installation

Python requirement for this project: 
`python_version <= 3.9`
  
`git clone https://github.com/ledio7/ADBench.git`

`cd ADBench`

`python3 -m venv bench`  &nbsp;&nbsp;&nbsp; *to create the virtual environment for the first time*

`source bench/bin/activate` &nbsp;&nbsp;&nbsp;*For mac0S/Linux*  &nbsp;&nbsp; or &nbsp;&nbsp;   `bench\Scripts\activate` &nbsp;&nbsp;&nbsp;*For Windows*

`pip install -r requirements.txt`
  
  
## How to obtain the results

### Main Results

`cd Experiments`

`python run_bench.py`

The results will be stored in the *./Experiments/Results* folder and they include: 

  * CSV file with metric values for all the runs
  
  * CSV file with aggregated metric values 
  
  * PR Curve and ROC Curve for each dataset (in *Main_Plots* folder)
  
  * Heatmaps (in *Main_Plots* folder)

Since there are large datasets and multiple runs for each dataset, the overall run takes several hours; if the user needs faster results, he/she can specify a smaller number of samples to process:

`python run_bench.py *number_of desired_samples*`

Bear in mind that if the number of desired samples is too small, some metric values will show 0 as result because there were not sufficient samples.

### Additional Results

In order to have further details about the results, the user can run: 

`python additional_plots.py`

which produces additional plots about each specific metric in the *additional_plots* folder for each dataset. 

### Statistical Test

In order to perform the Nemenyi Test and compare the algorithms (after the runs) on a specific metric:

`python statistic_test.py *desired_metric*`

where the metric must be chosen from one of these: *PR-AUC, ROC-AUC, F1_O, F1_1, Recall_0, Recall_1, Geo-Mean* (default: *PR-AUC*).

### Synthetic Experiments

There is also an additional folder called _Synthetic_ which shows the results and the performances of the models applied to synthetic data streams:

`cd Synthetic`

`python run_bench.py`

This experiment is way heavier than the previous one on real dataset because it has larger (synthetic) datasets.


### Credits

https://github.com/lucasczz/DAADS for the code implementation of the algorithms used in this project.
