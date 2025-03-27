# STOP
This repository contains the implementation for the paper: "Unraveling Spatial-Temporal and Out-of-Distribution Patterns for Multivariate
Time Series Classification", accepted by WWW 2025.

## Requirements
The dependencies can be installed by: 

```pip install -r requirements.txt```

## Data
UEA datasets can be downloaded in this [link](http://www.timeseriesclassification.com/). And the unzip file should be put into `datasets/`, where the original data can be located by `datasets/Multivariate_arff`.
Then you can run this command to preprocess the data.
```python preprocess.py```

## Usage
To train and evaluate STOP on a dataset, run the following command:
```python -u train_mul_ood.py --data <dataset_name> --seg_len <segment length> --r1 <r1> --r2 <r2>```

The detailed descriptions about the arguments are as following:


| Parameter name | Description of parameter                                                  |
|----------------|---------------------------------------------------------------------------|
| data           | The name of dataset.                                                      |
| seg_len        | The segment length.                                                       |
| r1             | the ratio of nodes to choose as neighbors in intra-correlation extraction |
| r2             | the ratio of nodes to choose as neighbors in inter-correlation extraction |                                                                    |

(For descriptions of all arguments, run `python train_mul_ood.py -h`.)

For example, dataset Cricket can be directly trained by the following command:
```python -u train_mul_ood.py --data Cricket --seg_len 20```

The results will be saved at ```result.txt```. 



Script: the script for reproduction is provided in ```UEA.sh```
