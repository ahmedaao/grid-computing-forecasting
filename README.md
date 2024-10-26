# Grid Computing Forecasting

![Author](https://img.shields.io/badge/Author-Ahmed%20Ait_Ouazzou-brightgreen)
[![GitHub](https://img.shields.io/badge/GitHub-Follow%20Me-lightgrey)](https://github.com/ahmedaao)


## Table of Contents

1. [Introduction](#introduction)
2. [DataSource](#datasource)
3. [Methods](#methods)
    - [Method 1: Benchmark Median](#method-1-benchmark-median)
    - [Method 2: XGBoost](#method-2-xgboost)
    - [Method 3: Grownet](#grownet)
4. [Result](#result)


## Introduction
We have a compute farm composed of several thousand CPUs that can be grouped into CPU clusters. These resources (CPU clusters) are intended to run processes submitted by several clients. One of the immediate questions we face is in what order the processes will be handled. A simple method would be to process them in the order they arrive (FCFS=First Come First Serve). The problem with this method is that long processes will monopolize resources and a bottleneck will form, meaning that many short processes will be queued while waiting for the long process to complete. This will degrade the overall performance of the compute farm. 

To adress this, we will approch the SJN (Shortest Job Next) algorithm, which prioritizes short processes over long ones. This algorithm will maximize job throughput. However, a new problem will emerge. Our system will now constantly receive short processes, consequently risking that long processes may never execute due to their low priority. 

To solve this problem, we will approximate the SJN algorithm(the approximate algorithm of SJN will be called aSJN) to take into account the waiting time of processes. Specifically, the longer a job waits in the queue, the higher its priority will increse so that it can be executed at some point. Here is the formula for aSJN: 

To calculate aSJ, we need to know the execution time of processes. Initially, we will calculate this using the median, which will serve as our benchmark. Then we will try to improve accuracy by applying machine learning models. 

In conclusion, the higher the accuracy for the execution time value, the closer we will be to theoretical SJN, and the better the optimization of our compute farm will be.


## DataSource
Datasource: GWA-T-1 trace in SQLite format [here](http://gwa.ewi.tudelft.nl/datasets/gwa-t-1-das2)


## Methods

### Method 1: Benchmark (Median for the "RunTime" feature)
We calculated the median of the past 'RunTime' for all jobs to make an approximate estimate.

### Method 2: XGBoost Regressor algorithm
XGBoost is an ensemble method that is very popular in ML. We will test it to improve the previous benchmark (simple median)  

### Method 3: Grownet
Grownet is a Neural Networks Gradient Boosting algorithm which uses neural networks as weak learners rather than decision trees. 


## Result
```markdown
| Model      | Benchmark | XGBoost  | Grownet  |
|------------|-----------|----------|----------| 
| Train      | 46.73     | 44.73    | 33.00    |
| Test       | 46.73     | 45.17    | 32.49    |
| Train Time | 5s        | 10min35s | 85min28s |
```
*MAE: Mean Absolute Error