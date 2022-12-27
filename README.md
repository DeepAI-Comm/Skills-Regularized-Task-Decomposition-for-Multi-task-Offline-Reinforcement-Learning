# Skills Regularized Task Decomposition for Multi-task Offline Reinforcement Learning

## Download offline datasets

1. Download offline_data.zip
https://drive.google.com/file/d/12gKAtvSr9XOLvHSiZSsX5BsqASg-Q1L4/view - No Airsim data
2. unzip offline_data.zip in single_task directory

## Run TD3+BC
```
python TID/TaskID_policy.py --mode replay
```

## Run Soft modularization
```
python TID/TaskID_policy.py --mode replay --algo SoftModule
```

## Run PCGrad
```
python TID/TaskID_policy.py --mode replay --algo PCGrad
```

## Run SRTD
```python
python SRTD/embeddings_train.py --mode replay
```

## Run SRTD
```python
python SRTD/embeddings_train.py --mode replay --data-augmentation
```