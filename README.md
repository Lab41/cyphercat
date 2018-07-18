# Cyphercat


## Setup 
```console
$ pip install -r requirements.txt
```


## Structure
[src_code/models.py](src_code/models.py): Pytorch model classes.    

[src_code/train.py](src_code/train.py): Generic training method for a classifier. Also contains training method for ML-Leaks attack[1].   

[src_code/metrics.py](src_code/metrics.py): Functions to calculate classifier accuracy and membership inference accuracy.   

[baselines/](baselines/): Various Jupyter notebooks containing baselines for popular datasets.   

[ml_leaks/](ml_leaks/): Implementations of adversary 1 and 3 from ML-Leaks[1]. 



## References 
1. Salem, Ahmed, et al. "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models." arXiv preprint arXiv:1806.01246 (2018). [Link](https://arxiv.org/abs/1806.01246)  



