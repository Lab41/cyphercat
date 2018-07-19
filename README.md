# CypherCat

Here are tools and software you can use to replicate our work.

## Research

We are focusing on two different areas of research, aiming to contribute to the CleverHans repository. These areas are:
- *Model inversion* attack is the process of either the model parameters or the data used to train the data.
- *Model Fooling/Evasion* attacks are related to tricking a model into performing considerably worse on basic examples

The below is created by our visualization software. The actual PDF has links to the arxiv papers. For inverting neural networks, the following wording is of relevance:
[![Model Inversion](Visualizations/Example-Invert.png)](Visualizations/inversion_defense.gv.pdf)
For fooling neural networks, this is the following papers and relevant work:
[![Model Fooling](Visualizations/Example-Fooling.png)](Visualizations/inversion_attack.gv.pdf)

## Environment and Software

### Setup 
```console
$ git clone https://github.com/Lab41/cyphercat.git
$ cd cyphercat
$ virtualenv cyphercat_virtualenv
$ source cyphercat_virtualenv/bin/activate
$ pip install -r requirements.txt
$ ipython kernel install --user --name=cyphercat_virtualenv
```
Select `cyphercat_virtualenv` kernel when running Jupyter.  

### Structure
[src_code/models.py](src_code/models.py): Pytorch model classes.    

[src_code/train.py](src_code/train.py): Generic training method for a classifier. Also contains training method for ML-Leaks attack[1].   

[src_code/metrics.py](src_code/metrics.py): Functions to calculate classifier accuracy and membership inference accuracy.   

[src_code/data_downloaders.py](src_code/data_downloaders.py): Helper functions to download and preprocess datasets.   

[baselines/](baselines/): Various Jupyter notebooks containing baselines for popular datasets.   

[ml_leaks/](ml_leaks/): Implementations of adversary 1 and 3 from ML-Leaks[1]. 

### Visualization

We are using [GraphViz](https://www.graphviz.org/) for our research in order to get a handle on the papers in the space, as well as describe our research. You can view some of that here. To install visualization tools via Mac, use:

```
brew install graphviz
pip install graphviz
```

## References 
1. Salem, Ahmed, et al. "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models." arXiv preprint arXiv:1806.01246 (2018). [Link](https://arxiv.org/abs/1806.01246)  

