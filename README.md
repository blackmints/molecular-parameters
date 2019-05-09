# Deep Learning Algorithm of Graph Convolutional Network: A Case of Aqueous Solubility Problems

This is a implementation of our paper "Deep Learning Algorithm of Graph Convolutional Network: A Case of Aqueous Solubility Problems":

Hyeoncheol Cho, Insung S. Choi, [Deep Learning Algorithm of Graph Convolutional Network: A Case of Aqueous Solubility Problems](https://onlinelibrary.wiley.com/doi/full/10.1002/bkcs.11730)

Our paper evaluates GCN model with influence of atomic input features

## Requirements

* Tensorflow
* Keras
* RDKit

## Datasets
* FreeSolv
* ESOL (= delaney)

## Experiments

Each one from the 12 atom features is deleted and trained to evaluate the difference between feature‐deleted model and the original one for understanding of the contribution of individual atom features to the aqueous solubility.

## Cite

If you use our model in your research, please cite:
```
@article{cho2019deep,
  title={Deep Learning Algorithm of Graph Convolutional Network: A Case of Aqueous Solubility Problems},
  author={Cho, Hyeoncheol and Choi, Insung},
  journal={Bulletin of the Korean Chemical Society},
  year={2019}
}
```
