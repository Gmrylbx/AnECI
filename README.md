# AnECI 

A PyTorch implementation of "Robust Attributed Network Embedding Preserving Community Information"


## Abstract 
Network embedding, also known as network representation, has attracted a surge of attention in data mining
and machine learning community recently as a fundamental
tool to analyze network data. Most existing deep learning based network embedding approaches focus on reconstructing
the pairwise connections of micro-structure but ignore the
community structure, which are easily disturbed by network
anomaly or attack. Thus, to address the aforementioned challenges simultaneously, we propose a novel robust framework
for attributed network embedding by preserving Community
Information (AnECI). Rather than using pairwise connection based micro-structure, we try to guide the node embedding by
the underlying community structure learned from data itself as
an unsupervised learning, which is expected to own stronger
anti-interference ability. Specially, we put forward with a new
modularity function for high-order proximity and overlapped
community to guide the network embedding of an attributed
graph encoder. We conducted extensive experiments on node
classification, anomaly detection and community detection tasks
on real benchmark data sets, and the results show that AnECI
is superior to the state-of-art attributed network embedding
methods.

## Requirements

```
matplotlib==3.1.1
numpy==1.18.5
torch==1.8.1
scipy==1.5.0
torchvision==0.8.1
networkx==2.6.3
scikit_learn==0.24.0
deeprobust
```

## Cite

