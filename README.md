# SNN_HAR
Pytorch implementation of Spiking Neural Networks for Human Activity Recognition. 
Paper link: [Wearable-based Human Activity Recognition with
Spatio-Temporal Spiking Neural Networks](https://arxiv.org/pdf/2212.02233.pdf).

## Installation and usage

Our implementation is based on Pytorch 1.10 or higher and other libraries. 
Please install the packages in the ``requirements.txt``:
```bash
pip install -r requirements.txt
```

## Datasets download

Please download the dataset and decompress them under the `data` directory before running the code:

- UCIHAR [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- SHAR [link](http://www.sal.disco.unimib.it/technologies/unimib-shar/)
- HHAR [link](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)

## Running ANNs

```bash
python main.py --dataset ucihar --backbone FCN --rep 5
python main.py --dataset shar --backbone FCN --rep 5
python main.py --dataset hhar --backbone FCN --rep 5
```

Here, for ANN baselines, we can change our backbone to FCN (CNN in the paper), DCL, LSTM, Transformer.
`--rep` specifies the number of repeats for computing mean and standard deviation of the accuracies. 


## Running SNNs

```bash
python main.py --dataset ucihar --backbone SFCN --lr 1e-3 --tau 0.75 --thresh 0.5
python main.py --dataset shar --backbone SFCN --lr 1e-3 --tau 0.25 --thresh 0.5
python main.py --dataset hhar --backbone SFCN --lr 1e-3 --tau 0.75 --thresh 0.5
```

For spiking versions of backbone, we offer SFCN adn SDCL.
Note that they need to be trained with slightly larger learning rate.
The `--tau` specifies the decay factor in LIF neurons (Sec 3.3 in paper). 


## Acknowledgement

Our code framework is strongly based on [Tian0426/CL_HAR](https://github.com/Tian0426/CL-HAR).
We'd like to thank the authors of this repo for their efforts. 

If you find our paper interesting, please kindly consider cite the paper. 
```
@article{li2022wearable,
  title={Wearable-based Human Activity Recognition with Spatio-Temporal Spiking Neural Networks},
  author={Li, Yuhang and Yin, Ruokai and Park, Hyoungseob and Kim, Youngeun and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2212.02233},
  year={2022}
}
```




