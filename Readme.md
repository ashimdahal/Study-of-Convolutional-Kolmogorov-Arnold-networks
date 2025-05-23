# Whats this?
AlexNET on Imagenet with Convolutional Kolmogorov Arnold Networks instaed of CNNs. :)

We train and test CKANs against 3 dataset and with 3 models on comparable grounds: AlexNet style architecture for ImageNet, LeNet style architecture for MNIST and Tabular CNN style architecture for MoA.

The results are presented in paper: [arxiv.](https://arxiv.org/abs/2501.15757)

The file structure in related to the paper is the following; some results from albation study were not reported in the paper.

```
├── graphs
│   ├── loss_comparision.png #not reported
│   ├── loss LeNet KAN.png #not reported
│   ├── loss LeNet.png #not reported
│   ├── model_comparison_radar_plots.png
│   ├── radar_plot.png
│   ├── validation_accuracy.png #not reported
│   ├── validation LeNet KAN.png #not reported
│   └── validation LeNet.png #not reported
├── hpcalexnet.sh #contains the hpc settings to train AlexNet KAN
├── hpclenet.sh #contains the hpc settings to train LeNet and LeNet KAN
├── hpctest.sh #contains the hpc settings to test metrics of either models
├── LICENSE
├── logs
│   ├── errorfastalexnetkan.log # reported
│   ├── errorfastlenetkan.log # reported
│   ├── errorlenetcnn.log # reported
│   ├── errortestalexkan.log #Not reported
│   ├── moatabularckan.log # reported
│   ├── moatabularcnn.log # reported
│   ├── outputcnn.log #reported
│   ├── outputfastalexnetkan.log #not reported --> this is albation study that didn't work out
│   ├── outputfastlenetkan.log # reported
│   ├── outputkan.log #NR
│   ├── outputlenetcnn.log #Reported
│   ├── outputtestalexkan.log #reported 
│   └── outputtestalexnetkanvalexnetcnn.log #reported
├── moa-second-price-winner-vs-convkan.ipynb #code for testing both kan vs cnn implementaion on tabular dataset
├── Readme.md # this file
├── reports #contains classification report for all reported data
│   ├── alexnetkan report.csv
│   ├── AlexNet KAN report.csv
│   ├── LeNet KAN report.csv
│   ├── LeNet report.csv
│   └── PyTorch AlexNet report.csv
└── scripts # not reported: alexnetfastkan (albation study that didn't work out)
    ├── alexnetcnn.py
    ├── alexnetfastkan.py
    ├── alexnetkan.py
    ├── download_alexnet.py
    ├── evaluate_moa_logs.py
    ├── lenetcnn.py
    ├── lenetfastkan.py
    ├── lenetkan.py
    ├── __pycache__
    │   ├── alexnetkan.cpython-38.pyc
    │   └── lenetfastkan.cpython-38.pyc
    ├── radar_plot.py
    └── test_models.py
```
