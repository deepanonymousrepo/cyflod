# CYFLOD: Cyclic Filtering and Loss Damping for Combating Noisy Labels in Fine-grained Visual Classification
# 1.  Challenges in LNL
![image](https://github.com/user-attachments/assets/21160c7c-c904-42db-9b22-7aebe21102e2)
Fig.1 LNL Challenges. Top: random noise due to mislabeled classes (blue rectangles) in the generic CIFAR-10 (left) and the fine-grained Stanford Cars (right) data sets. Bottom: dependent noise due to confusion of classes like ``deer'' and ``dog'' (red rectangles) for the generic CIFAR-10 and inter-class overlap between similar vehicles in the fine-grained Standford Cars (right) data sets.
# 2. CYFLOD Overview
![image](https://github.com/user-attachments/assets/02fc95f0-ee93-4cc5-a51a-7a361f499e9f)
Fig.2 CYFLOD Training Overview: he proposed training scheme starts with the full, noisy data set. We feed the data to the model and
train the model using transfer learning: a cyclic cleansing process combined with a loss damping iteratively removes the noisy samples.
| ![Image 1](./figs/loss-damper.png) | ![Image 2](./figs/cross-entropy-damped.png) | ![Image 3](./figs/gce-damped.png) |
|------------------------------|------------------------------|------------------------------|

Fig.3 Loss damping. Left: examples of loss damping functions with different $\delta$ values. Right: effect of loss damping on cross-entropy loss.
# 3. Results
| ![Image 1](./figs/food-101-radar.png) | ![Image 2](./figs/sym_cars_aircrafts_baseline_vs_cyflod.png) | ![Image 3](./figs/asym_cars_aircrafts_baseline_vs_cyflod.png) |
|------------------------------|------------------------------|------------------------------|



# 3.1 Results on two Fine-grained Datasets
We conducted extensive experiments on two fine-grained datasets, i.e., Stanford Cars and Aircraft, with a # Symmetric noise ratio of {20%. 40% }and Asymmetric noise ratios of {10%, 30%}.

### Symmetric Noise

| **Dataset**      | **Method**                                                   | **Sym. (20%)** | **Sym. (40%)** |
|-------------------|--------------------------------------------------------------|----------------|----------------|
| **Stanford Cars** | CE+SNSCL                                                    | 83.24          | 76.72          |
|                   | GCE+SNSCL                                                   | 73.78          | 58.11          |
|                   | SCE+SNSCL                                                   | 82.58          | 79.07          |
|                   | DivideMix+SNSCL                                             | 86.29          | 80.09          |
|                   | ![CYFLOD](https://img.shields.io/badge/CYFLOD-blue) (RCE + δ = 0.5) | **90.46**       | **84.79**       |
| **Aircraft**      | CE+SNSCL                                                    | 76.45          | 70.48          |
|                   | GCE+SNSCL                                                   | 72.67          | 64.70          |
|                   | SYM+SNSCL                                                   | 79.64          | 74.02          |
|                   | DivideMix+SNSCL                                             | 82.31          | 80.11          |
|                   | ![CYFLOD](https://img.shields.io/badge/CYFLOD-blue) (GCE + δ = 0.25) | **88.74**       | **83.19**       |

### Asymmetric Noise

| **Dataset**      | **Method**                                                   | **Asym. (10%)** | **Asym. (30%)** |
|-------------------|--------------------------------------------------------------|-----------------|-----------------|
| **Stanford Cars** | CE+SNSCL                                                    | 83.73           | 70.04           |
|                   | GCE+SNSCL                                                   | 80.33           | 64.64           |
|                   | SYM+SNSCL                                                   | 83.84           | 74.45           |
|                   | DivideMix+SNSCL                                             | 88.18           | 81.20           |
|                   | ![CYFLOD](https://img.shields.io/badge/CYFLOD-blue) (RCE + δ = 0.5) | **90.28**        | **82.35**        |
| **Aircraft**      | CE+SNSCL                                                    | 78.28           | 65.44           |
|                   | GCE+SNSCL                                                   | 73.85           | 63.34           |
|                   | SYM+SNSCL                                                   | 78.34           | 71.13           |
|                   | DivideMix+SNSCL                                             | 84.17           | 80.55           |
|                   | ![CYFLOD](https://img.shields.io/badge/CYFLOD-blue) (GCE + δ = 0.25) | **88.32**        | **76.50**        |

# 3.2 Food-101
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/ab815fff-9634-43b9-b4a9-3ee386b9dd36)
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/b97981e3-9d66-4bc6-830b-0721f527dad7)

# 4.3 CIFAR-10
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/cd10bf16-a2b6-4887-9ad6-5c33ef22e592)

# 5. How to Run the code ....
PyTorch Implementation of CYFLOD

# 5.1. Environment settings

Python 3.8, Pytorch 1.11, CUDA 11.1, Torchvision 12.2

# 5.2. Dataset

   Data considered in this study, you can download the dataset from original source as provided below.

   Stanford Car: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
   
   Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
   
   CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
   
   Food-101: https://vision.ee.ethz.ch/datasets_extra/food-101/

# 6. Training
To generate noise:

python noise_generator.py --datasetpath {car, aircraft, cifar10, Food-101} --noiseratio {0.1,0.2,0.3, ..., 0.80} --noisetype {sym,asym}

Different losses implementation with Dump is given  in dumped-losses.py

You can import the losses file in the main notebook, run and enjoy......


Cite

On publish ....
