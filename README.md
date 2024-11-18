# CYFLOD: Cyclic Filtering and Loss Damping for Combating Noisy Labels in Fine-grained Visual Classification
# 1.  Challenges in LNL
![image](https://github.com/user-attachments/assets/21160c7c-c904-42db-9b22-7aebe21102e2)
Fig.1 LNL Challenges. Top: random noise due to mislabeled classes (blue rectangles) in the generic CIFAR-10 (left) and the fine-grained Stanford Cars (right) data sets. Bottom: dependent noise due to confusion of classes like ``deer'' and ``dog'' (red rectangles) for the generic CIFAR-10 and inter-class overlap between similar vehicles in the fine-grained Standford Cars (right) data sets.
# 2. CYFLOD Overview
![image](https://github.com/user-attachments/assets/02fc95f0-ee93-4cc5-a51a-7a361f499e9f)
Fig.2 CYFLOD Training Overview: he proposed training scheme starts with the full, noisy data set. We feed the data to the model and
train the model using transfer learning: a cyclic cleansing process combined with a loss damping iteratively removes the noisy samples.
| ![Image 1](./fig/loss-damper.png) | ![Image 2](./fig/cross-entropy-damped.png) | ![Image 3](./fig/gce-damped.png) |
|------------------------------|------------------------------|------------------------------|



  <img src="![image](https://github.com/user-attachments/assets/34b787c6-f8f1-4b88-a7ae-a5b4ce1db013)" alt="Image 1" width="200"/>
  <img src="![image](https://github.com/user-attachments/assets/34b787c6-f8f1-4b88-a7ae-a5b4ce1db013)" alt="Image 2" width="200"/>
  <img src="p![image](https://github.com/user-attachments/assets/34b787c6-f8f1-4b88-a7ae-a5b4ce1db013)" alt="Image 3" width="200"/>


Fig.3 Loss damping. Left: examples of loss damping functions with different $\delta$ values. Right: effect of loss damping on cross-entropy loss.
# 3. Results

![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/677e3b0e-3288-414a-a176-6753393f219e)

# 3.1 Results on two Fine-grained Datasets
We conducted extensive experiments on two fine-grained datasets, i.e., Stanford Cars and Aircraft, with a # Symmetric noise ratio of {20%. 40% }and Asymmetric noise ratios of {10%, 30%}.
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/a118c600-06b5-4428-bc71-60ba202ff2ff)(https://github.com/GilalNauman/CYFLOD/assets/62802429/411c86c6-7f4c-4acd-b7a2-0ad4452849e0)
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
