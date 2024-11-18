# CYFLOD: Cyclic Filtering and Loss Damping for Combating Noisy Labels in Fine-grained Visual Classification
# 1.  Challenges in LNL
![image](https://github.com/user-attachments/assets/766655ce-f586-42ad-843c-abfa233ef62a)
Fig.1 Top: blue rectangles highlight random noise caused by mislabeled classes on the generic CIFAR-10 dataset (left) and on the fine-grained Stanford Cars dataset (right). Bottom: red rectangles highlight dependent noise arising from the confusion and similarity between
classes like "deer" and "dog" for the generic CIFAR-10 dataset (left), and the inter-class overlap between closely related vehicles in the fine-grained Stanford Cars dataset (right). 
# 2. CYFLOD Overview
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/4235f374-cdb6-4ea4-96f9-af87a753fd8f)
Fig.2 CYFLOD Training Overview: The proposed training scheme starts from the full noisy dataset. We feed the data to the model and train the model using transfer learning: a cyclic cleansing process integrated with a loss damping iteratively removes the noisy samples.


![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/4b928e2d-07ed-4325-8e2e-39655081b949) ![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/c7754140-d1dc-411c-821e-2f868f9acb6e)
Fig.3 Loss damping. Left: examples of loss damping functions with different $\delta$ values. Right: effect of loss damping on cross-entropy loss.
# 3. Results
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/677e3b0e-3288-414a-a176-6753393f219e)

# 3.1 Results on two Fine-grained Datasets
We conducted extensive experiments on two fine-grained datasets, i.e., Stanford Cars and Aircraft, with a # Symmetric noise ratio of {20%. 40% }and Asymmetric noise ratios of {10%, 30%}.
![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/a118c600-06b5-4428-bc71-60ba202ff2ff) ![image](https://github.com/GilalNauman/CYFLOD/assets/62802429/411c86c6-7f4c-4acd-b7a2-0ad4452849e0)
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
