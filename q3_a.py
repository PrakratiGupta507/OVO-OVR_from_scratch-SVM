# -*- coding: utf-8 -*-
"""Q3_a.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rpGI4jgQRprFLXIr5bkm5ns71x3ort7f
"""

from google.colab import drive
drive.mount('/content/drive')

import scipy.io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import data
dataset = scipy.io.loadmat('/content/drive/My Drive/ML assignment 3/dataset_b.mat')

#preprocessing on data
samples, label = dataset['samples'], dataset['labels']
df = pd.DataFrame(list(samples))
label = pd.DataFrame(label[0])
df['label'] = label

#make data for plot
x=df[0]
y=df[1]
z=df['label']
data = pd.DataFrame({"X_Value": x, "Y_Value": y, "Category": z})

#visualization of data through scatter plot
plt.figure(figsize=(15,10))
sns.scatterplot(data=data, x="X_Value", y="Y_Value", hue="Category",palette="bright")

