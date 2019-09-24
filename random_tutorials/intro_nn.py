
import pandas as pd
import numpy as np
import itertools

from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, BatchNormalization, MaxPool2D

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import RMSprop

import matplotlib.pyplot as plt
import matplotlib.image as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed()
sns.set(style='white', context='notebook', palette='deep')

dataset = pd.read_csv('./train.csv')
competition_dataset = pd.read_csv('./test.csv')

label = dataset['label']
feature = dataset.drop(labels=['label'], axis=1)

sns.countplot(label)

print("done")
