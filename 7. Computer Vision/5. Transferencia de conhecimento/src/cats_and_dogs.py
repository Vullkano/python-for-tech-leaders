import tensorflow as tf
import keras
from keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import cv2

import seaborn as sns
import sys
import os
from pathlib import Path

# Configurar a seed do TensorFlow
tf.keras.utils.set_random_seed(777)
# Definir a seed global para a operação de GPU
tf.config.experimental.enable_op_determinism()

# Tamanho das imagens
img_height = 256
img_width = 256

# Localização dos dados
path = Path(r"../datasets/cats_and_dogs/")

# Carregamento dos dados
train = keras.utils.image_dataset_from_directory(
    path / "train",
    label_mode="binary",
    class_names=["cats", "dogs"],
    image_size=(img_height, img_width),
)
validation, test = keras.utils.image_dataset_from_directory(
    path / "validation",
    label_mode="binary",
    class_names=["cats", "dogs"],
    image_size=(img_height, img_width),
    seed=777,
    validation_split=.5,
    subset="both",
)
train = train.cache()
validation = validation.cache()
test = test.cache()
# Labels
labels = ["cat", "dog"]

# Distribuição de classes
train_labels = np.concatenate([y for x, y in train], axis=0) 
validation_labels = np.concatenate([y for x, y in validation], axis=0) 
test_labels = np.concatenate([y for x, y in test], axis=0) 
print(f"Train: {len(train_labels)} samples, {sum(train_labels)[0]/len(train_labels):.2%} dogs\nValidation: {len(validation_labels)} samples, {sum(validation_labels)[0]/len(validation_labels):.2%} dogs\nTest: {len(test_labels)} samples, {sum(test_labels)[0]/len(test_labels):.2%} dogs")