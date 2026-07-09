# imports mais comuns que poderão ser necessários
import tensorflow as tf
import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# constantes - dimensão das imagens
IMG_HEIGHT = 28
IMG_WIDTH = 28

# constantes - labels/classes
LABELS = ["T-Shirt/Top",
          "Trouser",
          "Pullover",
          "Dress",
          "Coat",
          "Sandal",
          "Shirt",
          "Sneaker",
          "Bag",
          "Boot"]
N_CLASSES = 10

print("IMG_HEIGHT: " + str(IMG_HEIGHT))
print("IMG_WIDTH: " + str(IMG_WIDTH))
print("N_CLASSES: " + str(N_CLASSES) + "\n")

# callbacks
BEST_MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(
    filepath="tmp/best_model.weights.h5",      # ficheiro para os pesos do "melhor modelo"
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5)

# carregar o dataset FASHION_MNIST
dataset = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = dataset.load_data()

# normalização
x_train = x_train / 255.0
x_test = x_test / 255.0

print("Número de amostras no training set original: " + str(x_train.shape[0]))
print("Número de amostras no test set original: " + str(x_test.shape[0]))
print("Não esquecer que se pretende também gerar um validation set!")

# transformar vetor das labels numa matriz - adequado para a classificação
# multiclasse da Parte 1, mas não para a classificação binária da Parte 2
y_train = keras.utils.to_categorical(y_train,N_CLASSES)
y_test = keras.utils.to_categorical(y_test,N_CLASSES)

"""Continuar a partir da daqui. 

Será útil consultar os exemplos da aula... 

Resumo das tarefas:
   
   Parte 1
   a)	Obter um conjunto de validação;
   b)	Construir o modelo;
   c)	Compilar a rede;
   d)	Treinar o modelo – max 50 épocas, de pref. a usar callbacks;
   e)	Gráfico que mostre a evolução do treino;
   f)	Cálculo dos acertos no conjunto de teste;
   g)	Mostrar a matriz de confusão.
   
   Parte 2
   Construir nova rede e adaptar para classificação binária Vestuário / Calçado e Malas
   
   Parte 3
   Comparar classificações de ambos os modelos anteriores 
   (i.e., predições multi-classe binarizadas vs. predições binárias diretas)   

   BOM TRABALHO!!!"""
