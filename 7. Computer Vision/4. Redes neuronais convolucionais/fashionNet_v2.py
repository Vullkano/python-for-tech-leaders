import keras
from keras import layers

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

##############################################################################
# Preparação dos dados

# utilizar o dataset FASHION_MNIST builtin do tensorflow/keras
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# normalização dos valores de pixel para o intervalo [0 ... 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# preparar os "targets" da ground truth para o formato adequado, usando 10 classes
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# como não estava definido um conjunto de validação, usaram-se algumas amostras de treino
TRAIN_VAL_SPLIT = 10000
x_val = x_train[:TRAIN_VAL_SPLIT,:]
y_val = y_train[:TRAIN_VAL_SPLIT,:]
x_train = x_train[TRAIN_VAL_SPLIT:,:]
y_train = y_train[TRAIN_VAL_SPLIT:,:]


# constantes - labels, classes e dimensões das imagens
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
IMG_HEIGHT = 28
IMG_WIDTH = 28

# mostrar as dimensões das matrizes para treino e teste
print("Training samples shape:   ", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Val samples shape:        ", x_val.shape)
print("Val labels shape:     ", y_val.shape)
print("Test samples shape:       ", x_test.shape)
print("Test labels shape:    ", y_test.shape)


##############################################################################
# Callbacks

# ficheiro onde serão guardados os pesos do "melhor modelo" - ajustar a gosto
BEST_MODEL_PATH = "tmp/best_model.weights.h5"

BEST_MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5)


##############################################################################
# Definição e treino do modelo
#
# definição do modelo
model = keras.models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax'),
])


# compilação do modelo - escolha do algoritmo de otimização e função de perda
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])
# sumário
model.summary()

# treino
nEpochs = 20
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=nEpochs,
                    validation_data=(x_val, y_val),
                    callbacks=[BEST_MODEL_CHECKPOINT, EARLY_STOPPING])

##############################################################################
# Teste do modelo
#

# restaurar o melhor modelo que foi encontrado
model.load_weights(BEST_MODEL_PATH)

# obter predições no conjunto de teste ground truth
y_pred = model.predict(x_test)
pred_ids = np.argmax(y_pred, axis = 1)
true_ids = np.argmax(y_test, axis = 1)

# calcular acertos no conjunto de teste
n_misses = np.count_nonzero(true_ids != pred_ids)
n_preds = pred_ids.shape[0]
accuracy = (n_preds - n_misses) / n_preds

print("Falhou {:d} de {:d} exemplos".format(n_misses, n_preds))
print("Taxa de acertos: {:.2f} %".format(accuracy * 100))

# gerar uma matriz de confusão
cm = confusion_matrix(true_ids, pred_ids)


########################################################################
# Mostrar figuras

# exemplos de imagens onde falhou
missesIdx = np.flatnonzero(true_ids != pred_ids)
plt.figure(1, figsize=(10, 10))
for i in range(0, 16):
    idx = missesIdx[i]
    image = x_test[idx,:,:] * 255
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(image,cmap="gray")
    plt.title("Pred: " + LABELS[pred_ids[idx]] + "\nTrue: " + LABELS[true_ids[idx]])
    plt.axis("off")
    plt.tight_layout()

# evolução dos acertos
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper left")
plt.grid(True, ls='--')

# evolução da loss
plt.figure(3)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc="upper right")
plt.grid(True, ls='--')

# matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
disp.plot(cmap="Oranges", xticks_rotation=45)
plt.tight_layout()
plt.show()
