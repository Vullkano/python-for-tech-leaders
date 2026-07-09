import keras
from keras import layers

import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

###################################################################

# callbacks para guardar modelo com menor loss e
# para terminar treino se não houver melhorias na loss

BEST_MODEL_PATH = "tmp/best_model.weights.h5"

BEST_MODEL_CHECKPOINT = keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_PATH,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

EARLY_STOPPING = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Constantes
BATCH_SIZE = 64
IMG_HEIGHT = 160
IMG_WIDTH = 160

DATASET_PATH = "../datasets/flower_photos"  # ajustar consoante a localização
SEED = 1245                                 # semente para o split treino/validação
TRAIN_VAL_SPLIT = 0.2                       # fração de imagens para validação
NUM_CLASSES = 5

# Neste exemplo o dataset é carregado a partir do sistema de ficheiros
# e apenas é dividido em treino e validação - não se definiu o conjunto de teste

train_ds, val_ds = keras.utils.image_dataset_from_directory(
  DATASET_PATH,
  labels='inferred',
  label_mode='categorical',
  validation_split=TRAIN_VAL_SPLIT,
  subset="both",
  seed=SEED,
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

# labels inferidas a partir dos nomes dos diretorios
labels = train_ds.class_names
print(labels)

# colocar os datasets em memória - uma vez carregados a ordem dos batches já não muda
train_ds = train_ds.cache()
val_ds = val_ds.cache()

# arquitetura da CNN
model = keras.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=4),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

# Treino
EPOCHS = 30
history = model.fit(train_ds,
                    epochs=EPOCHS,
                    validation_data=val_ds,
                    callbacks=[EARLY_STOPPING, BEST_MODEL_CHECKPOINT])

# carregar o melhor modelo encontrado durante o treino
model.load_weights(BEST_MODEL_PATH)

# obter as predições e ground thruth num formato mais fácil de tratar
# (um vetor de ids das classes)

# realizar as predições
y_pred = model.predict(val_ds)
pred_ids = np.argmax(y_pred, axis=1)

# concatena os "targets" do conjunto de validação (pois estavam organizados em batches)
y_true = np.concat([y for x, y in val_ds], axis=0)
true_ids = np.argmax(y_true, axis=1)

# calcular a taxa de acertos
n_misses = np.count_nonzero(true_ids != pred_ids)
n_preds = pred_ids.shape[0]
accuracy = (n_preds - n_misses) / n_preds

print("Resultados do modelo com loss mais baixa")
print("Falhou em {:d} amostras num total de {:d} imagens de flores".format(n_misses, n_preds))
print("Taxa de acertos: {:.2f} %".format(accuracy * 100))

# gerar gráficos do treino e matriz de confusão
cm = confusion_matrix(true_ids, pred_ids)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

# evolução da loss e dos acertos
plt.figure(2, figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylim(0, 5)
plt.title('Training and Validation Loss')

# matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Greens', xticks_rotation=30)
plt.show()