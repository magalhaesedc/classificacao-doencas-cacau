#https://www.machinecurve.com/index.php/2020/02/18/how-to-use-k-fold-cross-validation-with-keras/

from keras.utils import np_utils
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from LoadData import loadData
import numpy as np
import Functions
import time
from keras.preprocessing.image import ImageDataGenerator

inicio = time.time()

# Model configuration
batch_size = 32
epochs = 100
num_folds = 10

#Carrega o BD das imagens (Treino e Validação) 
inicio_aux = time.time()
(X, y) = loadData()
fim = time.time()
Functions.printTime("Load Dataset Treino", inicio_aux, fim)

# Parse numbers as floats
X = X.astype('float32')

# Normalize data
X = X / 255

#Transformando os rótulos de decimal para vetores com valores binários
y = np_utils.to_categorical(y)

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []


# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
acc_train = []
acc_test = []
fold_no = 1
for train, test in kfold.split(X, y):
    
    inicio_aux = time.time()
    
    gerador_treinamento = ImageDataGenerator(rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
    gerador_validacao = ImageDataGenerator()
    
    base_treinamento = gerador_treinamento.flow(X[train],y[train], batch_size = batch_size)
    
    base_validacao = gerador_validacao.flow(X[test], y[test], batch_size = batch_size)
    
	#Criação do Modelo
    model = Sequential()
    model.add(Convolution2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation="sigmoid"))
    
	# Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

	# Fit data to model
    history = model.fit(base_treinamento, epochs=epochs, steps_per_epoch=X[train].shape[0] // batch_size, validation_data=base_validacao, validation_steps = X[test].shape[0] // batch_size)
    
    acc_train.append(history.history['accuracy'])
    acc_test.append(history.history['val_accuracy'])
    
	# Generate generalization metrics
    scores = model.evaluate(X[test], y[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    
    fim = time.time()
    Functions.printTime(f"Tempo de Treinamento - Fold {fold_no}", inicio_aux, fim)
    
    # Salva o modelo no formato JSON
    model_json = model.to_json()
    with open(f"model_fold{fold_no}.json", "w") as json_file:
        json_file.write(model_json)
    
    # Salva os pesos em HDF5
    model.save_weights(f"model_w_fold{fold_no}.h5")
    print("Modelo salvo no disco")  
    
    # Increase fold number
    fold_no = fold_no + 1

# Plota o histórico da acurácia - Treino
for i in range(0, len(acc_train)):
    plt.plot(acc_train[i], label = f"Fold {i+1}")
plt.title('Accuracy curve - Train')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_train_CV_DA.png', dpi=600)
plt.show()

# Plota o histórico da acurácia - Test
for i in range(0,len(acc_test)):
    plt.plot(acc_test[i], label = f"Fold {i+1}")
plt.title('Accuracy curve - Test')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.tight_layout()
plt.savefig('accuracy_test_CV_DA.png', dpi=600)
plt.show()

# == Exibe os resulados gerais ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
	print('------------------------------------------------------------------------')
	print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

# Salva os resultados da acurácia em arquivo CSV
folds_index = []
coluns = []
import pandas as pd
for i in range(1, num_folds+1):
    folds_index.append(f'fold{i}')
for i in range(1, epochs+1):
    coluns.append(f'epoca{i}')
result_train = pd.DataFrame(acc_train, index=folds_index, columns=coluns)
result_test = pd.DataFrame(acc_test, index=folds_index, columns=coluns)
result_train.to_csv('accuracy_train.csv', header=True, index=True)
result_test.to_csv('accuracy_test.csv', header=True, index=True)

fim = time.time()
Functions.printTime("Time Run Model", inicio, fim)