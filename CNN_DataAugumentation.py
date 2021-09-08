from LoadData import loadData, loadDataTest
import Functions
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import time

inicio = time.time()

# Configuração do Modelo
batch_size = 50
epochs = 100

#Carrega o BD das imagens (Treino e Validação) 
inicio_aux = time.time()
(X, y) = loadData()
fim = time.time()
Functions.printTime("Load Dataset Treino", inicio_aux, fim)

#Carrega o BD das imgagens (Teste)
inicio_aux = time.time()
(X_test, y_test) = loadDataTest()
fim = time.time()
Functions.printTime("Load Dataset Test", inicio_aux, fim)

#Redimensiona os dados para ficar no formato que o tensorflow trabalha
X = X.astype('float32') 
X_test = X_test.astype('float32')

#Normalizando os valores de 0-255 to 0.0-1.0
X /= 255.0
X_test /= 255.0
# print(X.shape)
# print(X_test.shape)

#Divide os dados em 82% para treino e 18% para teste **Os dados de testes já foram separados
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.18, train_size=0.82, stratify=y)

#Transformando os rótulos de decimal para vetores com valores binários
y_train = np_utils.to_categorical(y_train)
y_val = np_utils.to_categorical(y_val)
y_test = np_utils.to_categorical(y_test)

# Criação do Modelo
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
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

generator_train = ImageDataGenerator(rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
generator_val = ImageDataGenerator()

base_train = generator_train.flow(X_train, y_train, batch_size = batch_size)

base_val = generator_val.flow(X_val, y_val, batch_size = batch_size)

# Executa treinamento do modelo
inicio_aux = time.time()
history = model.fit(base_train, epochs=epochs, steps_per_epoch=X_train.shape[0] // batch_size, validation_data=base_val, validation_steps = X_val.shape[0] // batch_size)
fim = time.time()
Functions.printTime("Training", inicio_aux, fim)

#print(len(base_train))

# Plota o histórico da acurácia 
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig('curve_accuracy.png', dpi=300)
plt.show()

# Mostra a potuação da acurácia
scores = model.evaluate(X_test, y_test, verbose=0)
result_error = str("%.2f"%(1-scores[1]))
result = str("%.2f"%(scores[1]))
print("CNN Score:", result)
print("CNN Error:", result_error)

# Salva o modelo no formato JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Salva os pesos em HDF5
model.save_weights("model_w.h5")
print("Modelo salvo no disco")

# Salva os resultados da acurácia em arquivo CSV
index = []
for i in range(1, epochs+1):
    index.append(f'epoca{i}')
result_train = pd.DataFrame(history.history['accuracy'], index=index)
result_test = pd.DataFrame(history.history['val_accuracy'], index=index)
result_train.to_csv('accuracy_train.csv', header=False)
result_test.to_csv('accuracy_test.csv', header=False)

fim = time.time()
Functions.printTime("Time Run Model", inicio, fim)