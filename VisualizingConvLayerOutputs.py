import matplotlib.pyplot as plt
from keras.models import model_from_json
from keras.models import Model
from LoadData import loadDataUnit

json_file = open("model.json", "r")
load_model_json = json_file.read()
json_file.close()

#load model
model = model_from_json(load_model_json)

#load weights into model
model.load_weights("model_w.h5")    

#compile model and evaluate
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

print(len(model.layers))

#Define a new truncated model to only include the conv layers of interest
#conv_layer_index = [1, 3, 6, 8, 11, 13, 15] #TO define a shorter model
conv_layer_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  #TO define a shorter model
outputs = [model.layers[i].output for i in conv_layer_index]

# Imprimir nomes das camadas
i = 0
for out in outputs:
    print(type(out))
    print("\n",i)
    print(out)
    i+=1

model_short = Model(inputs=model.inputs, outputs=outputs)

print(model_short.summary())

#load dataset
X, y = loadDataUnit()

#normalize dataset from 0-255 to 0.0-1.0
X = X.astype("float32")
X /= 255.0

plt.imshow(X[0])
plt.savefig('input.png', dpi=200)
plt.show()

# Generate feature output by predicting on the input image
feature_output = model_short.predict(X)

# Extrair imagens individuais
camada = 7
imagem = 22
plt.imshow(feature_output[camada][0, :, :, imagem-1])
plt.savefig(f'{camada}.png', dpi=200)
plt.show()

# Extrair Todas as imagens
columns = 8
rows = 4
for ftr in feature_output:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig = plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        fig.set_xlabel(i)
        plt.imshow(ftr[0, :, :, i-1])
        #pos += 1
    plt.show()