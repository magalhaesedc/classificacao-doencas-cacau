import LoadData
import ConfusionMatrix
import matplotlib.pyplot as plt
from keras.models import model_from_json
#from sklearn.preprocessing import LabelEncoder
#from keras.utils import np_utils
import numpy as np
from sklearn.metrics import confusion_matrix

#read file
json_file = open("model.json", "r")
load_model_json = json_file.read()
json_file.close()

#load model
model = model_from_json(load_model_json)

#load weights into model
model.load_weights("model_w.h5")    

#compile model and evaluate
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])

#load dataset
X, y = LoadData.loadDataTest()

#normalize dataset from 0-255 to 0.0-1.0
X = X.astype("float32")
X /= 255.0

yp = model.predict_classes(X, verbose=0)
yp = yp.reshape(len(yp), 1)

print(yp.shape)
print(y.shape)
print("Acertos:", sum(y==yp)/len(y))
print("Erros: ", sum(y!=yp)/len(y))

np.set_printoptions(precision=2)
class_names = ["Saudável","Podridão Parda", "Broca da Vagem"]
confusionMatrix = confusion_matrix(y, yp)
plt.figure()
ConfusionMatrix.plot_confusion_matrix(confusionMatrix, classes=class_names, title='Matriz de Confusão')
plt.tight_layout()
plt.savefig('matriz_de_confusao.png', dpi=300)
plt.show()