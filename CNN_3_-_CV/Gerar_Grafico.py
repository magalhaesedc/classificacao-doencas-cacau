import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('resultado_cross-validation_train.csv')

K = 10

X = np.arange(1, K+1)
y = []

for l in range(len(df_train.index)):
    linha = df_train.loc[l,]
    linha = linha[1:]
    linha.astype('float32')
    y.append(linha.sum()/(len(df_train.columns)-1))
    
plt.plot(X, y, marker='o', linestyle='--', color='red')
plt.title('Precisão média para diferentes valores de k - Treino')
plt.xlabel('K')
plt.ylabel('Acurácia média')
plt.savefig('precisao_media_valores_de_k_-_treino.png', dpi=600)
plt.show()