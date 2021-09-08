import cv2
import numpy as np
import glob
import imageio
import os

def getImgByClass(path, category):
    files = glob.glob(path)
    # print(type(files))
    # print(files[0])
    X = []
    y = []
    for f in files:
        img = imageio.imread(f)
        resized = cv2.resize(img, (150, 150))
        X.append(resized)
        y.append([category])
    return X, y

def loadData():
    (X1, y1) = getImgByClass(os.getcwd()+"/banco_imagens/treino/saudavel/*.jpg", 0)
    (X2, y2) = getImgByClass(os.getcwd()+"/banco_imagens/treino/podridao_parda/*jpg", 1)
    (X3, y3) = getImgByClass(os.getcwd()+"/banco_imagens/treino/broca_da_vagem/*.jpg", 2)

    X = np.concatenate([X1,X2,X3], axis=0)
    y = np.concatenate([y1,y2,y3], axis=0)
    print(X.shape)
    return X, y

def loadDataUnit():
    print("Loading data...")
    (X4, y4) = getImgByClass(os.getcwd()+"/banco_imagens/teste/saudavel/healthy_635.jpg", 3)
    X = np.concatenate([X4], axis=0)
    y = np.concatenate([y4], axis=0)
    print(X.shape)
    return X, y

def loadDataTest():
    (X1, y1) = getImgByClass(os.getcwd()+"/banco_imagens/teste/saudavel/*.jpg", 0)
    (X2, y2) = getImgByClass(os.getcwd()+"/banco_imagens/teste/podridao_parda/*jpg", 1)
    (X3, y3) = getImgByClass(os.getcwd()+"/banco_imagens/teste/broca_da_vagem/*.jpg", 2)

    X = np.concatenate([X1,X2,X3], axis=0)
    y = np.concatenate([y1,y2,y3], axis=0)
    print(X.shape)
    return X, y