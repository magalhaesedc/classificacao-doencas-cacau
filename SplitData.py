import glob
import shutil
import numpy as np
import os

path_data = os.getcwd()+'/banco_imagens/treino/'
path_teste =  os.getcwd()+'/banco_imagens/teste/'

def splitData(folder, percent):
    files = glob.glob(path_data+folder+'/*.jpg')
    n = len(files) / 100 * percent
    imgs = np.random.choice(files, int(n), False)
    for img in imgs:
        shutil.move(img, path_teste+folder)