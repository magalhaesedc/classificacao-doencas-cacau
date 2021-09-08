import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('accuracy_cnn2.csv', names=['train', 'test'])

epoch = 100

X = np.arange(1, epoch+1)
y_train = df['train']
y_test = df['test']
    
plt.plot(X, y_train)
plt.plot(X, y_test)
plt.title('Learning curve')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('curve_accuracy.png', dpi=300)
plt.show()