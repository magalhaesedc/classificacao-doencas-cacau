import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('accuracy_cnn1.csv', names=['train', 'test'])

epoch = 30

X = np.arange(1, epoch+1)
y_train = df['train']
y_test = df['test']
    
plt.plot(X, y_train)
plt.plot(X, y_test)
plt.title('Learning curve')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.savefig('curve_accuracy.png', dpi=300)
plt.show()