#! /usr/bin/python3 
# remeber to make file executable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import confusion_matrix as cm
from sklearn.datasets import load_diabetes as ld
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler as ssc

dd = ld()
sc = ssc()
df = pd.DataFrame(dd['data'],columns=dd['feature_names'])
df['target'] = dd['target']
#Define x and y
x = df.values
x = sc.fit_transform(x)
y = df['target'].values
t = []
for i in range(len(y)):
    if y[i]<50:
        t.append('1')
    elif y[i]<100:
        t.append('2')
    elif y[i]<150:
        t.append('3')
    elif y[i]<200:
        t.append('4')
    elif y[i]<250:
        t.append('5')
    elif y[i]<300:
        t.append('6')
    elif y[i]<350:
        t.append('7')
    else:
        t.append('8')
y = t
#Split data
xt,xtt,yt,ytt = tts(x,y,test_size=0.3)
#Make model
nm = mlp(max_iter=9999, hidden_layer_sizes=(100,4))
nm.fit(x,y)
#Plot results
q = cm(y,nm.predict(x))
d = cmd(q)
d.plot(cmap='Greens')
plt.show()