#! /usr/bin/python3 
# remeber to make file executable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import confusion_matrix as cm
from sklearn.datasets import load_diabetes as ld
from sklearn.model_selection import train_test_split as tts

dd = ld()
df = pd.DataFrame(dd['data'],columns=dd['feature_names'])
df['target'] = dd['target']
#Define x and y
x = df.values
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
lm = lor(max_iter=99999)
lm.fit(x,y)
#Plot results
q = cm(y,lm.predict(x))
b = cmd(q)
b.plot(cmap='Greens')
plt.show()