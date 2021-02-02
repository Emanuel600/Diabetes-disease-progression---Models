#! /usr/bin/python3 
# remeber to make file executable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as lor
from sklearn.metrics import ConfusionMatrixDisplay as cmd
from sklearn.metrics import confusion_matrix as cm
from sklearn.datasets import load_diabetes as ld

dd = ld()
df = pd.DataFrame(dd['data'],columns=dd['feature_names'])
df['target'] = dd['target']
#Define x and y
x = df.values
y = df['target'].values
t = []
#Stratify data
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
#Make model (Already tested for overfitting)
lm = lor(max_iter=99999)
lm.fit(x,y)
#Plot results as a confusion matrix
q = cm(y,lm.predict(x))
b = cmd(q)
b.plot(cmap='Greens')
plt.show()
