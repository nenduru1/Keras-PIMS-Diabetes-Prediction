# -*- coding: utf-8 -*-
"""
Created on Sun Jul 09 15:12:26 2017

@author: o222069
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

seed=7
np.random.seed(seed)

data=np.loadtxt('pima-indians-diabetes.csv',delimiter=',')
X=data[:,0:8]
y=data[:,8]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Define and Compile
model=Sequential()
model.add(Dense(4,input_dim=8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the model
model.fit(X_train,y_train,epochs=50,batch_size=5)

y_pred=model.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#evaluate the model
scores=model.evaluate(X_train,y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(4,input_dim=8,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
    
classifier=KerasClassifier(build_classifier,batch_size=5,epochs=100)
accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5,n_jobs=1)

mean=accuracies.mean()

variance=accuracies.std()

#improving the ANN for overfitting sing dropout
from keras.layers import Dropout

classifier=Sequential()
classifier.add(Dense(4,input_dim=8,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(4, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(rate=0.2))

classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,epochs=50,batch_size=5)


#Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(4,input_dim=8,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[20,30],
           'epochs':[100,150],
           'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                        param_grid=parameters,
                        scoring='accuracy',
                        cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_Accuracy=grid_search.best_score_

#Tuning the ANN
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier=Sequential()
    classifier.add(Dense(12,input_dim=8,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(8, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return classifier
classifier=KerasClassifier(build_fn=build_classifier)
parameters={'batch_size':[2,1],
           'epochs':[100,150],
           'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                        param_grid=parameters,
                        scoring='accuracy',
                        cv=10)
grid_search=grid_search.fit(X_train,y_train)
best_parameters=grid_search.best_params_
best_Accuracy=grid_search.best_score_
print(best_parameters,best_Accuracy)
