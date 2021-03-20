

import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils

TEXT="Title"
CLASS="Conference"


df = pd.read_csv('title_conference.csv')
df = df[pd.notnull(df[CLASS])]
print(df.head(10))
#### print(df[TEXT].apply(lambda x: len(x.split(' '))).sum())
print(df[CLASS].unique())



train_size = int(len(df) * .7)
train_text = df[TEXT][:train_size]
train_class = df[CLASS][:train_size]

test_text = df[TEXT][train_size:]
test_class = df[CLASS][train_size:]

max_words = 1000
tokenize = text.Tokenizer(num_words=max_words, char_level=False)
tokenize.fit_on_texts(train_text) # only fit on train

x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)

encoder = LabelEncoder()
encoder.fit(train_class)
y_train = encoder.transform(train_class)
y_test = encoder.transform(test_class)

num_classes = np.max(y_train) + 1
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 200

# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)
model.save('text_model.h5', save_format="h5")


#from keras.models import load_model
#import numpy as np
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.models import load_model
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing import text, sequence
#from keras.preprocessing.sequence import pad_sequences
#model = load_model('text_model.h5')


#labels=['INFOCOM' ,'ISCAS','SIGGRAPH','VLDB','WWW']
#max_words = 1000
#new_text = ['Automatic sanitization of social network data to prevent inference attacks.']
#tk = text.Tokenizer(num_words=max_words, char_level=False)
#tk.fit_on_texts(new_text)
#seq = tk.texts_to_matrix(new_text)
#preds = model.predict(seq)
#classes = model.predict_classes(seq)
#print(preds)


#re_class=[]

#for c in classes :
#    a=labels[c] 
#    re_class.append(a)

#print(re_class)
#print("\n")
#for pred in preds:
#    i=0;
#    print(re_class[i])
#    for lb in labels:   
#        print("%s: %.10f%%" % (lb, pred[i]*100))
#        i+=1
#    print("\n")

#df = pd.DataFrame(data=['VLDB', 'ISCAS', 'SIGGRAPH' ,'INFOCOM' ,'WWW'], columns=['x'])
#le = preprocessing.LabelEncoder()
#le.fit(df['x'])
#### this prints ['first', 'fourth', 'second', 'third']
#encoded = le.transform(['VLDB', 'ISCAS', 'SIGGRAPH' ,'INFOCOM' ,'WWW']) 
#print (encoded)
