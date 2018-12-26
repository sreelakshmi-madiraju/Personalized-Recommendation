import pandas as pd
import numpy as np
import warnings

import keras
import keras.models as kmodels
import keras.layers as klayers
import keras.backend as K
from keras.optimizers import Adam
from sklearn import dummy, metrics, cross_validation, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder 
from gensim.models import Word2Vec
from keras.layers import Dense,LSTM
from keras.models import Sequential
from tensorflow import set_random_seed
from keras.models import load_model
from keras.utils import plot_model
warnings.filterwarnings('ignore')
np.random.seed(123)


df=pd.read_csv('./beauty_df.csv',header='infer')
df=df[['reviewerID', 'asin','overall','reviewTime']]

df.columns=['userid', 'item_id', 'rating', 'reviewTime']
df=df.sort_values(['reviewTime'],ascending=[True])

df.userid = df.userid.astype('category').cat.codes.values
df.item_id = df.item_id.astype('category').cat.codes.values

n_users, n_items = len(df.userid.unique()), len(df.item_id.unique())
print(n_users, n_items)


# Creating one-hot for output

y = np.zeros((len(df), 5))
ratings=df.rating.values
ratings=[int(each)-1 for each in ratings]
for i in range(0,len(df)):
    y[i,ratings[i]]=1
    
# Neural Network Architecture

# Input Layersmodel
# Item Vector
item_input = keras.layers.Input(shape=[1])
item_vec = keras.layers.Flatten()(keras.layers.Embedding(n_items + 1, 32)(item_input))
item_vec = keras.layers.Dropout(0.5)(item_vec)
#User Vector
user_input = keras.layers.Input(shape=[1])
user_vec = keras.layers.Flatten()(keras.layers.Embedding(n_users + 1, 32)(user_input))
user_vec = keras.layers.Dropout(0.5)(user_vec)

#merge inputs
input_vecs = keras.layers.concatenate([item_vec, user_vec])

nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(input_vecs))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dropout(0.5)(keras.layers.Dense(128, activation='relu')(nn))
nn = keras.layers.normalization.BatchNormalization()(nn)
nn = keras.layers.Dense(128, activation='relu')(nn)

result = keras.layers.Dense(5, activation='softmax')(nn)

# Compile the model

model = kmodels.Model([item_input, user_input], result)
model.compile('adam', 'categorical_crossentropy')
model.save('basic_nn_capillary.h5')

# Divide data into train, test and fit the model

train_item_id, test_item_id, train_userid, test_userid, train_y, test_y = train_test_split(df.item_id, df.userid, y)#, shuffle=False)

# fit train data to the model
history = model.fit([train_item_id, train_userid], train_y,nb_epoch=5,validation_data=([test_item_id, test_userid], test_y))

# Find the test errror and accuracy

print("test error is")
print(metrics.mean_absolute_error(np.argmax(test_y, 1)+1,np.argmax(model.predict([test_item_id, test_userid]), 1)+1))
print("accuracy is")
print(accuracy_score(np.argmax(test_y, 1)+1,np.argmax(model.predict([test_item_id, test_userid]), 1)+1))

#getting user,item embeddings
weights=model.get_weights()
user_embeddings = weights[1]
item_embeddings = weights[0]


with open('item_embed','wb')as f:
    cPickle.dump(item_embeddings,f)
with open('user_embed','wb')as f:
    cPickle.dump(user_embeddings,f)
