from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

data = pd.read_csv("data/covtype.csv")


X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

# replace class 7 with 0 to work with to_categorical
y = y.replace(7, 0)



# this remove features and all data analysis is used by the kaggler of this dataset competetion
# https://www.kaggle.com/roshanchoudhary/forest-cover-walkthrough-in-python-knn-96-51/notebook
# above link is used for getting the cleaned dataset insight
remove_features = ['Hillshade_3pm','Soil_Type7','Soil_Type8','Soil_Type14','Soil_Type15',
     'Soil_Type21','Soil_Type25','Soil_Type28','Soil_Type36','Soil_Type37']

X.drop(remove_features, axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

# deep learning model will be used to get some insights

num_classes = 7


y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

model_file = Path('trained_model.h5')
if model_file.is_file() == False:
    model = Sequential()
    model.add(Dense(1024, input_shape=(X_train.shape[1],), activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='adagrad',
                metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=15, batch_size=128)

    model.save('trained_model.h5')

own_model = load_model('trained_model.h5')

#y_pred = own_model.predict(X_test, batch_size=64)
print(own_model.metrics_names)
print(own_model.evaluate(X_test, y_test))