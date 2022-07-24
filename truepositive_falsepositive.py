import numpy as np
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

tot_data=np.load('rubi_data.pkl', allow_pickle=True)
np.random.shuffle(tot_data)
#random shuffle for k-fold validation
data=tot_data[:,1:]
label=np.array(tot_data[:, :1])
datal=data.tolist()
labell=label.tolist()

k=10
num_validation_sample=len(data)//k
num_epochs=56

true_positive_prediction=[]
false_positive_prediction=[]


for fold in range(k):
    validation_data=data[num_validation_sample*fold:num_validation_sample*(fold+1)]
    validation_label=label[num_validation_sample*fold:num_validation_sample*(fold+1)]
    v_label=to_categorical(validation_label)
    training_data=np.array(datal[:num_validation_sample*fold]+datal[num_validation_sample*(fold+1):])
    training_label=np.array(labell[:num_validation_sample*fold]+labell[num_validation_sample*(fold+1):])
    t_label=to_categorical(training_label)
    
    model=models.Sequential()
    model=models.Sequential()
    model.add(layers.Dense(1024, activation='relu', input_shape=(1000*2,)))
    model.add(layers.Dense(128,activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(training_data,t_label,epochs=num_epochs,batch_size=16,verbose=0)
    #I used minibatch in this case. It will take few minute
    #If you want to see long individual results, remove "verbose=0"
    #model.fit(training_data,t_label,epochs=num_epochs,batch_size=16)
    x=model.predict(validation_data)
    y=x>=0.892
    prediction_results=y.astype(int)[:,1]
    true_positive_prediction.append(np.dot(validation_label[:,0],prediction_results))
    false_positive_prediction.append(np.dot(np.invert(validation_label[:,0].astype(bool)).astype(int),prediction_results))

True_positive_rate=sum(true_positive_prediction)/(sum(label))
False_positive_rate=sum(false_positive_prediction)/(sum(label))

print("True positive rate:")
print(True_positive_rate[0]*100,"%")
print("False positive rate:")
print(False_positive_rate[0]*100,"%")