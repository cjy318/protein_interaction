import numpy as np
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

tot_data=np.load('rkin_data.pkl', allow_pickle=True)
np.random.shuffle(tot_data)
#random shuffle for k-fold validation
data=tot_data[:,1:]
label=np.array(tot_data[:, :1])
datal=data.tolist()
labell=label.tolist()

k=10
num_validation_sample=len(data)//k

k_fold_score=[]
for num_epochs in range(1,50):
    validation_scores=[]
    for fold in range(k):
        validation_data=data[num_validation_sample*fold:num_validation_sample*(fold+1)]
        validation_label=label[num_validation_sample*fold:num_validation_sample*(fold+1)]
        v_label=to_categorical(validation_label)
        training_data=np.array(datal[:num_validation_sample*fold]+datal[num_validation_sample*(fold+1):])
        training_label=np.array(labell[:num_validation_sample*fold]+labell[num_validation_sample*(fold+1):])
        t_label=to_categorical(training_label)
    
        model=models.Sequential()
        model.add(layers.Dense(1024, activation='relu', input_shape=(1000*2,)))
        model.add(layers.Dense(64,activation='relu'))
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(optimizer='Adadelta',loss='binary_crossentropy',metrics=['accuracy'])

        model.fit(training_data,t_label,epochs=num_epochs,batch_size=16,verbose=0)
        #I used minibatch in this case. It will take few minute
        #If you want to see long individual results, remove "verbose=0"
        #model.fit(training_data,t_label,epochs=num_epochs,batch_size=16)
        validation_score=model.evaluate(validation_data,v_label,verbose=0)[1]
        validation_scores.append(validation_score)
    k_fold_score.append(np.average(validation_scores))
import matplotlib.pyplot as pl
epoch=range(1,50)
pl.plot(epoch,k_fold_score,'ko')
pl.show()
print(k_fold_score)
