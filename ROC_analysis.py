import numpy as np
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical

tot_data=np.load('data1.pkl', allow_pickle=True)
np.random.shuffle(tot_data)
#random shuffle for k-fold validation
data=tot_data[:,1:]
label=np.array(tot_data[:, :1])
datal=data.tolist()
labell=label.tolist()

k=10
num_validation_sample=len(data)//k
num_epochs=9


True_positive_prediction=np.zeros(1000)
False_positive_prediction=np.zeros(1000)
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
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='Adadelta',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(training_data,t_label,epochs=num_epochs,batch_size=16,verbose=0)
    #I used minibatch in this case. It will take few minute
    #If you want to see long individual results, remove "verbose=0"
    #model.fit(training_data,t_label,epochs=num_epochs,batch_size=16)
    x=model.predict(validation_data)
    true_positive_prediction=[]
    false_positive_prediction=[]
    for threshold in np.arange(0.001,1.001,0.001).tolist():


        y=x>=threshold
        prediction_results=y.astype(int)[:,1]
        true_positive_prediction.append(np.dot(validation_label[:,0],prediction_results))
        false_positive_prediction.append(np.dot(np.invert(validation_label[:,0].astype(bool)).astype(int),prediction_results))
    True_positive_prediction=True_positive_prediction+np.array(true_positive_prediction)
    False_positive_prediction=False_positive_prediction+np.array(false_positive_prediction)

True_positive_rate=True_positive_prediction/(sum(label))
False_positive_rate=False_positive_prediction/(sum(label))
#Kinase results




tot_data=np.load('rubi_data.pkl', allow_pickle=True)
np.random.shuffle(tot_data)
#random shuffle for k-fold validation
data=tot_data[:,1:]
label=np.array(tot_data[:, :1])
datal=data.tolist()
labell=label.tolist()

k=10
num_validation_sample=len(data)//k
num_epochs=61


True_positive_prediction1=np.zeros(1000)
False_positive_prediction1=np.zeros(1000)
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
    model.add(layers.Dense(2, activation='softmax'))

    model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(training_data,t_label,epochs=num_epochs,batch_size=16,verbose=0)
    #I used minibatch in this case. It will take few minute
    #If you want to see long individual results, remove "verbose=0"
    #model.fit(training_data,t_label,epochs=num_epochs,batch_size=16)
    x=model.predict(validation_data)
    true_positive_prediction=[]
    false_positive_prediction=[]
    for threshold in np.arange(0.001,1.001,0.001).tolist():


        y=x>=threshold
        prediction_results=y.astype(int)[:,1]
        true_positive_prediction.append(np.dot(validation_label[:,0],prediction_results))
        false_positive_prediction.append(np.dot(np.invert(validation_label[:,0].astype(bool)).astype(int),prediction_results))
    True_positive_prediction1=True_positive_prediction1+np.array(true_positive_prediction)
    False_positive_prediction1=False_positive_prediction1+np.array(false_positive_prediction)

True_positive_rate1=True_positive_prediction1/(sum(label))
False_positive_rate1=False_positive_prediction1/(sum(label))

#ubi

tot_data=np.load('posi_data.pkl', allow_pickle=True)
np.random.shuffle(tot_data)
#random shuffle for k-fold validation
data=tot_data[:,1:]
label=np.array(tot_data[:, :1])
datal=data.tolist()
labell=label.tolist()

k=10
num_validation_sample=len(data)//k
num_epochs=56


True_positive_prediction2=np.zeros(1000)
False_positive_prediction2=np.zeros(1000)
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

    model.compile(optimizer='Adamax',loss='binary_crossentropy',metrics=['accuracy'])

    model.fit(training_data,t_label,epochs=num_epochs,batch_size=16,verbose=0)
    #I used minibatch in this case. It will take few minute
    #If you want to see long individual results, remove "verbose=0"
    #model.fit(training_data,t_label,epochs=num_epochs,batch_size=16)
    x=model.predict(validation_data)
    true_positive_prediction=[]
    false_positive_prediction=[]
    for threshold in np.arange(0.001,1.001,0.001).tolist():


        y=x>=threshold
        prediction_results=y.astype(int)[:,1]
        true_positive_prediction.append(np.dot(validation_label[:,0],prediction_results))
        false_positive_prediction.append(np.dot(np.invert(validation_label[:,0].astype(bool)).astype(int),prediction_results))
    True_positive_prediction2=True_positive_prediction2+np.array(true_positive_prediction)
    False_positive_prediction2=False_positive_prediction2+np.array(false_positive_prediction)

True_positive_rate2=True_positive_prediction2/(sum(label))
False_positive_rate2=False_positive_prediction2/(sum(label))









pl.plot(np.concatenate((np.zeros(1),False_positive_rate[::-1],np.ones(1))),np.concatenate((np.zeros(1),True_positive_rate[::-1],np.ones(1))),'g-',np.concatenate((np.zeros(1),False_positive_rate1[::-1],np.ones(1))),np.concatenate((np.zeros(1),True_positive_rate1[::-1],np.ones(1))),'b-',np.concatenate((np.zeros(1),False_positive_rate2[::-1],np.ones(1))),np.concatenate((np.zeros(1),True_positive_rate2[::-1],np.ones(1))),'r-',np.arange(0,1.01,0.01),np.arange(0,1.01,0.01),'k:')
AUC_kin=np.trapz(np.concatenate((np.zeros(1),True_positive_rate[::-1],np.ones(1))),np.concatenate((np.zeros(1),False_positive_rate[::-1],np.ones(1))))
AUC_ubi=np.trapz(np.concatenate((np.zeros(1),True_positive_rate1[::-1],np.ones(1))),np.concatenate((np.zeros(1),False_positive_rate1[::-1],np.ones(1))))
AUC_pdb=np.trapz(np.concatenate((np.zeros(1),True_positive_rate2[::-1],np.ones(1))),np.concatenate((np.zeros(1),False_positive_rate2[::-1],np.ones(1))))
print(AUC_kin)
print(AUC_ubi)
print(AUC_pdb)