# Programs and data in manuscript titled "Narrow funnel-like interaction energy distribution is an indicator of specific protein interaction partner"

This repository contains programs and data in manuscript titled "Narrow funnel-like interaction energy distribution is an indicator of specific protein interaction partner"

These programs use deep learning to differentiate interaction energy distribution of functionally interacting protein pair and that of non-interacting protein pair.

※Because every program contains data shuffling, output will be slightly different from data in manuscript.

●Requirements:

1.Numpy (https://scipy.org/install/)

2.Tensorflow (https://www.tensorflow.org/install)

3.Keras (https://keras.io/getting_started/)

4.matplotlib (https://matplotlib.org/stable/users/installing/index.html)

●Download

```
git clone https://github.com/cjy318/protein_interaction
cd protein_interaction
```

●Prediction accuracy test for each deep learning model.

Run with

```
./<type of protein interaction>_<layer structure>_<optimizer>.py
```

-type of interaction

PPI: general protein interaction prediction

rkin: kinase substrate prediction

rubi: E3 ubiquitin ligase substrate prediction

-layer structure

model1: 1024-2

model2: 1024-64-2

model3: 1024-128-2

model4: 2048-128-2

model5: 1024-128-16-2

model6: 512-128-32-8-2

model7: 2048-512-128-32-8-2

-optimizer

Adam

SGD

RMSprop

Adamax

Adadelta

Nadam

Ftrl

-Example

```
./PPI_model2_Adamax.py
%general protein interaction prediction using 1024-64-2 layer structure and Adamax optimizer
```

-output

•Prediction accuracy from 10 fold cross validation with each number of epochs(1-100 or 1-200)

•graph of prediction accuracies and epochs

》For test data from HDOCKlite, run with

```
./<type of protein interaction>_<optimizer>.py
```

-type of interaction

hkin: kinase substrate prediction

hubi: E3 ubiquitin ligase substrate prediction

-optimizer

Adam

SGD

RMSprop

Adamax

Adadelta

Nadam

Ftrl

-output

•Prediction accuracy from 10 fold cross validation with each number of epochs(1-50)

•graph of prediction accuracies and epochs

●ROC test

Run with

```
./ROC_analysis.py
```




