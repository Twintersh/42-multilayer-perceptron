## 42-multilayer-percepton, a project from 42 school

The aim of this project is to recreate a Multilayer perceptron from scratch to predict the outcome of a breast cancer diagnosis.

> "The goal of this project is to give you a first approach to artificial neural networks, and to have you implement the algorithms at the heart of the training process. At the same time you are going to have to get reacquainted with the manipulation of derivatives and linear algebra as they are indispensable mathematical tools for the success of the project."

### How to use it ?

**To use it you have to run src/splitDataset.py first.**

    python3 src/splitDataset.py -a [splitfactor] -r [random seed (optional)]
This program will split the data.csv file in two part : one to train the model, and the other to validate it. **The split factor must be between 1 and 0.** The higher it is, the more data it will put into the training data file.

**Now, you can run the training program.**

    python3 src/train.py -c [config file] -r [random seed (optional)]
In the repository, I supplied two config file. One that trains the model right, and another that produces a case of overfitting, there to test my early stopping implementation.

**Get the accuracy by running the predict program**

    python3 src/predict.py
this one is straightforward. It prints the accuracy of the model with the training and the validation data.

