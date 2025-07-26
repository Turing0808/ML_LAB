from ML_LAB3_Q1 import get_data
from ML_LAB3_Q4 import split_for_training
from ML_LAB3_Q5 import build_model

def check_model_accuracy(model, test_x, test_y):
    return model.score(test_x, test_y) # returns float between 0 and 1

features, labels = get_data() # for loading features,labels
train_x, test_x, train_y, test_y = split_for_training(features, labels) # splitting the data
knn_model = build_model(train_x, train_y, k=3)

acc = check_model_accuracy(knn_model, test_x, test_y)
print("The Test Accuracy is : ", acc)
