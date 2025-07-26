from sklearn.metrics import confusion_matrix, classification_report
from ML_LAB3_Q1 import get_data
from ML_LAB3_Q4 import split_for_training
from ML_LAB3_Q5 import build_model

def show_evaluation(model, train_x, train_y, test_x, test_y):
    print(" Training Evaluation : ")
    preds_train = model.predict(train_x)
    print(confusion_matrix(train_y, preds_train))
    print(classification_report(train_y, preds_train))

    print("\n Testing Evaluation : ")
    preds_test = model.predict(test_x)
    print(confusion_matrix(test_y, preds_test))
    print(classification_report(test_y, preds_test))

features, labels = get_data()
train_x, test_x, train_y, test_y = split_for_training(features, labels)
knn_model = build_model(train_x, train_y, k=3)

show_evaluation(knn_model, train_x, train_y, test_x, test_y)
