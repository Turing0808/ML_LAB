from ML_LAB3_Q1 import get_data
from ML_LAB3_Q4 import split_for_training
from ML_LAB3_Q5 import build_model

def predict_inputs(model, inputs): # its for predicting the output class using our trained model
    return model.predict(inputs)
 
features, labels = get_data() # loading the data
train_x, test_x, train_y, test_y = split_for_training(features, labels) #splitting the data
knn_model = build_model(train_x, train_y, k=3)

sample_preds = predict_inputs(knn_model, test_x[:5])
print("Predictions on first 5 test rows:", sample_preds)
