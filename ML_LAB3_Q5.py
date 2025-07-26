from sklearn.neighbors import KNeighborsClassifier
from ML_LAB3_Q1 import get_data
from ML_LAB3_Q4 import split_for_training

def build_model(train_x, train_y, k=3): # knn with k value as 3
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(train_x, train_y)
    return model


features, labels = get_data()
train_x, test_x, train_y, test_y = split_for_training(features, labels)
knn_model = build_model(train_x, train_y, k=3)
print(" The Model is trained with k =", 3)
