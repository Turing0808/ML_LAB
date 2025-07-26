from sklearn.model_selection import train_test_split
from ML_LAB3_Q1 import get_data

def split_for_training(data, tags, test_ratio=0.3): # splitting dataset for training and testing
    return train_test_split(data, tags, test_size=test_ratio, random_state=42)

features, labels = get_data()
train_x, test_x, train_y, test_y = split_for_training(features, labels)
print("Train size:", len(train_x), "Test size:", len(test_x))
