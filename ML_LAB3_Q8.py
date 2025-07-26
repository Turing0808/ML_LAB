import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from ML_LAB3_Q1 import get_data
from ML_LAB3_Q4 import split_for_training

def plot_accuracy(train_x, train_y, test_x, test_y):
    k_values = range(1, 12)
    scores = []

    for k in k_values: # creating and training the model
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_x, train_y)
        acc = model.score(test_x, test_y) # to calculate the accuracy on test set
        scores.append(acc)

    plt.plot(list(k_values), scores, marker='o', color='green')
    plt.title("Accuracy vs k (kNN)")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    return scores

features, labels = get_data()
train_x, test_x, train_y, test_y = split_for_training(features, labels)
plot_accuracy(train_x, train_y, test_x, test_y)
