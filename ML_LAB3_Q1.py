import pandas as pd
import numpy as np

def get_data(file_path="merged_vpn_data.csv"):
    data = pd.read_csv(file_path, header=None) #loading dataset
    data.dropna(inplace=True)

    inputs = data.iloc[:, :-1].values
    raw_labels = data.iloc[:, -1].astype(str).values

    labels = np.array([1 if tag.strip() == "VPN-BROWSING" else 0 for tag in raw_labels]) # Is used for handling spaces or if newline occurs

    unique, counts = np.unique(labels, return_counts=True) # counting the classes
    print("Class counts:", dict(zip(unique, counts)))  # 0: Non-VPN, 1: VPN-BROWSING

    return inputs, labels

def class_stats(inputs, labels):
    data_0 = inputs[labels == 0]
    data_1 = inputs[labels == 1]

    if len(data_0) == 0 or len(data_1) == 0:
        print("One of the classes is missing")
        return None, None, None, None, np.nan

    mean_0 = np.mean(data_0, axis=0)
    mean_1 = np.mean(data_1, axis=0)

    std_0 = np.std(data_0, axis=0)
    std_1 = np.std(data_1, axis=0)

    dist = np.linalg.norm(mean_0 - mean_1)
    return mean_0, mean_1, std_0, std_1, dist


features, labels = get_data()
mean_0, mean_1, std_0, std_1, gap = class_stats(features, labels)

if not np.isnan(gap):
    print("Distance between Non-VPN and VPN class centroids:", gap)
else:
    print(" Could not compute class distance.")
