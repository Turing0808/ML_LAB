import pandas as pd

nonvpn = pd.read_csv("nonvpndataset.txt", header=None)
vpn = pd.read_csv("vpndataset.txt", header=None)

combined = pd.concat([nonvpn, vpn], ignore_index=True) #merging them together
combined.to_csv("merged_vpn_data.csv", index=False, header=False) #saving the merged dataset