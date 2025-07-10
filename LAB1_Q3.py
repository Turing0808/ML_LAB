list1 = [1, 2, 3, 4, 5, 5]
list2 = [4, 5, 6, 7, 5]
common = list(set(list1) & set(list2))
print("Common elements:", common)
print("Count:", len(common))
