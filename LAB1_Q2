A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

if len(A[0]) != len(B):
    print("Error: Matrices can't be multiplied.")
else:
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            total = 0
            for k in range(len(B)):
                total += A[i][k] * B[k][j]
            row.append(total)
        result.append(row)
    print("Product matrix:")
    for row in result:
        print(row)
