
#The Libraries used to carry out Ensembling
import numpy as np
import pandas as pd


#Importing Datasets
d1 = pd.read_csv("~/Desktop/Submission_ONE.csv")
d2 = pd.read_csv("~/Desktop/SubmissionTwo.csv")


#Dropping ID's
data1 = d1.drop(columns=['id'], axis=1)
data2 = d2.drop(columns=['id'], axis=1)

#Creating Data frame From Scores
X = pd.DataFrame()

X['A'] = data1['label']
X['B'] = data2['label']


#taking transpose
X_transpose = pd.DataFrame.transpose(X)

# creating a list
X2 = X_transpose.values.tolist()
X1 = X.values.tolist()

result = ([0,0],[0,0])

for i in range(len(X2)):
    for j in range(len(X1[0])):
        for k in range (len(X1)):
            result[i][j] += X2[i][k] * X1[k][j]
for r in result:
    print(r)

rinv = np.linalg.inv(result)

# Scores generated based on Accuracy
score1 = 57458 * ((2 * 0.8864) - 1)
score2 = 57458 * ((2 * 0.9097) - 1)
XTY = [[score1], [score2]]


coeff = ([0],[0])
list = []
for l in range(len(rinv)):
    for m in range(len(XTY[0])):
        for n in range (len(XTY)):
            coeff[l][m] += rinv[l][n] * XTY[n][m]
for a in coeff:
    print(a)
    for u in a:
        list.append(u)
print(list)
print(X)

X['A'] = X['A'] * list[0]
X['B'] = X['B'] * list[1]

#Output Data frame
y = pd.DataFrame()
y['Output'] = X['A'] + X['B'] 

final = pd.read_csv("~/Desktop/SubmissionOne.csv")

final = final.drop(columns=['label'], axis=1)

final['label'] = y['Output']
print(final)
final.to_csv("ensemble2.csv")

# submission dataframe
final

# New submission file based on ensembling
final.to_csv("ensembleML.csv")

