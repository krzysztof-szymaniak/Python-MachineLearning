import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


def users(movie_id):  # generator zwracajacy id osob ktore ocenily ToyStory
    for index_u, row_u in ratings.iterrows():
        if row_u['movieId'] == movie_id:
            yield row_u['userId']


m = 200
runPrediction = True  # True -> przewiduje ostatnie 15 ocen, False -> regresja na calym zbiorze
ratings = pd.read_csv('ratings.csv')
ratings = ratings[ratings['movieId'] < m + 1]
Y = ratings[ratings['movieId'] == 1]
Y = Y['rating'].to_numpy().reshape(-1, 1)

X = np.zeros((215, m))
i = 0
for u in users(1):
    print('Building row: ', i, '...')
    user_movies = ratings[ratings['userId'] == u]
    user_movies = user_movies[user_movies['movieId'] != 1]  # nie bierzemy oceny ToyStory podczas przewidywania
    for index, film in user_movies.iterrows():
        X[i, int(film['movieId']) - 2] = film['rating']
    i += 1
print('Generating linear model...')
model = linear_model.LinearRegression()
if runPrediction:
    trainX = X[:200, :]
    trainY = Y[:200]
    testX = X[200:, :]
    testY = Y[200:]
    model.fit(trainX, trainY)
    P = model.predict(testX)
    print("Model coefficients:\n", model.coef_)
    x = np.arange(215)
    x = x[200:]
    for i in range(15):
        print("testY[{}]={}, prediction[{}]={}, diff={}".format(i, testY[i], i, P[i], testY[i] - P[i]))
else:
    model.fit(X, Y)
    print("Model coefficients:\n", model.coef_)
    P = model.predict(X)
    testY = Y
    x = np.arange(215)
# budowanie wykresu
plt.figure(figsize=(12, 6))
plt.scatter(x, testY - P, color='red')
plt.ylim(-5, 5)
plt.xlabel('i')
plt.ylabel('error')
plt.title("Różnica między Y[i] i predicted[i] dla m=" + str(m))
plt.grid(True, color='black')
plt.show()
