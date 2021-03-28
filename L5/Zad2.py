import numpy as np
import pandas as pd
import random

np.seterr(divide='ignore', invalid='ignore')
my_ratings = np.zeros((9019, 1))
my_ratings[2571] = 5  # 2571 - Matrix
my_ratings[32] = 4  # 32 - Twelve Monkeys
my_ratings[260] = 5  # 260 - Star Wars IV
my_ratings[1097] = 4  # E.T. the Extra-Terrestrial

my_ratings[296] = 5  # Pulp Fiction
my_ratings[6934] = 4  # Matrix Revolutions

my_ratings[1203] = 5  # 12 Angry Men
my_ratings[665] = 5  # Underground
my_ratings[3528] = 5  # Prince of Tides
my_ratings[1955] = 5  # Kramer vs. Kramer
my_ratings_norm = np.nan_to_num(my_ratings / np.linalg.norm(my_ratings))

print('Reading data...')
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
ratings = ratings[ratings['movieId'] < 10000]
movies = movies[movies['movieId'] < 10000]

x = np.zeros((611, 9019))
i = 0
for u in range(1, 611):
    print('Building row ', u, '...')
    user_movies = ratings[ratings['userId'] == u]
    for index, film in user_movies.iterrows():
        x[u, int(film['movieId'])] = film['rating']

print("Calculating...")
X = np.nan_to_num(x / np.linalg.norm(x, axis=0))
z = np.dot(X, my_ratings_norm)
Z = np.nan_to_num(z / np.linalg.norm(z))
R = np.dot(X.T, Z)
print("Getting results...")
results = [(R[i, 0], i) for i in range(len(R))]
results.sort(reverse=True, key=lambda e: e[0])

notif = ['Wedlug moich obliczen powinien spodobac ci sie film', 'Sprobuj takze obejrzec',
         'Moja propozycja na wieczorny seans dla Ciebie to', 'Moze masz ochote na',
         'Prawdopodobnie polubisz tez', 'Polecam',  'Swietnym filmem jest']

opt = ''
while opt != 'q' and results:
    movie = results[0][1]  # element ze szczytu ma najwieksze prawdopodbiestwo trafienia w gusta
    if my_ratings[movie] == 0:  # wyswietla tylko filmy, ktorych user jeszcze nie ocenil
        row = movies[movies['movieId'] == movie]
        title = row['title'].values[0]
        print("\n\n", notif[random.randrange(len(notif))], title)
        opt = input("\nWcisnij 'q', zeby wyjsc lub 'n' dla nastepnej propozycji...\n")
    results.pop(0)
