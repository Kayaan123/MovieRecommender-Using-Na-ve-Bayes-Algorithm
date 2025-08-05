import numpy as np
from collections import defaultdict
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
data_path = '/Users/kayaansomchand/Documents/Internship/MovieRecommender/MovieRecommender-Using-Na-ve-Bayes-Algorithm/ml-latest-small/ratings.csv'

"""
This function loads the data from the csv file, 


"""
def load_rating_data(data_path):
    movie_id_mapping = {} #maps movie_ids to column indices
    movie_n_rating = defaultdict(int) #counts how many times each movies was rating
    ratings_temp = [] #temporarily stores 

    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        user_ids = set() #track unique userIDs
        for row in reader:
            user_ids.add(int(row[0]))
            user_id = int(row[0]) - 1
            movie_id = row[1]
            rating = float(row[2])

            if movie_id not in movie_id_mapping:
                movie_id_mapping[movie_id] = len(movie_id_mapping)

            ratings_temp.append((user_id, movie_id_mapping[movie_id], rating))
            if rating > 0:
                movie_n_rating[movie_id] += 1

    
    n_movies = len(movie_id_mapping)
    n_users = max(user_ids)
    data = np.zeros([n_users, n_movies], dtype=np.float32)

    for user_id, movie_index, rating in ratings_temp:
        data[user_id, movie_index] = rating

    return data, movie_n_rating, movie_id_mapping, n_movies,n_users

def display_distribution(data):
    values, counts = np.unique(data[data > 0], return_counts=True)  # ignore zeros
    for value, count in zip(values, counts):
        print(f'Number of ratings with value {value}: {count}')

data, movie_n_rating, movie_id_mapping,n_movies,n_users = load_rating_data(data_path)
display_distribution(data)
print("\n\nNumber of Users", n_users)
print("\n\nNumber of Movies", n_movies)

movie_id_most, n_rating_most = sorted(movie_n_rating.items(), key=lambda d: d[1], reverse=True)[0]
print(f'Movie ID {movie_id_most} has {n_rating_most} ratings.')

X_raw= np.delete(data,movie_id_mapping[movie_id_most],axis=1)
Y_raw = data[:,movie_id_mapping[movie_id_most]]
X = X_raw[Y_raw > 0]
Y=Y_raw[Y_raw > 0]
display_distribution(Y)
recommended = 3
Y[Y<=recommended] = 0
Y[Y>recommended] = 1
n_pos = (Y==1).sum()
n_neg = (Y==0).sum()

print(f'{n_pos} positive samples and {n_neg} negative samples.')

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

clf = MultinomialNB(alpha=1,fit_prior=True)
clf.fit(X_train,Y_train)

prediction_prob = clf.predict_proba(X_test)
print(prediction_prob[0:10])
prediction = clf.predict(X_test)
print(prediction[:10])

accuracy = clf.score(X_test, Y_test)
print(f'accuracy is: {accuracy*100:.1f}%')