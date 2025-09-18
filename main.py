import numpy as np
from collections import defaultdict
import csv
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
print(f'accuracy is: {accuracy*100:.1f}%') #TN+TP/TN+TP+FP+FN
print(confusion_matrix(Y_test,prediction,labels=[0,1])) # |TN FP|FN TP|
precision_score(Y_test,prediction,pos_label=1) #TP/TP+FP'
recall_score(Y_test,prediction,pos_label=1) #TP/TP+FN
f1_score(Y_test,prediction,pos_label=1)#2* precision*recall/precision+recall
report = classification_report(Y_test,prediction)
print(report)
pos_prob = prediction_prob[:,1]
thresholds = np.arange(0.0,1.1,0.05)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob,Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y==1:
                true_pos[i] += 1
            else:
                false_pos[i]+=1
        else:
            break

n_pos_test = (Y_test==1).sum()
n_neg_test = (Y_test==0).sum()
true_pos_rate = [tp/n_pos_test for tp in true_pos]
false_pos_rate = [fp/n_neg_test for fp in false_pos]


plt.figure()
lw = 2
plt.plot(false_pos_rate,true_pos_rate,color = 'darkorange', lw=lw)
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operator Characteristic')
plt.legend(loc="lower right")
plt.show()

roc_auc_score(Y_test,pos_prob)
k = 5
k_fold = StratifiedKFold(n_splits=k, random_state=42)
smoothing_factor_option = [1,2,3,4,5,6]
fit_prior_option = [True,False]
auc_record = {}

for train_indices, test_indices in k_fold.split(X,Y):
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train,Y_test = Y[train_indices], Y[test_indices]
    for alpha in smoothing_factor_option:
        if alpha not in auc_record:
            auc_record = {}
        for fit_prior in fit_prior_option:
            clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
            clf.fit(X_train, Y_train)
            prediction_prob = clf.predict_proba(X_test)
            pos_prob=prediction_prob[:,1]
            auc = roc_auc_score(Y_test,pos_prob)
            auc_record[alpha][fit_prior] = auc + auc_record[alpha].get(fit_prior,0.0)

for smoothing, smoothing_record in auc_record.items():
    for fit_prior, auc, in smoothing_record.items():
        print(f' {smoothing} {fit_prior} {auc/k:.5f}')

clf = MultinomialNB(alpha=2.0, fit_prior=False)
clf.fit(X_train, Y_train)
pos_prob = clf.predict_proba(X_test)[:, 1]
print('AUC with the the best model:', roc_auc_score(Y_test,pos_prob))
