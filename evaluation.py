# Aviv Ben Yaakov 206261695

from sklearn.metrics import mean_squared_error
from math import sqrt
import math
# Import Pandas
import pandas as pd
import data
import numpy as np

users_predictions = None
users_data = None
users_data_partial = None


def bulid_user_predictions(test, cf, user_based):
    """*** create user-predictions dictionary ***"""
    global users_predictions
    users_predictions = {}
    users_id = test['userId'].unique()

    "calculate the movies predictions for all the unique users in the test set"
    "save the (id, predictions) as (key, value) in the dictionary"
    for uid in users_id:
        predictions = cf.predict_movies(uid, 10, user_based)
        predictions_id_array = np.vectorize(cf.title_movie_id.get)(predictions)
        users_predictions[uid] = predictions_id_array


def precision_10(test_set, cf, is_user_based=False):
    """*** YOUR CODE HERE ***"""
    global users_predictions, users_data_partial
    test_df = test_set

    "save only the rows where the rating is atleast 4.0 and group by userId"
    if users_data_partial is None:
        test_df = test_df.drop(test_df.loc[test_df['rating'] < 4.0].index)
        users_data_partial = test_df.groupby('userId')

    "if doesn't initilized, create the user-predictions dictionary"
    if users_predictions is None:
        bulid_user_predictions(test_df, cf, is_user_based)

    total_hit = 0

    "calculate the hit value of each user and add to the total hit"
    for uid, group in users_data_partial:
        movies_id_array = group['movieId'].values
        count = len(np.intersect1d(users_predictions[uid], movies_id_array))
        current_hit = count / 10
        total_hit += current_hit

    "calculte the p@k value by deviding the total hit by the number of users"
    val = total_hit / len(users_data_partial)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based=False):
    """*** YOUR CODE HERE ***"""
    global users_predictions, users_data_partial
    test_df = test_set

    "save only the rows where the rating is atleast 4.0 and group by userId"
    if users_data_partial is None:
        test_df = test_df.drop(test_df.loc[test_df['rating'] < 4.0].index)
        users_data_partial = test_df.groupby('userId')

    "if doesn't initilized, create the user-predictions dictionary"
    if users_predictions is None:
        bulid_user_predictions(test_df, cf, is_user_based)

    total_hit = 0

    "calculate the hit value of each user and add to the total hit"
    for uid, group in users_data_partial:
        movies_id_arr = group['movieId'].values
        position = 1
        current_hit = 0
        for movie_id in users_predictions[uid]:
            if movie_id in movies_id_arr:
                current_hit += 1 / position
            position += 1
        total_hit += current_hit

    "calculte the ARHA value by deviding the total hit by the number of users"
    val = total_hit / len(users_data_partial)
    print("ARHA: " + str(val))


def RSME(test_set, cf, is_user_based=False):
    """*** YOUR CODE HERE ***"""
    if is_user_based:
        predicted_ratings = cf.user_based_matrix
    else:
        predicted_ratings = cf.item_based_metrix

    test_df = test_set
    users_data = test_df.groupby('userId')
    n_ratings = test_df['rating'].count()
    total_rmse = 0

    "calculate the rmse value of each user in the test set and add to the total sum"
    for uid, group in users_data:
        norm_user_id = cf.users_id_dict[uid]
        movies_id_array = group['movieId'].values
        actual_ratings_array = group['rating'].values
        current_rmse = 0

        "iterate over the ratings of the user, and get the corresponding rating from the predicted_ratings matrix"
        "the rmse value is the quadratic distance between the actual rating and the predicted one"
        for movie_Id, actual_rating in zip(movies_id_array, actual_ratings_array):
            norm_movie_id = cf.movies_id_dict[movie_Id]
            predicted_rating = predicted_ratings[norm_user_id, norm_movie_id]
            current_rmse += math.pow((predicted_rating - actual_rating), 2)
        total_rmse += current_rmse

    "calculte the RMSE value by deviding the total_rmse by the number of ratings in the test set"
    val = sqrt(total_rmse / n_ratings)
    print("RMSE: " + str(val))
