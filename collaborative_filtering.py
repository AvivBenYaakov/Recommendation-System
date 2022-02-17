# Aviv Ben Yaakov 206261695
import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.users_id_dict = {}
        self.users_real_id_dict = {}
        self.movies_id_dict = {}
        self.movie_real_id_dict = {}
        self.movie_id_title = {}
        self.title_movie_id = {}
        self.movies_df = None
        self.ratings_df = None
        self.dicts_created = False

    def create_dictionaries(self, users_unique, movies_unique, titles_unique, movieId_unique, users_n, movies_n):
        self.users_id_dict = dict(zip(users_unique, range(users_n)))
        self.users_real_id_dict = dict(zip(range(users_n), users_unique))

        self.movies_id_dict = dict(zip(movies_unique, range(movies_n)))
        self.movie_real_id_dict = dict(zip(range(movies_n), movies_unique))

        self.movie_id_title = dict(zip(movieId_unique, titles_unique))
        self.title_movie_id = dict(zip(titles_unique, movieId_unique))

    def create_fake_user(self, rating):
        """*** YOUR CODE HERE ***"""
        user_id = 283238
        movies_df = self.movies_df

        # save subset of the first 100 rows
        movies_subset_df = movies_df.head(100)

        # devide the DataFrame to comedies, darmas, and the rest
        comedies_subset_df = movies_subset_df.loc[movies_subset_df['genres'].str.contains('Comedy')]
        dramas_subset_df = movies_subset_df.loc[movies_subset_df['genres'] == 'Drama']
        remaining_subset_df = movies_subset_df.loc[(movies_subset_df['genres'] != 'Drama') &
                                                   (movies_subset_df['genres'] != 'Comedy')]

        # get the movie Id's
        comedies_id = comedies_subset_df['movieId'].tolist()
        dramas_id = dramas_subset_df['movieId'].tolist()
        remaining_id = remaining_subset_df['movieId'].tolist()

        # rate all the comedies 5.0
        df1 = pd.DataFrame({"userId": [user_id] * len(comedies_id), "movieId": comedies_id,
                            "rating": [5.0] * len(comedies_id)})
        # rate all the dramas 1.0
        df2 = pd.DataFrame({"userId": [user_id] * len(dramas_id), "movieId": dramas_id,
                            "rating": [1.0] * len(dramas_id)})
        # rate all the rest of the movies 3.0
        df3 = pd.DataFrame({"userId": [user_id] * len(remaining_id), "movieId": remaining_id,
                            "rating": [3.0] * len(remaining_id)})

        # concat old dataFrame with new ones
        rating = pd.concat([rating, df1, df2, df3])
        return rating

    def create_user_based_matrix(self, data):
        """*** YOUR CODE HERE ***"""
        self.ratings_df = data[0]
        self.movies_df = data[1]

        ratings_df = self.ratings_df
        movies_df = self.movies_df

        ratings_df = self.create_fake_user(ratings_df)
        self.ratings_df = ratings_df

        # save unique usersId and moviesId
        users_unique = ratings_df['userId'].unique()
        movies_unique = ratings_df['movieId'].unique()

        # save titles and moviesId
        titles_unique = movies_df['title'].values
        movieId_unique = movies_df['movieId'].values

        users_n = len(users_unique)
        movies_n = len(movies_unique)

        if not self.dicts_created:
            self.create_dictionaries(users_unique, movies_unique, titles_unique, movieId_unique, users_n, movies_n)
            self.dicts_created = True

        # initilize ratings matrix with nan values
        rating_matrix = np.empty((users_n, movies_n))
        rating_matrix[:] = np.nan

        # insert all the ratings from the DataFrame in the right cells
        for row in ratings_df.itertuples():
            normalized_user_id = self.users_id_dict[row[1]]
            normalized_movies_id = self.movies_id_dict[row[2]]
            rating = row[3]
            rating_matrix[normalized_user_id, normalized_movies_id] = rating

        mean_user_rating = np.nanmean(rating_matrix, axis=1).reshape(-1, 1)
        ratings_diff = (rating_matrix - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0

        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')

        # create the user based matrix
        self.user_based_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T

    def create_item_based_matrix(self, data):
        """*** YOUR CODE HERE ***"""
        self.ratings_df = data[0]
        self.movies_df = data[1]

        ratings_df = self.ratings_df
        movies_df = self.movies_df

        ratings_df = self.create_fake_user(ratings_df)
        self.ratings_df = ratings_df

        users_unique = ratings_df['userId'].unique()
        movies_unique = ratings_df['movieId'].unique()

        titles_unique = movies_df['title'].values
        movieId_unique = movies_df['movieId'].values

        users_n = len(users_unique)
        movies_n = len(movies_unique)

        if not self.dicts_created:
            self.create_dictionaries(users_unique, movies_unique, titles_unique, movieId_unique, users_n, movies_n)
            self.dicts_created = True

        rating_matrix = np.empty((users_n, movies_n))
        rating_matrix[:] = np.nan
        for row in ratings_df.itertuples():
            normalized_user_id = self.users_id_dict[row[1]]
            normalized_movies_id = self.movies_id_dict[row[2]]
            rating = row[3]
            rating_matrix[normalized_user_id, normalized_movies_id] = rating

        mean_user_rating = np.nanmean(rating_matrix, axis=1).reshape(-1, 1)
        ratings_diff = (rating_matrix - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0

        item_similarity = 1 - pairwise_distances(ratings_diff.T, metric='cosine')

        self.item_based_metrix = mean_user_rating + ratings_diff.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)])

    def predict_movies(self, user_id, k, is_user_based=False):
        """*** YOUR CODE HERE ***"""

        if is_user_based:
            ratings_matrix = self.user_based_matrix
        else:
            ratings_matrix = self.item_based_metrix

        ratings_df = self.ratings_df

        # save all movies Id's of the given user Id
        user_movies = ratings_df.loc[ratings_df['userId'] == int(user_id), 'movieId']
        user_movies = user_movies.values

        # save all the normalize moviesId
        user_movies = np.vectorize(self.movies_id_dict.get)(user_movies)

        # save the normlize user Id
        index = self.users_id_dict[int(user_id)]

        # get the correct row ub the rating_matrix
        user_ratings = np.take(ratings_matrix, [index], axis=0)

        # put zeroes in the indicies of this user movies, so they will not apear in the predictions.
        np.put(user_ratings, list(user_movies), 0.0)

        # find the movie Id of the top k movies
        max_k = np.argpartition(user_ratings[0], -k)[-k:]

        favorite_titles = []

        # map each movieId to it's corrusponding title name
        for movie_id in reversed(max_k):
            real_id = self.movie_real_id_dict[movie_id]
            title = self.movie_id_title[real_id]
            favorite_titles.append(title)
        return favorite_titles
