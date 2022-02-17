# Aviv Ben Yaakov 206261695


import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes,
        # non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a
        # dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    """*** YOUR CODE HERE ***"""
    # users_n = len(data[0])
    ratings_df = data[0]

    # get all unique user rankings
    unique_users = ratings_df['userId'].value_counts()
    unique_users_count = unique_users.tolist()

    # get all unique movie rankings
    unique_movies = ratings_df['movieId'].value_counts()
    unique_movies_count = unique_movies.tolist()

    users_n = len(unique_users)
    movies_n = len(unique_movies)
    ratings_n = len(ratings_df)

    # print the number of unique users
    print(f'Unique users: {users_n}')

    # print the number of unique movies
    print(f'Unique movies: {movies_n}')

    # print the total number of rankings
    print(f'Total number of ratings is: {ratings_n}')

    # print the user who rated the most movies
    print(f'The user who rated the most movies is: {unique_users_count[0]}')
    # print the user who rated the least movies
    print(f'The user who rated the least movies is: {unique_users_count[-1]}')

    # print the movie that have been rated the most
    print(f'The movie that has been rated the most is: {unique_movies_count[0]}')
    # print the movie that have been rated the least
    print(f'The movie that has been rated the least is: {unique_movies_count[-1]}')


def plot_data(data, plot=True):
    """*** YOUR CODE HERE ***"""
    ratings_df = data[0]
    ratings = ratings_df['rating'].value_counts()
    count, rating = ratings.tolist(), ratings.keys().tolist()

    "display count per rating graph"
    sns.set_theme(style="whitegrid")
    sns.barplot(x=rating, y=count)
    plt.ticklabel_format(style='plain', axis='y')
    plt.savefig("ratings_distribution.png")
    if plot:
        plt.show()
