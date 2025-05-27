import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def preprocess(self):
        self.movies['genres'] = self.movies['genres'].fillna('')
        tfidf = TfidfVectorizer(token_pattern=r'[^|]+')
        self.tfidf_matrix = tfidf.fit_transform(self.movies['genres'])

    def build_similarity_matrix(self):
        # cosine_similarity возвращает np.ndarray
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    def recommend(self, user_id, top_n=10):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        high_rated = user_ratings[user_ratings['rating'] >= 3.0]

        if high_rated.empty:
            return self.get_popular_movies(top_n)

        # Получаем индексы фильмов с высокими рейтингами
        movie_indices = []
        for m in high_rated['movieId']:
            idx_list = self.movies.index[self.movies['movieId'] == m].tolist()
            if idx_list:
                movie_indices.append(idx_list[0])

        if not movie_indices:
            return self.get_popular_movies(top_n)

        # Усредняем TF-IDF векторы
        user_profile = self.tfidf_matrix[movie_indices].mean(axis=0)

        # если это scipy sparse matrix
        if hasattr(user_profile, "toarray"):
            user_profile = user_profile.toarray()
        # если получился np.matrix — преобразуем в ndarray
        if isinstance(user_profile, np.matrix):
            user_profile = np.asarray(user_profile)

        similarities = cosine_similarity(user_profile, self.tfidf_matrix).flatten()

        scores = list(enumerate(similarities))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_n]

        recs = [{'movieId': self.movies.loc[idx, 'movieId'], 'score': score} for idx, score in top_scores]
        return pd.DataFrame(recs)

    def get_popular_movies(self, top_n=10):
        popular = (
            self.ratings.groupby('movieId')['rating']
            .count()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        return pd.DataFrame({'movieId': popular, 'score': [1.0]*len(popular)})

    def update_content_features(self, new_movies_df):
        self.movies = pd.concat([self.movies, new_movies_df], ignore_index=True)
        self.preprocess()
        self.build_similarity_matrix()
