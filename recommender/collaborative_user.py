import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
from sklearn.neighbors import NearestNeighbors
import logging
from tqdm import tqdm


class UserBasedRecommender:
    def __init__(self, movies_path, ratings_path, min_ratings=20, top_users=5000):
        self.movies = pd.read_csv(movies_path)
        self.ratings = self._preprocess_ratings(ratings_path, min_ratings)

        self.matrix = None
        self.user_mapper = None
        self.movie_mapper = None
        self.movie_ids = None
        self.model = None

        self._prepare_sparse_matrix(top_users)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _preprocess_ratings(self, path, min_ratings):
        """Фильтрация неактивных пользователей и непопулярных фильмов."""
        ratings = pd.read_csv(path)

        user_counts = ratings['userId'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        ratings = ratings[ratings['userId'].isin(valid_users)]

        movie_counts = ratings['movieId'].value_counts()
        valid_movies = movie_counts[movie_counts >= 10].index
        return ratings[ratings['movieId'].isin(valid_movies)]

    def _prepare_sparse_matrix(self, top_users):
        """Построение разреженной матрицы взаимодействий user-movie."""
        self.movie_ids = self.ratings['movieId'].unique()
        self.movie_mapper = {movie: idx for idx, movie in enumerate(self.movie_ids)}

        self.user_ids = self.ratings['userId'].unique()
        self.user_mapper = {user: idx for idx, user in enumerate(self.user_ids)}

        self.matrix = lil_matrix((len(self.user_ids), len(self.movie_ids)), dtype=np.float32)
        for user_id, group in tqdm(self.ratings.groupby('userId'), desc="Building matrix"):
            user_idx = self.user_mapper[user_id]
            movie_indices = [self.movie_mapper[m] for m in group['movieId']]
            self.matrix[user_idx, movie_indices] = group['rating'].values

        self.matrix = self.matrix.tocsr()
        self._train_knn(top_users)

    def _train_knn(self, top_users):
        """Обучение KNN модели по подмножеству пользователей."""
        sample_size = min(top_users, self.matrix.shape[0])
        indices = np.random.choice(self.matrix.shape[0], sample_size, replace=False)

        self.model = NearestNeighbors(
            metric='cosine',
            algorithm='brute',
            n_neighbors=20
        )
        self.model.fit(self.matrix[indices])

    def recommend(self, user_id, top_n=10):
        """Формирование рекомендаций на основе похожих пользователей."""
        try:
            if user_id not in self.user_mapper:
                return self._get_popular_movies(top_n)

            user_idx = self.user_mapper[user_id]
            distances, indices = self.model.kneighbors(self.matrix[user_idx])
            neighbor_indices = indices.flatten()

            scores = self.matrix[neighbor_indices].mean(axis=0).A1

            viewed = self.matrix[user_idx].indices
            scores[viewed] = -np.inf

            valid_indices = np.where(scores > 0)[0]
            if len(valid_indices) == 0:
                return self._get_popular_movies(top_n)

            top_indices = valid_indices[np.argsort(-scores[valid_indices])[:top_n]]

            recommendations = []
            for idx in top_indices:
                movie_id = self.movie_ids[idx]
                title = self.movies[self.movies['movieId'] == movie_id]['title'].values[0]
                recommendations.append({
                    'movieId': movie_id,
                    'title': title,
                    'score': scores[idx]
                })

            return pd.DataFrame(recommendations)

        except Exception as e:
            self.logger.error(f"Recommendation error: {str(e)}")
            return self._get_popular_movies(top_n)

    def _get_popular_movies(self, top_n):
        """Fallback: топ популярных фильмов."""
        return (
            self.ratings.groupby('movieId')
            .agg(rating_count=('userId', 'count'), avg_rating=('rating', 'mean'))
            .sort_values(['rating_count', 'avg_rating'], ascending=False)
            .head(top_n)
            .merge(self.movies, on='movieId')
            [['movieId', 'title', 'rating_count', 'avg_rating']]
        )
