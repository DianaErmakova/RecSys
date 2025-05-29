import pandas as pd


class PopularityRecommender:
    def __init__(self, movies_path, ratings_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.popular_movies = self._compute_popularity()

    def _compute_popularity(self, min_ratings=50):
        """Вычисление популярных фильмов по среднему рейтингу и количеству оценок."""
        movie_stats = self.ratings.groupby('movieId').agg(
            count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()

        popular = movie_stats[movie_stats['count'] >= min_ratings]
        popular_sorted = popular.sort_values(by='avg_rating', ascending=False)

        return pd.merge(popular_sorted, self.movies, on='movieId')[
            ['movieId', 'title', 'avg_rating', 'count']
        ]

    def recommend(self, top_n=10):
        """Рекомендация самых популярных фильмов."""
        return self.popular_movies.head(top_n)

    def update_popular_movies(self):
        """Обновление популярных фильмов (например, в продакшене)."""
        self.popular_movies = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
            .join(self.movies.set_index('movieId'), on='movieId')
            .sort_values(['count', 'avg_rating'], ascending=False)
            .head(10)[['title']]
            .reset_index(drop=True)
        )
