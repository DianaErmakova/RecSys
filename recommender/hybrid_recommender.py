import pandas as pd
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative_user import UserBasedRecommender

class HybridRecommender:
    def __init__(self, movies_path, ratings_path):
        self.cb = ContentBasedRecommender(movies_path, ratings_path)
        self.cf = UserBasedRecommender(movies_path, ratings_path)
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        print("Ratings columns:", self.ratings.columns)  # отладка

        if 'movieId' not in self.ratings.columns:
            raise KeyError("ratings.csv должен содержать колонку 'movieId'")

    def get_popular_movies(self, top_n=10):
        print("Columns in ratings before grouping:", self.ratings.columns)  # отладка
        popular = (
            self.ratings.groupby('movieId')
            .agg(avg_rating=('rating', 'mean'), count=('rating', 'count'))
            .join(self.movies.set_index('movieId'), on='movieId')
            .sort_values(['count', 'avg_rating'], ascending=False)
            .head(top_n)[['title']]
            .reset_index(drop=True)
        )
        return popular

    def recommend(self, user_id, top_n=10, weights=(0.5, 0.4, 0.1)):
        try:
            cf_recs = self.cf.recommend(user_id, top_n=top_n)
            cb_recs = self.cb.recommend(user_id)
        except Exception as e:
            print(f"Error in CF or CB recommendation: {e}")
            cf_recs, cb_recs = pd.DataFrame(), []

        if cb_recs and isinstance(cb_recs, list):
            cb_df = pd.DataFrame(cb_recs, columns=["title", "score"])
        else:
            cb_df = pd.DataFrame(columns=["title", "score"])

        pop_df = self.get_popular_movies(top_n)
        pop_df['score'] = 1.0  # временный score, чтобы можно было объединить

        all_titles = pd.concat([
            cf_recs.assign(source='cf'),
            cb_df.assign(source='cb'),
            pop_df.assign(source='pop')
        ], ignore_index=True)

        score_map = {'cf': weights[0], 'cb': weights[1], 'pop': weights[2]}
        all_titles['score'] = all_titles['source'].map(score_map)

        hybrid_recs = (
            all_titles.groupby('title')['score']
            .sum()
            .reset_index()
            .sort_values(by='score', ascending=False)
            .head(top_n)
        )
        return hybrid_recs
