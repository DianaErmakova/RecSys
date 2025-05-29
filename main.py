import time
import logging
import pandas as pd

from recommender.content_based import ContentBasedRecommender
from recommender.collaborative_user import UserBasedRecommender
from recommender.popularity_based import PopularityRecommender
from recommender.hybrid_recommender import HybridRecommender
from evaluation.metrics import Evaluator


def load_data():
    ratings = pd.read_csv('data/rating.csv')
    return ratings[ratings['rating'] >= 3.5]


def prepare_test_data(ratings):
    test_data = ratings.groupby('userId').filter(lambda x: len(x) >= 5)
    test_users = test_data['userId'].unique()[:500]
    return test_data[test_data['userId'].isin(test_users)]

def demo_content_based():
    print("=" * 50)
    print("Content-Based рекомендации")

    cb_recommender = ContentBasedRecommender("data/movie.csv", "data/rating.csv")
    cb_recommender.preprocess()
    cb_recommender.build_similarity_matrix()

    user_id = 1
    cb_recommendations = cb_recommender.recommend(user_id)

    print(f"Рекомендации на основе контента для пользователя {user_id}:")
    for _, row in cb_recommendations.iterrows():
        print(f"{row['movieId']} (схожесть: {row['score']:.3f})")

    cb_recommendations.to_csv("user1_cb_recommendations.csv", index=False)


def demo_user_based():
    print("\n" + "=" * 50)
    print("User-Based Collaborative Filtering")

    ub_recommender = UserBasedRecommender("data/movie.csv", "data/rating.csv")

    for uid in [1, 5, 10]:
        print(f"\nРекомендации на основе похожих пользователей для user_id={uid}:")
        ub_recommendations = ub_recommender.recommend(user_id=uid)
        print(ub_recommendations)
        ub_recommendations.to_csv(f"user{uid}_userbased_recommendations.csv", index=False)


def demo_popularity_based():
    print("\n" + "=" * 50)
    print("Популярные фильмы (Heuristic-based)")

    pop_rec = PopularityRecommender("data/movie.csv", "data/rating.csv")
    top_movies = pop_rec.recommend(top_n=10)
    print(top_movies)

    top_movies.to_csv("top_popular_movies.csv", index=False)

def demo_hybrid():
    print("\n" + "=" * 50)
    print("Гибридные рекомендации")

    hybrid = HybridRecommender("data/movie.csv", "data/rating.csv")
    for uid in [1, 5, 1000]:  # включая нового пользователя
        print(f"\nРекомендации для пользователя {uid}:")
        print(hybrid.recommend(user_id=uid))


def evaluate_content_based():
    logging.basicConfig(level=logging.INFO)

    ratings = load_data()
    test_data = prepare_test_data(ratings)

    recommender = ContentBasedRecommender("data/movie.csv", "data/rating.csv")
    recommender.preprocess()
    recommender.build_similarity_matrix()

    evaluator = Evaluator(recommender=recommender, test_df=test_data, k=10)

    start = time.time()
    user_id = 1
    print("Рекомендации пользователя 1:")
    recs_df = recommender.recommend(user_id)
    print(recs_df)

    recs = set(recs_df['movieId'])
    relevant = evaluator.ground_truth.get(user_id, set())
    print("Пересечение рекомендаций и релевантных:", recs & relevant)

    results = evaluator.evaluate(sample_size=100)
    logging.info(f"Evaluation completed in {time.time() - start:.2f} seconds")

    print("\nFinal Metrics:")
    print(f"Precision@10: {results['precision@k']:.4f}")
    print(f"Recall@10: {results['recall@k']:.4f}")
    print(f"Coverage: {results['coverage']:.4f}")


def main():
    demo_content_based()
    demo_user_based()
    demo_popularity_based()
    demo_hybrid()
    evaluate_content_based()


if __name__ == "__main__":
    main()
