import time
import logging
import pandas as pd
from recommender.content_based import ContentBasedRecommender
from evaluation.metrics import Evaluator


def load_data():
    ratings = pd.read_csv('data/rating.csv')
    ratings = ratings[ratings['rating'] >= 3.5]
    return ratings


def prepare_test_data(ratings):
    test_data = ratings.groupby('userId').filter(lambda x: len(x) >= 5)
    test_users = test_data['userId'].unique()[:500]
    return test_data[test_data['userId'].isin(test_users)]


def main():
    logging.basicConfig(level=logging.INFO)
    ratings = load_data()
    test_data = prepare_test_data(ratings)

    recommender = ContentBasedRecommender(
        "data/movie.csv",
        "data/rating.csv"
    )
    recommender.preprocess()
    recommender.build_similarity_matrix()

    evaluator = Evaluator(
        recommender=recommender,
        test_df=test_data,
        k=10
    )

    start = time.time()
    user_id = 1
    print("Рекомендации пользователя 1:")
    print(recommender.recommend(1))
    recs = set(recommender.recommend(1)['movieId'])
    relevant = evaluator.ground_truth.get(1, set())
    print("Пересечение рекомендаций и релевантных:", recs & relevant)

    results = evaluator.evaluate(sample_size=100)
    logging.info(f"Evaluation completed in {time.time() - start:.2f} seconds")

    print("\nFinal Metrics:")
    print(f"Precision@10: {results['precision@k']:.4f}")
    print(f"Recall@10: {results['recall@k']:.4f}")
    print(f"Coverage: {results['coverage']:.4f}")


if __name__ == "__main__":
    main()
