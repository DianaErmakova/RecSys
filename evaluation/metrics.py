import numpy as np
import logging


class Evaluator:
    def __init__(self, recommender, test_df, k=10):
        self.recommender = recommender
        self.test_df = test_df
        self.k = k
        self.logger = logging.getLogger(__name__)

        self.ground_truth = {
            user: set(group[group['rating'] >= 3.0]['movieId'])
            for user, group in self.test_df.groupby('userId')
        }

    def evaluate_user(self, user_id):
        try:
            recommendations = self.recommender.recommend(user_id, self.k)
            recs = set(recommendations['movieId']) if not recommendations.empty else set()
            relevant = self.ground_truth.get(user_id, set())

            precision = len(recs & relevant) / self.k if recs else 0
            recall = len(recs & relevant) / len(relevant) if relevant else 0
            coverage = 1 if not recommendations.empty else 0

            return {'precision': precision, 'recall': recall, 'coverage': coverage}
        except Exception as e:
            self.logger.error(f"Error evaluating user {user_id}: {e}")
            return {'precision': 0, 'recall': 0, 'coverage': 0}

    def evaluate(self, sample_size=100):
        users = list(self.ground_truth.keys())
        sampled_users = np.random.choice(users, size=min(sample_size, len(users)), replace=False)

        metrics = [self.evaluate_user(user) for user in sampled_users]

        return {
            'precision@k': np.mean([m['precision'] for m in metrics]),
            'recall@k': np.mean([m['recall'] for m in metrics]),
            'coverage': np.mean([m['coverage'] for m in metrics]),
        }
