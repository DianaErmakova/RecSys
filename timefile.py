'''
from recommender.content_based import ContentBasedRecommender
from recommender.collaborative_user import UserBasedRecommender
from recommender.popularity_based import PopularityRecommender
from recommender.hybrid_recommender import HybridRecommender
from evaluation.metrics import Evaluator
from sklearn.model_selection import train_test_split
import pandas as pd

# Content-based
print("=" * 50)
print("Content-Based рекомендации")

cb_recommender = ContentBasedRecommender("data/movie.csv", "data/rating.csv")
cb_recommender.preprocess()
cb_recommender.build_similarity_matrix()

user_id = 1
cb_recommendations = cb_recommender.recommend(user_id)

print(f"Рекомендации на основе контента для пользователя {user_id}:")
for title, score in cb_recommendations:
    print(f"{title} (схожесть: {score:.3f})")

# Сохраняем в CSV
pd.DataFrame(cb_recommendations, columns=["title", "similarity"]).to_csv("user1_cb_recommendations.csv", index=False)

# Collaborative Filtering (User-based)
print("\n" + "=" * 50)
print("User-Based Collaborative Filtering")

ub_recommender = UserBasedRecommender("data/movie.csv", "data/rating.csv")

for uid in [1, 5, 10]:
    print(f"\nРекомендации на основе похожих пользователей для user_id={uid}:")
    ub_recommendations = ub_recommender.recommend(user_id=uid)
    print(ub_recommendations)

    # Дополнительно: можно сохранить в отдельные CSV, если надо
    ub_recommendations.to_csv(f"user{uid}_userbased_recommendations.csv", index=False)

print("\n" + "=" * 50)
print("Популярные фильмы (Heuristic-based)")

pop_rec = PopularityRecommender("data/movie.csv", "data/rating.csv")
top_movies = pop_rec.recommend(top_n=10)
print(top_movies)

# Сохраняем в файл
top_movies.to_csv("top_popular_movies.csv", index=False)
print("\n" + "=" * 50)
print("Гибридные рекомендации")

hybrid = HybridRecommender("data/movie.csv", "data/rating.csv")
for uid in [1, 5, 1000]:  # включая нового пользователя
    print(f"\nРекомендации для пользователя {uid}:")
    print(hybrid.recommend(user_id=uid))
'''

