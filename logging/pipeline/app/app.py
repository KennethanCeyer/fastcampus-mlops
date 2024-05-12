import logging
import time
import random
import os
from logging.handlers import RotatingFileHandler

log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)

log_file_path = f"{log_directory}/recommendations.log"
logger = logging.getLogger("RecommendationLogger")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())


def generate_recommendations() -> list[tuple[str, float]]:
    items = [f"item_{i}" for i in range(100)]
    recommended_items = random.sample(items, 5)
    scores = [round(random.random(), 2) for _ in range(5)]
    return list(zip(recommended_items, scores))


def get_random_user_id() -> int:
    return random.randint(1, 1000)


if __name__ == "__main__":
    while True:
        user_id = get_random_user_id()
        recommendations = generate_recommendations()

        for item, score in recommendations:
            logger.info(f"User {user_id}: Recommended item {item} with score {score}")

        time.sleep(1)
