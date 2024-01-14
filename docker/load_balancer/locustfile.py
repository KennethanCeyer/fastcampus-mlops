from locust import HttpUser, between, task
import random


class WebsiteUser(HttpUser):
    host = "http://localhost:8000"
    wait_time = between(1, 5)

    @task
    def get_primes(self):
        end = random.randint(1000, 50000)
        start = random.randint(1, end // 2)
        self.client.get(f"/primes?start={start}&end={end}")
