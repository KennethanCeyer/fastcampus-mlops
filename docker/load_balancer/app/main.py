import uvicorn
from fastapi import FastAPI

from settings import settings

app = FastAPI()


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


@app.get("/primes", response_model=list[int])
def find_primes(start: int, end: int) -> list[int]:
    primes = [n for n in range(start, end + 1) if is_prime(n)]
    return primes


if __name__ == "__main__":
    uvicorn.run(app, host=settings.conf_host, port=settings.conf_port)
