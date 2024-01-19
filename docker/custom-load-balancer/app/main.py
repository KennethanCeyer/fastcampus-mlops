import uvicorn
from fastapi import FastAPI

from settings import settings

app = FastAPI()


@app.get("/")
def read_root():
    return {"app_name": settings.conf_app_name}


if __name__ == "__main__":
    uvicorn.run(app, host=settings.conf_host, port=settings.conf_port)
