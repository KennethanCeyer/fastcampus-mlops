from fastapi import FastAPI
from settings import settings
from model.router import router as model_router


app = FastAPI()
app.include_router(model_router)


@app.get("/")
async def get_health() -> dict[str, str]:
    return {"message": "new version ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.server_host, port=settings.server_port)
