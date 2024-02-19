from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_path: str = "best_model.pth"
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    class Config:
        env_file = ".env"


settings = Settings()
