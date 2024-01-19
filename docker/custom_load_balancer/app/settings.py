from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    conf_host: str = "0.0.0.0"
    conf_port: int = 8000
    conf_app_name: str = "default"


settings = Settings()
