from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    conf_host: str = "0.0.0.0"
    conf_port: int = 8000

    conf_database_hostname: str = "mysql"
    conf_database_port: str = "3306"
    conf_database_name: str = "mysql"
    conf_database_user: str = "user"
    conf_database_password: str = "password123%"


settings = Settings()
