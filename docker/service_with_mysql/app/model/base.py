from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from settings import settings


connect_url = URL.create(
    "mysql+pymysql",
    username=settings.conf_database_user,
    password=settings.conf_database_password,
    host=settings.conf_database_hostname,
    port=settings.conf_database_port,
    database=settings.conf_database_name,
)

DATABASE_URL = "mysql+pymysql://fast_campus:fast_campus123%@mysql/fast_campus"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
