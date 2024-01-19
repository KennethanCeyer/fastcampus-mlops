from model.models import User, UserCreate, UserOut
from fastapi import FastAPI, Depends, HTTPException

from model.base import SessionLocal, engine
from model.base import Base


app = FastAPI()


def get_db() -> SessionLocal:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/", response_model=UserOut)
def create_user(user: UserCreate, db: SessionLocal = Depends(get_db)) -> UserOut:
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@app.get("/users/", response_model=list[UserOut])
def read_users(
    skip: int = 0, limit: int = 10, db: SessionLocal = Depends(get_db)
) -> list[UserOut]:
    users = db.query(User).offset(skip).limit(limit).all()
    return users


@app.get("/users/{user_id}", response_model=UserOut)
def read_user(user_id: int, db: SessionLocal = Depends(get_db)) -> UserOut:
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: int, user_update: UserCreate, db: SessionLocal = Depends(get_db)
) -> UserOut:
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.name = user_update.name
    db_user.email = user_update.email
    db.commit()
    db.refresh(db_user)
    return db_user


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int, db: SessionLocal = Depends(get_db)) -> None:
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(db_user)
    db.commit()


if __name__ == "__main__":
    import uvicorn

    Base.metadata.create_all(bind=engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
