from pydantic import BaseModel


class RatingRequest(BaseModel):
    userId: int
    movieId: int


class RatingResponse(BaseModel):
    userId: int
    movieId: int
    predictedRating: float
