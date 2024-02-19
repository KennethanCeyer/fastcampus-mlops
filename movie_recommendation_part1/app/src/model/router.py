import torch
from fastapi import APIRouter

from model.model import model
from model.schemas import RatingRequest
from model.schemas import RatingResponse

router = APIRouter()


@router.post("/predict/", response_model=RatingResponse)
async def predict_rating(request: RatingRequest) -> RatingResponse:
    user_movie_tensor = torch.tensor(
        [[request.userId - 1, request.movieId - 1]], dtype=torch.int32
    )
    with torch.no_grad():
        predicted_rating = model(user_movie_tensor).item() * 5
    rounded_rating = round(predicted_rating * 2) / 2
    return RatingResponse(
        userId=request.userId, movieId=request.movieId, predictedRating=rounded_rating
    )
