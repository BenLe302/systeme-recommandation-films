#!/usr/bin/env python3
"""
API mock simplifiée pour les tests
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Any
from datetime import datetime
import uvicorn
import random

# Modèles Pydantic
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="ID de l'utilisateur")
    n_recommendations: int = Field(10, ge=1, le=100, description="Nombre de recommandations")
    model_type: str = Field("svd", description="Type de modèle à utiliser")
    exclude_seen: bool = Field(True, description="Exclure les films déjà vus")

class PredictionRequest(BaseModel):
    user_id: int = Field(..., description="ID de l'utilisateur")
    movie_id: int = Field(..., description="ID du film")
    model_type: str = Field("svd", description="Type de modèle à utiliser")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    uptime_seconds: float

# Créer l'application FastAPI
app = FastAPI(
    title="API de Test Mock",
    description="API mock pour les tests",
    version="1.0.0"
)

app_start_time = datetime.now()

@app.get("/")
async def root():
    return {
        "message": "API de Test Mock",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de santé de l'API"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded={"svd": True, "nmf": True},
        uptime_seconds=uptime
    )

@app.post("/recommendations")
async def get_recommendations(request: RecommendationRequest):
    """Retourne des recommandations mock"""
    
    # Générer des recommandations fictives
    recommendations = []
    for i in range(request.n_recommendations):
        movie_id = random.randint(1, 1000)
        score = round(random.uniform(3.0, 5.0), 2)
        recommendations.append({
            "movie_id": movie_id,
            "predicted_rating": score,
            "title": f"Film {movie_id}",
            "genres": "Action|Drama"
        })
    
    return {
        "user_id": request.user_id,
        "recommendations": recommendations,
        "model_used": request.model_type,
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": 50.0
    }

@app.post("/predict")
async def predict_rating(request: PredictionRequest):
    """Retourne une prédiction mock"""
    
    # Générer une prédiction fictive
    predicted_rating = round(random.uniform(2.5, 5.0), 2)
    
    return {
        "user_id": request.user_id,
        "movie_id": request.movie_id,
        "predicted_rating": predicted_rating,
        "model_used": request.model_type,
        "timestamp": datetime.now().isoformat(),
        "execution_time_ms": 25.0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)