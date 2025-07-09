#!/usr/bin/env python3
"""
API FastAPI pour le système de recommandation de films
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# FastAPI et dépendances
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

# Importer nos modules
from src.models.collaborative_filtering_simple import CollaborativeFilteringManager
from models.content_based_filtering import ContentBasedManager
from models.hybrid_system import HybridSystemManager
from evaluation.metrics import ModelEvaluator
from utils.database import DatabaseManager
from utils.monitoring import MetricsCollector

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Modèles Pydantic pour l'API
class UserRating(BaseModel):
    """Modèle pour un rating utilisateur"""
    user_id: int = Field(..., description="ID de l'utilisateur")
    movie_id: int = Field(..., description="ID du film")
    rating: float = Field(..., ge=0.5, le=5.0, description="Note entre 0.5 et 5.0")
    timestamp: Optional[int] = Field(None, description="Timestamp du rating")

class RecommendationRequest(BaseModel):
    """Modèle pour une demande de recommandation"""
    user_id: int = Field(..., description="ID de l'utilisateur")
    n_recommendations: int = Field(10, ge=1, le=100, description="Nombre de recommandations")
    model_type: str = Field("hybrid_weighted", description="Type de modèle à utiliser")
    exclude_seen: bool = Field(True, description="Exclure les films déjà vus")
    
    @validator('model_type')
    def validate_model_type(cls, v):
        allowed_types = ['collaborative_svd', 'collaborative_nmf', 'collaborative_knn', 
                        'content_tfidf', 'content_genre', 'hybrid_weighted', 'hybrid_switching']
        if v not in allowed_types:
            raise ValueError(f'Type de modèle non supporté. Types disponibles: {allowed_types}')
        return v

class MovieInfo(BaseModel):
    """Modèle pour les informations d'un film"""
    movie_id: int
    title: str
    genres: List[str]
    year: Optional[int] = None
    average_rating: Optional[float] = None
    rating_count: Optional[int] = None

class RecommendationResponse(BaseModel):
    """Modèle pour la réponse de recommandation"""
    user_id: int
    recommendations: List[Dict[str, Any]]
    model_used: str
    timestamp: str
    execution_time_ms: float

class PredictionRequest(BaseModel):
    """Modèle pour une demande de prédiction"""
    user_id: int = Field(..., description="ID de l'utilisateur")
    movie_id: int = Field(..., description="ID du film")
    model_type: str = Field("hybrid_weighted", description="Type de modèle à utiliser")

class PredictionResponse(BaseModel):
    """Modèle pour la réponse de prédiction"""
    user_id: int
    movie_id: int
    predicted_rating: float
    model_used: str
    timestamp: str
    execution_time_ms: float

class HealthResponse(BaseModel):
    """Modèle pour la réponse de santé"""
    status: str
    timestamp: str
    models_loaded: Dict[str, bool]
    database_connected: bool
    uptime_seconds: float

# Variables globales pour les modèles
models = {
    'collaborative': None,
    'content': None,
    'hybrid': None
}

# Données en mémoire
data_cache = {
    'movies': None,
    'ratings': None,
    'user_profiles': None
}

# Configuration et utilitaires
config = None
db_manager = None
metrics_collector = None
app_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de cycle de vie de l'application"""
    global models, data_cache, config, db_manager, metrics_collector, app_start_time
    
    # Démarrage
    logger.info("Démarrage de l'API de recommandation...")
    app_start_time = datetime.now()
    
    try:
        # Charger la configuration
        config_path = Path("config/config.yaml")
        if not config_path.exists():
            config_path = Path("../../config/config.yaml")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        # Initialiser les composants
        await initialize_components()
        
        logger.info("API de recommandation démarrée avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage: {e}")
        raise
    
    yield
    
    # Arrêt
    logger.info("Arrêt de l'API de recommandation...")
    if db_manager:
        await db_manager.close()

async def initialize_components():
    """Initialise tous les composants de l'API"""
    global models, data_cache, db_manager, metrics_collector
    
    try:
        # Initialiser le gestionnaire de base de données
        db_manager = DatabaseManager(config)
        await db_manager.initialize()
        
        # Initialiser le collecteur de métriques
        metrics_collector = MetricsCollector(config)
        
        # Charger les modèles
        await load_models()
        
        # Charger les données en cache
        await load_data_cache()
        
        logger.info("Tous les composants initialisés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        raise

async def load_models():
    """Charge tous les modèles de recommandation"""
    global models
    
    try:
        models_dir = Path(config['data']['models_dir'])
        
        # Charger les modèles collaboratifs
        logger.info("Chargement des modèles collaboratifs...")
        models['collaborative'] = CollaborativeFilteringManager()
        collab_path = models_dir / "collaborative"
        if collab_path.exists():
            models['collaborative'].load_models(str(collab_path))
        
        # Charger les modèles basés sur le contenu
        logger.info("Chargement des modèles de contenu...")
        models['content'] = ContentBasedManager()
        content_path = models_dir / "content"
        if content_path.exists():
            models['content'].load_models(str(content_path))
        
        # Charger les modèles hybrides
        logger.info("Chargement des modèles hybrides...")
        models['hybrid'] = HybridSystemManager()
        hybrid_path = models_dir / "hybrid"
        if hybrid_path.exists():
            models['hybrid'].load_models(str(hybrid_path))
        
        logger.info("Modèles chargés avec succès")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des modèles: {e}")
        # Initialiser des modèles vides pour éviter les erreurs
        models['collaborative'] = CollaborativeFilteringManager()
        models['content'] = ContentBasedManager()
        models['hybrid'] = HybridSystemManager()

async def load_data_cache():
    """Charge les données en cache pour améliorer les performances"""
    global data_cache
    
    try:
        data_dir = Path(config['data']['processed_dir'])
        
        # Charger les données des films
        movies_path = data_dir / "movies_processed.csv"
        if movies_path.exists():
            data_cache['movies'] = pd.read_csv(movies_path)
            logger.info(f"Chargé {len(data_cache['movies'])} films")
        
        # Charger les données de ratings (échantillon pour la performance)
        ratings_path = data_dir / "train_data.csv"
        if ratings_path.exists():
            # Charger seulement un échantillon pour la mémoire
            data_cache['ratings'] = pd.read_csv(ratings_path).sample(n=min(100000, len(pd.read_csv(ratings_path))))
            logger.info(f"Chargé {len(data_cache['ratings'])} ratings en cache")
        
        # Créer des profils utilisateurs simplifiés
        if data_cache['ratings'] is not None:
            user_stats = data_cache['ratings'].groupby('userId').agg({
                'rating': ['mean', 'count'],
                'movieId': 'nunique'
            }).round(2)
            user_stats.columns = ['avg_rating', 'rating_count', 'unique_movies']
            data_cache['user_profiles'] = user_stats.to_dict('index')
            logger.info(f"Créé {len(data_cache['user_profiles'])} profils utilisateurs")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du cache: {e}")
        # Initialiser des caches vides
        data_cache = {'movies': pd.DataFrame(), 'ratings': pd.DataFrame(), 'user_profiles': {}}

# Créer l'application FastAPI
app = FastAPI(
    title="API de Recommandation de Films",
    description="API pour obtenir des recommandations de films personnalisées",
    version="1.0.0",
    lifespan=lifespan
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sécurité (optionnelle)
security = HTTPBearer(auto_error=False)

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Authentification optionnelle"""
    # Implémentation simple - en production, utiliser JWT ou OAuth
    if credentials and credentials.credentials == "demo-token":
        return {"user_id": "demo", "permissions": ["read", "write"]}
    return None

# Routes de l'API

@app.get("/", response_model=Dict[str, str])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API de Recommandation de Films",
        "version": "1.0.0",
        "author": "Dady Akrou Cyrille",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de l'état de santé de l'API"""
    global app_start_time
    
    # Vérifier l'état des modèles
    models_status = {
        'collaborative': models['collaborative'] is not None and len(models['collaborative'].trained_models) > 0,
        'content': models['content'] is not None and len(models['content'].trained_models) > 0,
        'hybrid': models['hybrid'] is not None and len(models['hybrid'].trained_models) > 0
    }
    
    # Vérifier la connexion à la base de données
    db_connected = db_manager is not None and await db_manager.is_connected()
    
    # Calculer l'uptime
    uptime = (datetime.now() - app_start_time).total_seconds() if app_start_time else 0
    
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=models_status,
        database_connected=db_connected,
        uptime_seconds=uptime
    )

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Obtenir des recommandations pour un utilisateur"""
    start_time = datetime.now()
    
    try:
        # Enregistrer la métrique de requête
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_request, "recommendations", request.model_type)
        
        # Déterminer le modèle à utiliser
        model_manager, model_name = _get_model_manager(request.model_type)
        
        if model_manager is None:
            raise HTTPException(status_code=503, detail=f"Modèle {request.model_type} non disponible")
        
        # Obtenir les recommandations
        if request.model_type.startswith('hybrid'):
            recommendations = model_manager.get_recommendations(
                model_name, request.user_id, data_cache['ratings'], request.n_recommendations
            )
        elif request.model_type.startswith('collaborative'):
            recommendations = model_manager.get_recommendations(
                model_name, request.user_id, request.n_recommendations
            )
        elif request.model_type.startswith('content'):
            recommendations = model_manager.get_user_recommendations(
                model_name, request.user_id, data_cache['ratings'], request.n_recommendations
            )
        else:
            raise HTTPException(status_code=400, detail="Type de modèle non reconnu")
        
        # Enrichir avec les informations des films
        enriched_recommendations = await _enrich_recommendations(recommendations, request.exclude_seen, request.user_id)
        
        # Calculer le temps d'exécution
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Enregistrer les métriques de performance
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_latency, "recommendations", execution_time)
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=enriched_recommendations,
            model_used=request.model_type,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la génération de recommandations: {e}")
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_error, "recommendations", str(e))
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.post("/predict", response_model=PredictionResponse)
async def predict_rating(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Prédire la note qu'un utilisateur donnerait à un film"""
    start_time = datetime.now()
    
    try:
        # Déterminer le modèle à utiliser
        model_manager, model_name = _get_model_manager(request.model_type)
        
        if model_manager is None:
            raise HTTPException(status_code=503, detail=f"Modèle {request.model_type} non disponible")
        
        # Faire la prédiction
        if request.model_type.startswith('hybrid'):
            predicted_rating = model_manager.trained_models[model_name].predict(request.user_id, request.movie_id)
        elif request.model_type.startswith('collaborative'):
            predicted_rating = model_manager.trained_models[model_name].predict(request.user_id, request.movie_id)
        else:
            # Pour les modèles de contenu, utiliser une approximation
            predicted_rating = 3.5  # Note moyenne par défaut
        
        # Calculer le temps d'exécution
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Enregistrer les métriques
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_request, "prediction", request.model_type)
            background_tasks.add_task(metrics_collector.record_latency, "prediction", execution_time)
        
        return PredictionResponse(
            user_id=request.user_id,
            movie_id=request.movie_id,
            predicted_rating=float(predicted_rating),
            model_used=request.model_type,
            timestamp=datetime.now().isoformat(),
            execution_time_ms=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_error, "prediction", str(e))
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.post("/ratings")
async def add_rating(
    rating: UserRating,
    background_tasks: BackgroundTasks,
    current_user: Optional[Dict] = Depends(get_current_user)
):
    """Ajouter un nouveau rating utilisateur"""
    try:
        # Valider les données
        if rating.timestamp is None:
            rating.timestamp = int(datetime.now().timestamp())
        
        # Sauvegarder en base de données
        if db_manager:
            await db_manager.save_rating(rating.dict())
        
        # Mettre à jour le cache si nécessaire
        background_tasks.add_task(_update_cache_with_rating, rating)
        
        # Enregistrer la métrique
        if metrics_collector:
            background_tasks.add_task(metrics_collector.record_request, "add_rating", "user_feedback")
        
        return {"message": "Rating ajouté avec succès", "rating_id": f"{rating.user_id}_{rating.movie_id}"}
        
    except Exception as e:
        logger.error(f"Erreur lors de l'ajout du rating: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de l'ajout du rating")

@app.get("/movies/{movie_id}", response_model=MovieInfo)
async def get_movie_info(movie_id: int):
    """Obtenir les informations d'un film"""
    try:
        if data_cache['movies'] is None or data_cache['movies'].empty:
            raise HTTPException(status_code=503, detail="Données des films non disponibles")
        
        movie_data = data_cache['movies'][data_cache['movies']['movieId'] == movie_id]
        
        if movie_data.empty:
            raise HTTPException(status_code=404, detail="Film non trouvé")
        
        movie = movie_data.iloc[0]
        
        # Calculer les statistiques de rating
        movie_ratings = data_cache['ratings'][data_cache['ratings']['movieId'] == movie_id] if data_cache['ratings'] is not None else pd.DataFrame()
        
        return MovieInfo(
            movie_id=int(movie['movieId']),
            title=movie['title'],
            genres=movie['genres'].split('|') if pd.notna(movie['genres']) else [],
            year=int(movie['year']) if 'year' in movie and pd.notna(movie['year']) else None,
            average_rating=float(movie_ratings['rating'].mean()) if not movie_ratings.empty else None,
            rating_count=len(movie_ratings) if not movie_ratings.empty else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du film {movie_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: int):
    """Obtenir le profil d'un utilisateur"""
    try:
        if data_cache['user_profiles'] is None:
            raise HTTPException(status_code=503, detail="Profils utilisateurs non disponibles")
        
        profile = data_cache['user_profiles'].get(user_id)
        
        if profile is None:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
        
        return {
            "user_id": user_id,
            "average_rating": profile['avg_rating'],
            "total_ratings": profile['rating_count'],
            "unique_movies_rated": profile['unique_movies']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du profil utilisateur {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/models")
async def list_available_models():
    """Lister tous les modèles disponibles"""
    available_models = []
    
    # Modèles collaboratifs
    if models['collaborative'] and models['collaborative'].trained_models:
        for model_name in models['collaborative'].trained_models.keys():
            available_models.append(f"collaborative_{model_name}")
    
    # Modèles de contenu
    if models['content'] and models['content'].trained_models:
        for model_name in models['content'].trained_models.keys():
            available_models.append(f"content_{model_name}")
    
    # Modèles hybrides
    if models['hybrid'] and models['hybrid'].trained_models:
        for model_name in models['hybrid'].trained_models.keys():
            available_models.append(f"hybrid_{model_name}")
    
    return {
        "available_models": available_models,
        "total_models": len(available_models),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def get_metrics(current_user: Optional[Dict] = Depends(get_current_user)):
    """Obtenir les métriques de performance de l'API"""
    if metrics_collector:
        return await metrics_collector.get_metrics()
    else:
        return {"message": "Collecteur de métriques non disponible"}

# Fonctions utilitaires

def _get_model_manager(model_type: str):
    """Obtient le gestionnaire de modèle approprié"""
    if model_type.startswith('collaborative_'):
        model_name = model_type.replace('collaborative_', '')
        return models['collaborative'], model_name
    elif model_type.startswith('content_'):
        model_name = model_type.replace('content_', '')
        return models['content'], model_name
    elif model_type.startswith('hybrid_'):
        model_name = model_type.replace('hybrid_', '')
        return models['hybrid'], model_name
    else:
        return None, None

async def _enrich_recommendations(recommendations: List[Tuple[int, float]], exclude_seen: bool, user_id: int) -> List[Dict[str, Any]]:
    """Enrichit les recommandations avec les informations des films"""
    enriched = []
    
    # Obtenir les films déjà vus par l'utilisateur si nécessaire
    seen_movies = set()
    if exclude_seen and data_cache['ratings'] is not None:
        user_ratings = data_cache['ratings'][data_cache['ratings']['userId'] == user_id]
        seen_movies = set(user_ratings['movieId'].tolist())
    
    for movie_id, score in recommendations:
        # Exclure les films déjà vus si demandé
        if exclude_seen and movie_id in seen_movies:
            continue
        
        # Obtenir les informations du film
        movie_info = {"movie_id": movie_id, "title": f"Film {movie_id}", "genres": []}
        
        if data_cache['movies'] is not None and not data_cache['movies'].empty:
            movie_data = data_cache['movies'][data_cache['movies']['movieId'] == movie_id]
            if not movie_data.empty:
                movie = movie_data.iloc[0]
                movie_info = {
                    "movie_id": movie_id,
                    "title": movie['title'],
                    "genres": movie['genres'].split('|') if pd.notna(movie['genres']) else [],
                    "year": int(movie['year']) if 'year' in movie and pd.notna(movie['year']) else None
                }
        
        movie_info["recommendation_score"] = float(score)
        enriched.append(movie_info)
    
    return enriched

async def _update_cache_with_rating(rating: UserRating):
    """Met à jour le cache avec un nouveau rating"""
    try:
        if data_cache['ratings'] is not None:
            # Ajouter le nouveau rating au cache (simplifié)
            new_row = pd.DataFrame([{
                'userId': rating.user_id,
                'movieId': rating.movie_id,
                'rating': rating.rating,
                'timestamp': rating.timestamp
            }])
            data_cache['ratings'] = pd.concat([data_cache['ratings'], new_row], ignore_index=True)
            
            # Mettre à jour le profil utilisateur
            if rating.user_id in data_cache['user_profiles']:
                user_ratings = data_cache['ratings'][data_cache['ratings']['userId'] == rating.user_id]
                data_cache['user_profiles'][rating.user_id] = {
                    'avg_rating': user_ratings['rating'].mean(),
                    'rating_count': len(user_ratings),
                    'unique_movies': user_ratings['movieId'].nunique()
                }
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du cache: {e}")

if __name__ == "__main__":
    # Configuration pour le développement
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )