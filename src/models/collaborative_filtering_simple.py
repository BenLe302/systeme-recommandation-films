#!/usr/bin/env python3
"""
Modèles de filtrage collaboratif simplifiés (sans dépendance à 'surprise')
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

# Scikit-learn pour les algorithmes
from sklearn.decomposition import TruncatedSVD, NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseCollaborativeModel(ABC):
    """
    Classe de base pour tous les modèles de filtrage collaboratif
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le modèle de base
        
        Args:
            config: Configuration du modèle
        """
        self.config = config
        self.model = None
        self.is_trained = False
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        self.user_movie_matrix = None
        self.mean_rating = 0.0
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle
        
        Args:
            train_data: Données d'entraînement
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note pour un utilisateur et un film
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            
        Returns:
            Note prédite
        """
        pass
    
    @abstractmethod
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de tuples (movie_id, score)
        """
        pass
    
    def _prepare_data(self, train_data: pd.DataFrame) -> csr_matrix:
        """
        Prépare les données en créant la matrice utilisateur-film
        
        Args:
            train_data: Données d'entraînement
            
        Returns:
            Matrice sparse utilisateur-film
        """
        # Encoder les IDs
        train_data = train_data.copy()
        train_data['user_encoded'] = self.user_encoder.fit_transform(train_data['userId'])
        train_data['movie_encoded'] = self.movie_encoder.fit_transform(train_data['movieId'])
        
        # Calculer la moyenne des ratings
        self.mean_rating = train_data['rating'].mean()
        
        # Créer la matrice utilisateur-film
        n_users = len(self.user_encoder.classes_)
        n_movies = len(self.movie_encoder.classes_)
        
        user_movie_matrix = csr_matrix(
            (train_data['rating'], 
             (train_data['user_encoded'], train_data['movie_encoded'])),
            shape=(n_users, n_movies)
        )
        
        self.user_movie_matrix = user_movie_matrix
        return user_movie_matrix
    
    def save_model(self, filepath: str) -> None:
        """
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained,
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder,
            'user_movie_matrix': self.user_movie_matrix,
            'mean_rating': self.mean_rating,
            'user_factors': getattr(self, 'user_factors', None),
            'item_factors': getattr(self, 'item_factors', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modèle sauvegardé: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Charge un modèle sauvegardé
        
        Args:
            filepath: Chemin du modèle
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.user_encoder = model_data['user_encoder']
        self.movie_encoder = model_data['movie_encoder']
        self.user_movie_matrix = model_data['user_movie_matrix']
        self.mean_rating = model_data['mean_rating']
        
        # Charger les facteurs s'ils existent
        self.user_factors = model_data.get('user_factors', None)
        self.item_factors = model_data.get('item_factors', None)
        
        logger.info(f"Modèle chargé: {filepath}")

class SVDModel(BaseCollaborativeModel):
    """
    Modèle SVD simplifié utilisant TruncatedSVD de scikit-learn
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres SVD
        svd_params = config.get('collaborative', {}).get('svd', {})
        self.n_components = svd_params.get('n_factors', 50)
        
        # Initialiser le modèle SVD
        self.model = TruncatedSVD(
            n_components=self.n_components,
            random_state=42
        )
        
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle SVD
        
        Args:
            train_data: Données d'entraînement avec colonnes userId, movieId, rating
        """
        logger.info("Entraînement du modèle SVD simplifié...")
        
        # Préparer les données
        user_movie_matrix = self._prepare_data(train_data)
        
        # Entraîner le modèle SVD
        self.user_factors = self.model.fit_transform(user_movie_matrix)
        self.item_factors = self.model.components_.T
        
        self.is_trained = True
        logger.info("Modèle SVD entraîné avec succès")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note pour un utilisateur et un film
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            
        Returns:
            Note prédite
        """
        if not self.is_trained:
            return self.mean_rating
        
        try:
            # Encoder les IDs
            user_encoded = self.user_encoder.transform([user_id])[0]
            movie_encoded = self.movie_encoder.transform([movie_id])[0]
            
            # Prédiction
            prediction = np.dot(self.user_factors[user_encoded], self.item_factors[movie_encoded])
            
            # Normaliser entre 0.5 et 5.0
            prediction = max(0.5, min(5.0, prediction))
            
            return prediction
            
        except (ValueError, IndexError):
            # Utilisateur ou film non vu pendant l'entraînement
            return self.mean_rating
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de tuples (movie_id, score)
        """
        if not self.is_trained:
            return []
        
        try:
            # Encoder l'ID utilisateur
            user_encoded = self.user_encoder.transform([user_id])[0]
            
            # Calculer les scores pour tous les films
            user_vector = self.user_factors[user_encoded]
            scores = np.dot(user_vector, self.item_factors.T)
            
            # Obtenir les films déjà vus par l'utilisateur
            seen_movies = set(self.user_movie_matrix[user_encoded].nonzero()[1])
            
            # Créer la liste des recommandations
            recommendations = []
            for movie_encoded, score in enumerate(scores):
                if movie_encoded not in seen_movies:
                    movie_id = self.movie_encoder.inverse_transform([movie_encoded])[0]
                    recommendations.append((movie_id, float(score)))
            
            # Trier par score décroissant et retourner le top N
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except (ValueError, IndexError):
            return []

class NMFModel(BaseCollaborativeModel):
    """
    Modèle NMF (Non-negative Matrix Factorization)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres NMF
        nmf_params = config.get('collaborative', {}).get('nmf', {})
        self.n_components = nmf_params.get('n_factors', 50)
        
        # Initialiser le modèle NMF
        self.model = NMF(
            n_components=self.n_components,
            random_state=42,
            max_iter=200
        )
        
        self.user_factors = None
        self.item_factors = None
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle NMF
        
        Args:
            train_data: Données d'entraînement
        """
        logger.info("Entraînement du modèle NMF...")
        
        # Préparer les données
        user_movie_matrix = self._prepare_data(train_data)
        
        # Entraîner le modèle NMF
        self.user_factors = self.model.fit_transform(user_movie_matrix.toarray())
        self.item_factors = self.model.components_.T
        
        self.is_trained = True
        logger.info("Modèle NMF entraîné avec succès")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note pour un utilisateur et un film
        """
        if not self.is_trained:
            return self.mean_rating
        
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            movie_encoded = self.movie_encoder.transform([movie_id])[0]
            
            prediction = np.dot(self.user_factors[user_encoded], self.item_factors[movie_encoded])
            prediction = max(0.5, min(5.0, prediction))
            
            return prediction
            
        except (ValueError, IndexError):
            return self.mean_rating
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        """
        if not self.is_trained:
            return []
        
        try:
            user_encoded = self.user_encoder.transform([user_id])[0]
            user_vector = self.user_factors[user_encoded]
            scores = np.dot(user_vector, self.item_factors.T)
            
            seen_movies = set(self.user_movie_matrix[user_encoded].nonzero()[1])
            
            recommendations = []
            for movie_encoded, score in enumerate(scores):
                if movie_encoded not in seen_movies:
                    movie_id = self.movie_encoder.inverse_transform([movie_encoded])[0]
                    recommendations.append((movie_id, float(score)))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            return recommendations[:n_recommendations]
            
        except (ValueError, IndexError):
            return []

class CollaborativeFilteringManager:
    """
    Gestionnaire pour tous les modèles de filtrage collaboratif
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialise le gestionnaire
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            # Configuration par défaut
            self.config = {
                'collaborative': {
                    'svd': {'n_factors': 50},
                    'nmf': {'n_factors': 50}
                }
            }
        
        self.models = {
            'svd': SVDModel(self.config),
            'nmf': NMFModel(self.config)
        }
        
        self.trained_models = set()
    
    def train_model(self, model_name: str, train_data: pd.DataFrame) -> None:
        """
        Entraîne un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('svd', 'nmf')
            train_data: Données d'entraînement
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        logger.info(f"Entraînement du modèle {model_name}...")
        self.models[model_name].fit(train_data)
        self.trained_models.add(model_name)
        logger.info(f"Modèle {model_name} entraîné avec succès")
    
    def predict(self, model_name: str, user_id: int, movie_id: int) -> float:
        """
        Fait une prédiction avec un modèle spécifique
        
        Args:
            model_name: Nom du modèle
            user_id: ID de l'utilisateur
            movie_id: ID du film
            
        Returns:
            Note prédite
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        return self.models[model_name].predict(user_id, movie_id)
    
    def recommend(self, model_name: str, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Génère des recommandations avec un modèle spécifique
        
        Args:
            model_name: Nom du modèle
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de recommandations
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle non supporté: {model_name}")
        
        return self.models[model_name].recommend(user_id, n_recommendations)
    
    def save_models(self, models_dir: str) -> None:
        """
        Sauvegarde tous les modèles entraînés
        
        Args:
            models_dir: Répertoire de sauvegarde
        """
        models_path = Path(models_dir)
        models_path.mkdir(parents=True, exist_ok=True)
        
        for model_name in self.trained_models:
            filepath = models_path / f"{model_name}_model.pkl"
            self.models[model_name].save_model(str(filepath))
    
    def load_models(self, models_dir: str) -> None:
        """
        Charge tous les modèles sauvegardés
        
        Args:
            models_dir: Répertoire des modèles
        """
        models_path = Path(models_dir)
        
        for model_name in self.models.keys():
            filepath = models_path / f"{model_name}_model.pkl"
            if filepath.exists():
                self.models[model_name].load_model(str(filepath))
                self.trained_models.add(model_name)
                logger.info(f"Modèle {model_name} chargé avec succès")