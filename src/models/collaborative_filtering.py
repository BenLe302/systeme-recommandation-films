#!/usr/bin/env python3
"""
Modèles de filtrage collaboratif pour le système de recommandation
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

# Surprise library pour les algorithmes de recommandation
from surprise import Dataset, Reader, SVD, NMF, KNNBasic, KNNWithMeans, KNNWithZScore
from surprise.model_selection import cross_validate, GridSearchCV
from surprise import accuracy
from surprise.trainset import Trainset
from surprise.prediction_set import PredictionSet

# Scikit-learn pour les métriques additionnelles
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
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
        self.user_encoder = None
        self.movie_encoder = None
        
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
            'movie_encoder': self.movie_encoder
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
        self.user_encoder = model_data.get('user_encoder')
        self.movie_encoder = model_data.get('movie_encoder')
        
        logger.info(f"Modèle chargé: {filepath}")

class SVDModel(BaseCollaborativeModel):
    """
    Modèle SVD (Singular Value Decomposition) pour le filtrage collaboratif
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres SVD
        svd_params = config.get('collaborative', {}).get('svd', {})
        self.n_factors = svd_params.get('n_factors', 100)
        self.n_epochs = svd_params.get('n_epochs', 20)
        self.lr_all = svd_params.get('lr_all', 0.005)
        self.reg_all = svd_params.get('reg_all', 0.02)
        
        # Initialiser le modèle SVD
        self.model = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            random_state=42
        )
        
        self.trainset = None
        self.reader = Reader(rating_scale=(0.5, 5.0))
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle SVD
        
        Args:
            train_data: Données d'entraînement avec colonnes userId, movieId, rating
        """
        logger.info("Entraînement du modèle SVD...")
        
        # Préparer les données pour Surprise
        dataset = Dataset.load_from_df(
            train_data[['userId', 'movieId', 'rating']], 
            self.reader
        )
        
        # Créer le trainset
        self.trainset = dataset.build_full_trainset()
        
        # Entraîner le modèle
        self.model.fit(self.trainset)
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
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
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
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Obtenir tous les films
        all_movies = self.trainset.all_items()
        
        # Obtenir les films déjà notés par l'utilisateur
        try:
            user_items = set([item for (item, _) in self.trainset.ur[self.trainset.to_inner_uid(user_id)]])
        except ValueError:
            # Utilisateur inconnu, recommander les films les plus populaires
            user_items = set()
        
        # Films non notés
        unrated_movies = [movie for movie in all_movies if movie not in user_items]
        
        # Prédire les notes pour les films non notés
        predictions = []
        for movie in unrated_movies:
            movie_id = self.trainset.to_raw_iid(movie)
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        # Trier par note prédite décroissante
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def get_similar_users(self, user_id: int, n_users: int = 10) -> List[Tuple[int, float]]:
        """
        Trouve les utilisateurs similaires
        
        Args:
            user_id: ID de l'utilisateur
            n_users: Nombre d'utilisateurs similaires
            
        Returns:
            Liste de tuples (user_id, similarité)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        
        # Obtenir le vecteur utilisateur
        user_vector = self.model.pu[inner_user_id]
        
        # Calculer la similarité avec tous les autres utilisateurs
        similarities = []
        for other_inner_id in range(self.trainset.n_users):
            if other_inner_id != inner_user_id:
                other_vector = self.model.pu[other_inner_id]
                similarity = 1 - cosine(user_vector, other_vector)
                other_user_id = self.trainset.to_raw_uid(other_inner_id)
                similarities.append((other_user_id, similarity))
        
        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n_users]

class NMFModel(BaseCollaborativeModel):
    """
    Modèle NMF (Non-negative Matrix Factorization) pour le filtrage collaboratif
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres NMF
        nmf_params = config.get('collaborative', {}).get('nmf', {})
        self.n_factors = nmf_params.get('n_factors', 50)
        self.n_epochs = nmf_params.get('n_epochs', 50)
        self.reg_pu = nmf_params.get('reg_pu', 0.06)
        self.reg_qi = nmf_params.get('reg_qi', 0.06)
        
        # Initialiser le modèle NMF
        self.model = NMF(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            random_state=42
        )
        
        self.trainset = None
        self.reader = Reader(rating_scale=(0.5, 5.0))
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle NMF
        
        Args:
            train_data: Données d'entraînement
        """
        logger.info("Entraînement du modèle NMF...")
        
        # Préparer les données pour Surprise
        dataset = Dataset.load_from_df(
            train_data[['userId', 'movieId', 'rating']], 
            self.reader
        )
        
        # Créer le trainset
        self.trainset = dataset.build_full_trainset()
        
        # Entraîner le modèle
        self.model.fit(self.trainset)
        self.is_trained = True
        
        logger.info("Modèle NMF entraîné avec succès")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note pour un utilisateur et un film
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Logique similaire à SVD
        all_movies = self.trainset.all_items()
        
        try:
            user_items = set([item for (item, _) in self.trainset.ur[self.trainset.to_inner_uid(user_id)]])
        except ValueError:
            user_items = set()
        
        unrated_movies = [movie for movie in all_movies if movie not in user_items]
        
        predictions = []
        for movie in unrated_movies:
            movie_id = self.trainset.to_raw_iid(movie)
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]

class KNNModel(BaseCollaborativeModel):
    """
    Modèle KNN (K-Nearest Neighbors) pour le filtrage collaboratif
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres KNN
        knn_params = config.get('collaborative', {}).get('knn', {})
        self.k = knn_params.get('k', 40)
        self.sim_options = {
            'name': knn_params.get('similarity', 'cosine'),
            'user_based': knn_params.get('user_based', True)
        }
        
        # Initialiser le modèle KNN
        self.model = KNNWithMeans(
            k=self.k,
            sim_options=self.sim_options,
            random_state=42
        )
        
        self.trainset = None
        self.reader = Reader(rating_scale=(0.5, 5.0))
    
    def fit(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne le modèle KNN
        
        Args:
            train_data: Données d'entraînement
        """
        logger.info("Entraînement du modèle KNN...")
        
        # Préparer les données pour Surprise
        dataset = Dataset.load_from_df(
            train_data[['userId', 'movieId', 'rating']], 
            self.reader
        )
        
        # Créer le trainset
        self.trainset = dataset.build_full_trainset()
        
        # Entraîner le modèle
        self.model.fit(self.trainset)
        self.is_trained = True
        
        logger.info("Modèle KNN entraîné avec succès")
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note pour un utilisateur et un film
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
        
        prediction = self.model.predict(user_id, movie_id)
        return prediction.est
    
    def recommend(self, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Logique similaire aux autres modèles
        all_movies = self.trainset.all_items()
        
        try:
            user_items = set([item for (item, _) in self.trainset.ur[self.trainset.to_inner_uid(user_id)]])
        except ValueError:
            user_items = set()
        
        unrated_movies = [movie for movie in all_movies if movie not in user_items]
        
        predictions = []
        for movie in unrated_movies:
            movie_id = self.trainset.to_raw_iid(movie)
            pred_rating = self.predict(user_id, movie_id)
            predictions.append((movie_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n_recommendations]
    
    def get_neighbors(self, user_id: int) -> List[Tuple[int, float]]:
        """
        Obtient les voisins les plus proches d'un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            
        Returns:
            Liste de tuples (neighbor_id, similarité)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        try:
            inner_user_id = self.trainset.to_inner_uid(user_id)
        except ValueError:
            return []
        
        # Obtenir les voisins depuis le modèle KNN
        neighbors = self.model.get_neighbors(inner_user_id, k=self.k)
        
        # Convertir en IDs externes avec similarités
        neighbor_list = []
        for neighbor_inner_id in neighbors:
            neighbor_user_id = self.trainset.to_raw_uid(neighbor_inner_id)
            # La similarité peut être obtenue depuis la matrice de similarité
            similarity = self.model.sim[inner_user_id, neighbor_inner_id]
            neighbor_list.append((neighbor_user_id, similarity))
        
        return neighbor_list

class CollaborativeFilteringManager:
    """
    Gestionnaire pour tous les modèles de filtrage collaboratif
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le gestionnaire
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {
            'svd': SVDModel(self.config),
            'nmf': NMFModel(self.config),
            'knn': KNNModel(self.config)
        }
        
        self.trained_models = {}
    
    def train_model(self, model_name: str, train_data: pd.DataFrame) -> None:
        """
        Entraîne un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('svd', 'nmf', 'knn')
            train_data: Données d'entraînement
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        logger.info(f"Entraînement du modèle {model_name}...")
        self.models[model_name].fit(train_data)
        self.trained_models[model_name] = self.models[model_name]
        logger.info(f"Modèle {model_name} entraîné avec succès")
    
    def train_all_models(self, train_data: pd.DataFrame) -> None:
        """
        Entraîne tous les modèles
        
        Args:
            train_data: Données d'entraînement
        """
        for model_name in self.models.keys():
            self.train_model(model_name, train_data)
    
    def get_recommendations(self, model_name: str, user_id: int, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Obtient des recommandations d'un modèle spécifique
        
        Args:
            model_name: Nom du modèle
            user_id: ID de l'utilisateur
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de recommandations
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].recommend(user_id, n_recommendations)
    
    def save_models(self, save_dir: str) -> None:
        """
        Sauvegarde tous les modèles entraînés
        
        Args:
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = save_path / f"{model_name}_model.pkl"
            model.save_model(str(filepath))
    
    def load_models(self, save_dir: str) -> None:
        """
        Charge tous les modèles sauvegardés
        
        Args:
            save_dir: Répertoire des modèles
        """
        save_path = Path(save_dir)
        
        for model_name in self.models.keys():
            filepath = save_path / f"{model_name}_model.pkl"
            if filepath.exists():
                self.models[model_name].load_model(str(filepath))
                self.trained_models[model_name] = self.models[model_name]
                logger.info(f"Modèle {model_name} chargé")

def main():
    """
    Fonction principale pour tester les modèles
    """
    # Exemple d'utilisation
    manager = CollaborativeFilteringManager()
    
    # Charger les données d'entraînement
    train_data = pd.read_csv("data/processed/train_data.csv")
    
    # Entraîner tous les modèles
    manager.train_all_models(train_data)
    
    # Obtenir des recommandations
    user_id = 1
    for model_name in ['svd', 'nmf', 'knn']:
        recommendations = manager.get_recommendations(model_name, user_id, 5)
        print(f"\nRecommandations {model_name} pour l'utilisateur {user_id}:")
        for movie_id, score in recommendations:
            print(f"  Film {movie_id}: {score:.3f}")
    
    # Sauvegarder les modèles
    manager.save_models("data/models")

if __name__ == "__main__":
    main()