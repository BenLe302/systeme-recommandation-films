#!/usr/bin/env python3
"""
Système de recommandation hybride combinant filtrage collaboratif et basé sur le contenu
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import pandas as pd
import numpy as np
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod

# Importer nos modèles
from .collaborative_filtering_simple import CollaborativeFilteringManager
from .content_based_filtering import ContentBasedManager

# Scikit-learn pour les métriques et l'optimisation
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseHybridModel(ABC):
    """
    Classe de base pour tous les modèles hybrides
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le modèle hybride de base
        
        Args:
            config: Configuration du modèle
        """
        self.config = config
        self.collaborative_manager = None
        self.content_manager = None
        self.is_trained = False
        self.weights = None
        
    @abstractmethod
    def fit(self, train_data: pd.DataFrame, movies_data: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle hybride
        
        Args:
            train_data: Données d'entraînement
            movies_data: Données des films
            val_data: Données de validation (optionnel)
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
    def recommend(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de tuples (movie_id, score)
        """
        pass
    
    def save_model(self, filepath: str) -> None:
        """
        Sauvegarde le modèle hybride
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'config': self.config,
            'is_trained': self.is_trained,
            'weights': self.weights,
            'collaborative_manager': self.collaborative_manager,
            'content_manager': self.content_manager
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modèle hybride sauvegardé: {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Charge un modèle hybride sauvegardé
        
        Args:
            filepath: Chemin du modèle
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.is_trained = model_data['is_trained']
        self.weights = model_data['weights']
        self.collaborative_manager = model_data['collaborative_manager']
        self.content_manager = model_data['content_manager']
        
        logger.info(f"Modèle hybride chargé: {filepath}")

class WeightedHybridModel(BaseHybridModel):
    """
    Modèle hybride utilisant une combinaison pondérée des prédictions
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres du modèle hybride
        hybrid_params = config.get('hybrid', {})
        self.collaborative_models = hybrid_params.get('collaborative_models', ['svd', 'nmf', 'knn'])
        self.content_models = hybrid_params.get('content_models', ['tfidf', 'genre'])
        self.optimization_method = hybrid_params.get('optimization_method', 'grid_search')
        
        # Poids initiaux (seront optimisés)
        self.weights = {
            'collaborative': {model: 1.0 for model in self.collaborative_models},
            'content': {model: 1.0 for model in self.content_models}
        }
        
        # Normalisation des scores
        self.scaler = MinMaxScaler()
        
    def fit(self, train_data: pd.DataFrame, movies_data: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle hybride pondéré
        
        Args:
            train_data: Données d'entraînement
            movies_data: Données des films
            val_data: Données de validation pour optimiser les poids
        """
        logger.info("Entraînement du modèle hybride pondéré...")
        
        # 1. Entraîner les modèles collaboratifs
        logger.info("Entraînement des modèles collaboratifs...")
        self.collaborative_manager = CollaborativeFilteringManager()
        for model_name in self.collaborative_models:
            self.collaborative_manager.train_model(model_name, train_data)
        
        # 2. Entraîner les modèles basés sur le contenu
        logger.info("Entraînement des modèles basés sur le contenu...")
        self.content_manager = ContentBasedManager()
        for model_name in self.content_models:
            self.content_manager.train_model(model_name, movies_data, train_data)
        
        # 3. Optimiser les poids si des données de validation sont fournies
        if val_data is not None:
            logger.info("Optimisation des poids...")
            self._optimize_weights(val_data, movies_data)
        else:
            # Utiliser des poids uniformes
            self._set_uniform_weights()
        
        self.is_trained = True
        logger.info("Modèle hybride pondéré entraîné avec succès")
        logger.info(f"Poids finaux: {self.weights}")
    
    def _set_uniform_weights(self) -> None:
        """
        Définit des poids uniformes pour tous les modèles
        """
        total_models = len(self.collaborative_models) + len(self.content_models)
        uniform_weight = 1.0 / total_models
        
        for model in self.collaborative_models:
            self.weights['collaborative'][model] = uniform_weight
        
        for model in self.content_models:
            self.weights['content'][model] = uniform_weight
    
    def _optimize_weights(self, val_data: pd.DataFrame, movies_data: pd.DataFrame) -> None:
        """
        Optimise les poids en utilisant les données de validation
        
        Args:
            val_data: Données de validation
            movies_data: Données des films
        """
        if self.optimization_method == 'grid_search':
            self._grid_search_optimization(val_data, movies_data)
        elif self.optimization_method == 'scipy_minimize':
            self._scipy_optimization(val_data, movies_data)
        else:
            logger.warning(f"Méthode d'optimisation inconnue: {self.optimization_method}")
            self._set_uniform_weights()
    
    def _grid_search_optimization(self, val_data: pd.DataFrame, movies_data: pd.DataFrame) -> None:
        """
        Optimisation par recherche sur grille
        
        Args:
            val_data: Données de validation
            movies_data: Données des films
        """
        logger.info("Optimisation par recherche sur grille...")
        
        # Définir la grille de recherche
        weight_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Créer toutes les combinaisons possibles
        param_grid = {}
        for model in self.collaborative_models:
            param_grid[f'collab_{model}'] = weight_values
        for model in self.content_models:
            param_grid[f'content_{model}'] = weight_values
        
        best_rmse = float('inf')
        best_weights = None
        
        # Échantillonner un sous-ensemble pour accélérer l'optimisation
        val_sample = val_data.sample(min(1000, len(val_data)), random_state=42)
        
        # Tester un sous-ensemble de combinaisons (pour éviter l'explosion combinatoire)
        grid = list(ParameterGrid(param_grid))
        sample_size = min(100, len(grid))  # Limiter à 100 combinaisons
        sampled_grid = np.random.choice(len(grid), sample_size, replace=False)
        
        for i, idx in enumerate(sampled_grid):
            params = grid[idx]
            
            # Normaliser les poids pour qu'ils somment à 1
            total_weight = sum(params.values())
            if total_weight == 0:
                continue
            
            normalized_params = {k: v/total_weight for k, v in params.items()}
            
            # Convertir en format de poids
            test_weights = {
                'collaborative': {},
                'content': {}
            }
            
            for model in self.collaborative_models:
                test_weights['collaborative'][model] = normalized_params[f'collab_{model}']
            for model in self.content_models:
                test_weights['content'][model] = normalized_params[f'content_{model}']
            
            # Évaluer cette combinaison de poids
            rmse = self._evaluate_weights(test_weights, val_sample, movies_data)
            
            if rmse < best_rmse:
                best_rmse = rmse
                best_weights = test_weights.copy()
            
            if (i + 1) % 20 == 0:
                logger.info(f"Testé {i + 1}/{sample_size} combinaisons, meilleur RMSE: {best_rmse:.4f}")
        
        if best_weights is not None:
            self.weights = best_weights
            logger.info(f"Meilleurs poids trouvés avec RMSE: {best_rmse:.4f}")
        else:
            logger.warning("Aucune combinaison valide trouvée, utilisation de poids uniformes")
            self._set_uniform_weights()
    
    def _scipy_optimization(self, val_data: pd.DataFrame, movies_data: pd.DataFrame) -> None:
        """
        Optimisation avec scipy.optimize
        
        Args:
            val_data: Données de validation
            movies_data: Données des films
        """
        logger.info("Optimisation avec scipy...")
        
        # Échantillonner pour accélérer
        val_sample = val_data.sample(min(500, len(val_data)), random_state=42)
        
        # Nombre total de modèles
        n_models = len(self.collaborative_models) + len(self.content_models)
        
        def objective(weights_array):
            # Convertir le tableau en dictionnaire de poids
            test_weights = {
                'collaborative': {},
                'content': {}
            }
            
            idx = 0
            for model in self.collaborative_models:
                test_weights['collaborative'][model] = weights_array[idx]
                idx += 1
            for model in self.content_models:
                test_weights['content'][model] = weights_array[idx]
                idx += 1
            
            return self._evaluate_weights(test_weights, val_sample, movies_data)
        
        # Contraintes: tous les poids doivent sommer à 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bornes: tous les poids entre 0 et 1
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Point de départ: poids uniformes
        x0 = np.ones(n_models) / n_models
        
        # Optimisation
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            # Convertir le résultat en poids
            optimal_weights = {
                'collaborative': {},
                'content': {}
            }
            
            idx = 0
            for model in self.collaborative_models:
                optimal_weights['collaborative'][model] = result.x[idx]
                idx += 1
            for model in self.content_models:
                optimal_weights['content'][model] = result.x[idx]
                idx += 1
            
            self.weights = optimal_weights
            logger.info(f"Optimisation réussie avec RMSE: {result.fun:.4f}")
        else:
            logger.warning("Échec de l'optimisation, utilisation de poids uniformes")
            self._set_uniform_weights()
    
    def _evaluate_weights(self, test_weights: Dict[str, Dict[str, float]], val_data: pd.DataFrame, movies_data: pd.DataFrame) -> float:
        """
        Évalue une combinaison de poids sur les données de validation
        
        Args:
            test_weights: Poids à tester
            val_data: Données de validation
            movies_data: Données des films
            
        Returns:
            RMSE sur les données de validation
        """
        predictions = []
        actuals = []
        
        for _, row in val_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']
            
            try:
                predicted_rating = self._predict_with_weights(user_id, movie_id, test_weights, movies_data)
                predictions.append(predicted_rating)
                actuals.append(actual_rating)
            except Exception:
                # Ignorer les prédictions qui échouent
                continue
        
        if len(predictions) == 0:
            return float('inf')
        
        return np.sqrt(mean_squared_error(actuals, predictions))
    
    def _predict_with_weights(self, user_id: int, movie_id: int, weights: Dict[str, Dict[str, float]], movies_data: pd.DataFrame = None) -> float:
        """
        Fait une prédiction avec des poids spécifiques
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            weights: Poids à utiliser
            movies_data: Données des films (pour le contenu)
            
        Returns:
            Prédiction pondérée
        """
        predictions = []
        weights_sum = 0
        
        # Prédictions collaboratives
        for model_name in self.collaborative_models:
            weight = weights['collaborative'][model_name]
            if weight > 0:
                try:
                    pred = self.collaborative_manager.trained_models[model_name].predict(user_id, movie_id)
                    predictions.append(weight * pred)
                    weights_sum += weight
                except Exception:
                    continue
        
        # Prédictions basées sur le contenu (utiliser la moyenne des ratings de l'utilisateur comme approximation)
        if movies_data is not None:
            for model_name in self.content_models:
                weight = weights['content'][model_name]
                if weight > 0:
                    try:
                        # Pour le contenu, utiliser une approximation basée sur la similarité
                        # Ceci est une simplification - dans un vrai système, on aurait une méthode plus sophistiquée
                        content_score = 3.5  # Score par défaut
                        predictions.append(weight * content_score)
                        weights_sum += weight
                    except Exception:
                        continue
        
        if weights_sum == 0:
            return 3.5  # Rating moyen par défaut
        
        return sum(predictions) / weights_sum
    
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
        
        return self._predict_with_weights(user_id, movie_id, self.weights)
    
    def recommend(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur en combinant toutes les approches
        
        Args:
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de tuples (movie_id, score)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de faire des recommandations")
        
        # Collecter les recommandations de tous les modèles
        all_recommendations = {}
        
        # Recommandations collaboratives
        for model_name in self.collaborative_models:
            weight = self.weights['collaborative'][model_name]
            if weight > 0:
                try:
                    recs = self.collaborative_manager.get_recommendations(model_name, user_id, n_recommendations * 2)
                    for movie_id, score in recs:
                        if movie_id not in all_recommendations:
                            all_recommendations[movie_id] = 0
                        all_recommendations[movie_id] += weight * score
                except Exception as e:
                    logger.warning(f"Erreur avec le modèle collaboratif {model_name}: {e}")
                    continue
        
        # Recommandations basées sur le contenu
        for model_name in self.content_models:
            weight = self.weights['content'][model_name]
            if weight > 0:
                try:
                    recs = self.content_manager.get_user_recommendations(model_name, user_id, ratings_data, n_recommendations * 2)
                    for movie_id, score in recs:
                        if movie_id not in all_recommendations:
                            all_recommendations[movie_id] = 0
                        all_recommendations[movie_id] += weight * score
                except Exception as e:
                    logger.warning(f"Erreur avec le modèle de contenu {model_name}: {e}")
                    continue
        
        # Trier et retourner les meilleures recommandations
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_recommendations[:n_recommendations]

class SwitchingHybridModel(BaseHybridModel):
    """
    Modèle hybride qui bascule entre les approches selon le contexte
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Seuils pour le basculement
        hybrid_params = config.get('hybrid', {})
        self.min_ratings_collaborative = hybrid_params.get('min_ratings_collaborative', 10)
        self.min_ratings_content = hybrid_params.get('min_ratings_content', 5)
        
    def fit(self, train_data: pd.DataFrame, movies_data: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle hybride de basculement
        """
        logger.info("Entraînement du modèle hybride de basculement...")
        
        # Entraîner les deux types de modèles
        self.collaborative_manager = CollaborativeFilteringManager()
        self.collaborative_manager.train_model('svd', train_data)  # Utiliser SVD comme modèle principal
        
        self.content_manager = ContentBasedManager()
        self.content_manager.train_model('tfidf', movies_data, train_data)  # Utiliser TF-IDF comme modèle principal
        
        self.is_trained = True
        logger.info("Modèle hybride de basculement entraîné avec succès")
    
    def _choose_strategy(self, user_id: int, ratings_data: pd.DataFrame) -> str:
        """
        Choisit la stratégie à utiliser selon le profil utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            
        Returns:
            'collaborative', 'content', ou 'hybrid'
        """
        user_ratings = ratings_data[ratings_data['userId'] == user_id]
        n_ratings = len(user_ratings)
        
        if n_ratings >= self.min_ratings_collaborative:
            return 'collaborative'
        elif n_ratings >= self.min_ratings_content:
            return 'content'
        else:
            return 'hybrid'  # Utiliser les deux pour les nouveaux utilisateurs
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit la note en choisissant la meilleure stratégie
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        # Pour la prédiction, utiliser principalement le collaboratif
        try:
            return self.collaborative_manager.trained_models['svd'].predict(user_id, movie_id)
        except Exception:
            return 3.5  # Rating moyen par défaut
    
    def recommend(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande en choisissant la stratégie appropriée
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        strategy = self._choose_strategy(user_id, ratings_data)
        
        if strategy == 'collaborative':
            return self.collaborative_manager.get_recommendations('svd', user_id, n_recommendations)
        elif strategy == 'content':
            return self.content_manager.get_user_recommendations('tfidf', user_id, ratings_data, n_recommendations)
        else:  # hybrid
            # Combiner les deux approches
            collab_recs = self.collaborative_manager.get_recommendations('svd', user_id, n_recommendations // 2)
            content_recs = self.content_manager.get_user_recommendations('tfidf', user_id, ratings_data, n_recommendations // 2)
            
            # Combiner et dédupliquer
            all_recs = {}
            for movie_id, score in collab_recs:
                all_recs[movie_id] = score
            
            for movie_id, score in content_recs:
                if movie_id in all_recs:
                    all_recs[movie_id] = (all_recs[movie_id] + score) / 2
                else:
                    all_recs[movie_id] = score
            
            sorted_recs = sorted(all_recs.items(), key=lambda x: x[1], reverse=True)
            return sorted_recs[:n_recommendations]

class HybridSystemManager:
    """
    Gestionnaire pour tous les modèles hybrides
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le gestionnaire hybride
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        self.models = {
            'weighted': WeightedHybridModel(self.config),
            'switching': SwitchingHybridModel(self.config)
        }
        
        self.trained_models = {}
    
    def train_model(self, model_name: str, train_data: pd.DataFrame, movies_data: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """
        Entraîne un modèle hybride spécifique
        
        Args:
            model_name: Nom du modèle ('weighted', 'switching')
            train_data: Données d'entraînement
            movies_data: Données des films
            val_data: Données de validation (optionnel)
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle hybride inconnu: {model_name}")
        
        logger.info(f"Entraînement du modèle hybride {model_name}...")
        self.models[model_name].fit(train_data, movies_data, val_data)
        self.trained_models[model_name] = self.models[model_name]
        logger.info(f"Modèle hybride {model_name} entraîné avec succès")
    
    def train_all_models(self, train_data: pd.DataFrame, movies_data: pd.DataFrame, val_data: pd.DataFrame = None) -> None:
        """
        Entraîne tous les modèles hybrides
        
        Args:
            train_data: Données d'entraînement
            movies_data: Données des films
            val_data: Données de validation (optionnel)
        """
        for model_name in self.models.keys():
            self.train_model(model_name, train_data, movies_data, val_data)
    
    def get_recommendations(self, model_name: str, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Obtient des recommandations d'un modèle hybride spécifique
        
        Args:
            model_name: Nom du modèle
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de recommandations
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle hybride {model_name} non entraîné")
        
        return self.trained_models[model_name].recommend(user_id, ratings_data, n_recommendations)
    
    def save_models(self, save_dir: str) -> None:
        """
        Sauvegarde tous les modèles hybrides entraînés
        
        Args:
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = save_path / f"hybrid_{model_name}_model.pkl"
            model.save_model(str(filepath))
    
    def load_models(self, save_dir: str) -> None:
        """
        Charge tous les modèles hybrides sauvegardés
        
        Args:
            save_dir: Répertoire des modèles
        """
        save_path = Path(save_dir)
        
        for model_name in self.models.keys():
            filepath = save_path / f"hybrid_{model_name}_model.pkl"
            if filepath.exists():
                self.models[model_name].load_model(str(filepath))
                self.trained_models[model_name] = self.models[model_name]
                logger.info(f"Modèle hybride {model_name} chargé")

def main():
    """
    Fonction principale pour tester les modèles hybrides
    """
    # Exemple d'utilisation
    manager = HybridSystemManager()
    
    # Charger les données
    train_data = pd.read_csv("data/processed/train_data.csv")
    val_data = pd.read_csv("data/processed/val_data.csv")
    movies_data = pd.read_csv("data/processed/movies_processed.csv")
    
    # Entraîner tous les modèles hybrides
    manager.train_all_models(train_data, movies_data, val_data)
    
    # Obtenir des recommandations
    user_id = 1
    for model_name in ['weighted', 'switching']:
        recommendations = manager.get_recommendations(model_name, user_id, train_data, 5)
        print(f"\nRecommandations hybrides {model_name} pour l'utilisateur {user_id}:")
        for movie_id, score in recommendations:
            print(f"  Film {movie_id}: {score:.3f}")
    
    # Sauvegarder les modèles
    manager.save_models("data/models")

if __name__ == "__main__":
    main()