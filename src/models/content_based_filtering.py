#!/usr/bin/env python3
"""
Modèles de filtrage basé sur le contenu pour le système de recommandation
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

# Scikit-learn pour le traitement de texte et les métriques
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# NLTK pour le traitement de texte avancé
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Gensim pour Word2Vec et Doc2Vec
# Import optionnel de gensim
try:
    from gensim.models import Word2Vec, Doc2Vec
    from gensim.models.doc2vec import TaggedDocument
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Gensim non disponible. Utilisation des alternatives scikit-learn.")

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class BaseContentModel(ABC):
    """
    Classe de base pour tous les modèles de filtrage basé sur le contenu
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
        self.movies_features = None
        self.similarity_matrix = None
        self.movie_indices = None
        
    @abstractmethod
    def fit(self, movies_data: pd.DataFrame, ratings_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle
        
        Args:
            movies_data: Données des films
            ratings_data: Données des ratings (optionnel)
        """
        pass
    
    @abstractmethod
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Trouve les films similaires
        
        Args:
            movie_id: ID du film
            n_similar: Nombre de films similaires
            
        Returns:
            Liste de tuples (movie_id, similarité)
        """
        pass
    
    @abstractmethod
    def recommend_for_user(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur basé sur son historique
        
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
        Sauvegarde le modèle
        
        Args:
            filepath: Chemin de sauvegarde
        """
        model_data = {
            'model': self.model,
            'config': self.config,
            'is_trained': self.is_trained,
            'movies_features': self.movies_features,
            'similarity_matrix': self.similarity_matrix,
            'movie_indices': self.movie_indices
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
        self.movies_features = model_data['movies_features']
        self.similarity_matrix = model_data['similarity_matrix']
        self.movie_indices = model_data['movie_indices']
        
        logger.info(f"Modèle chargé: {filepath}")

class TFIDFContentModel(BaseContentModel):
    """
    Modèle basé sur TF-IDF pour analyser les genres et descriptions des films
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Paramètres TF-IDF
        tfidf_params = config.get('content_based', {}).get('tfidf', {})
        self.max_features = tfidf_params.get('max_features', 5000)
        self.ngram_range = tuple(tfidf_params.get('ngram_range', [1, 2]))
        self.min_df = tfidf_params.get('min_df', 2)
        self.max_df = tfidf_params.get('max_df', 0.8)
        
        # Initialiser le vectoriseur TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True
        )
        
        # Préprocesseur de texte
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def preprocess_text(self, text: str) -> str:
        """
        Préprocesse le texte
        
        Args:
            text: Texte à préprocesser
            
        Returns:
            Texte préprocessé
        """
        if pd.isna(text):
            return ""
        
        # Tokenisation
        tokens = word_tokenize(text.lower())
        
        # Suppression des mots vides et lemmatisation
        processed_tokens = []
        for token in tokens:
            if token.isalpha() and token not in self.stop_words:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return ' '.join(processed_tokens)
    
    def create_content_features(self, movies_data: pd.DataFrame) -> str:
        """
        Crée les caractéristiques de contenu pour chaque film
        
        Args:
            movies_data: DataFrame des films
            
        Returns:
            Série avec les caractéristiques combinées
        """
        # Combiner les genres et le titre pour créer une description
        content_features = []
        
        for _, movie in movies_data.iterrows():
            features = []
            
            # Ajouter les genres (répétés pour donner plus de poids)
            if pd.notna(movie['genres']) and movie['genres'] != '(no genres listed)':
                genres = movie['genres'].replace('|', ' ').replace('-', ' ')
                features.extend([genres] * 3)  # Répéter 3 fois pour plus de poids
            
            # Ajouter le titre nettoyé
            if pd.notna(movie.get('title_clean')):
                title_words = movie['title_clean'].replace('-', ' ').replace(':', ' ')
                features.append(title_words)
            
            # Ajouter l'année comme contexte
            if pd.notna(movie.get('year')):
                decade = str(int(movie['year'] // 10) * 10) + 's'
                features.append(decade)
            
            # Combiner toutes les caractéristiques
            combined_features = ' '.join(features)
            content_features.append(self.preprocess_text(combined_features))
        
        return content_features
    
    def fit(self, movies_data: pd.DataFrame, ratings_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle TF-IDF
        
        Args:
            movies_data: Données des films
            ratings_data: Données des ratings (non utilisé pour TF-IDF)
        """
        logger.info("Entraînement du modèle TF-IDF...")
        
        # Créer les caractéristiques de contenu
        content_features = self.create_content_features(movies_data)
        
        # Ajuster le vectoriseur TF-IDF
        self.movies_features = self.tfidf_vectorizer.fit_transform(content_features)
        
        # Calculer la matrice de similarité cosinus
        self.similarity_matrix = cosine_similarity(self.movies_features)
        
        # Créer un mapping des IDs de films vers les indices
        self.movie_indices = pd.Series(
            movies_data.index, 
            index=movies_data['movieId']
        ).to_dict()
        
        self.is_trained = True
        logger.info(f"Modèle TF-IDF entraîné avec {self.movies_features.shape[0]} films et {self.movies_features.shape[1]} caractéristiques")
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Trouve les films similaires basés sur le contenu
        
        Args:
            movie_id: ID du film
            n_similar: Nombre de films similaires
            
        Returns:
            Liste de tuples (movie_id, similarité)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné avant de trouver des films similaires")
        
        if movie_id not in self.movie_indices:
            logger.warning(f"Film {movie_id} non trouvé")
            return []
        
        # Obtenir l'indice du film
        movie_idx = self.movie_indices[movie_id]
        
        # Obtenir les scores de similarité
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Trier par similarité décroissante (exclure le film lui-même)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_similar+1]
        
        # Convertir les indices en IDs de films
        similar_movies = []
        movie_id_list = list(self.movie_indices.keys())
        
        for idx, score in similarity_scores:
            similar_movie_id = movie_id_list[idx]
            similar_movies.append((similar_movie_id, score))
        
        return similar_movies
    
    def recommend_for_user(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur basé sur son profil
        
        Args:
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de tuples (movie_id, score)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        # Obtenir les films notés par l'utilisateur
        user_ratings = ratings_data[ratings_data['userId'] == user_id]
        
        if user_ratings.empty:
            logger.warning(f"Aucun rating trouvé pour l'utilisateur {user_id}")
            return []
        
        # Calculer le profil utilisateur basé sur les films qu'il a aimés (rating >= 4)
        liked_movies = user_ratings[user_ratings['rating'] >= 4.0]
        
        if liked_movies.empty:
            # Si aucun film aimé, utiliser tous les films avec pondération par rating
            liked_movies = user_ratings
        
        # Créer le profil utilisateur en moyennant les caractéristiques des films aimés
        user_profile = np.zeros(self.movies_features.shape[1])
        total_weight = 0
        
        for _, rating_row in liked_movies.iterrows():
            movie_id = rating_row['movieId']
            rating = rating_row['rating']
            
            if movie_id in self.movie_indices:
                movie_idx = self.movie_indices[movie_id]
                movie_features = self.movies_features[movie_idx].toarray().flatten()
                
                # Pondérer par le rating
                weight = rating / 5.0  # Normaliser entre 0 et 1
                user_profile += weight * movie_features
                total_weight += weight
        
        if total_weight > 0:
            user_profile /= total_weight
        
        # Calculer la similarité avec tous les films
        user_profile_reshaped = user_profile.reshape(1, -1)
        similarities = cosine_similarity(user_profile_reshaped, self.movies_features).flatten()
        
        # Obtenir les films déjà notés pour les exclure
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        # Créer la liste des recommandations
        recommendations = []
        movie_id_list = list(self.movie_indices.keys())
        
        for idx, similarity in enumerate(similarities):
            movie_id = movie_id_list[idx]
            if movie_id not in rated_movie_ids:
                recommendations.append((movie_id, similarity))
        
        # Trier par similarité décroissante
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit le rating qu'un utilisateur donnerait à un film
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            
        Returns:
            Rating prédit (entre 1 et 5)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        if movie_id not in self.movie_indices:
            # Si le film n'est pas dans les données d'entraînement, retourner la moyenne
            return 3.5
        
        # Obtenir l'indice du film
        movie_idx = self.movie_indices[movie_id]
        
        # Calculer la similarité moyenne avec tous les autres films
        # (approximation simple pour la prédiction de rating)
        similarities = self.similarity_matrix[movie_idx]
        
        # Retourner un rating basé sur la similarité moyenne
        # Plus la similarité moyenne est élevée, plus le rating est élevé
        avg_similarity = np.mean(similarities)
        
        # Mapper la similarité (0-1) vers un rating (1-5)
        # Ajouter un peu de bruit pour éviter des prédictions trop uniformes
        predicted_rating = 1 + (avg_similarity * 4) + np.random.normal(0, 0.1)
        
        # S'assurer que le rating est dans la plage valide
        return max(1.0, min(5.0, predicted_rating))

class GenreBasedModel(BaseContentModel):
    """
    Modèle basé uniquement sur les genres des films
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.genre_similarity_matrix = None
        self.genre_features = None
    
    def create_genre_features(self, movies_data: pd.DataFrame) -> pd.DataFrame:
        """
        Crée une matrice binaire des genres
        
        Args:
            movies_data: DataFrame des films
            
        Returns:
            DataFrame avec les caractéristiques de genre
        """
        # Obtenir tous les genres uniques
        all_genres = set()
        for genres_str in movies_data['genres'].dropna():
            if genres_str != '(no genres listed)':
                genres = genres_str.split('|')
                all_genres.update(genres)
        
        # Créer la matrice binaire des genres
        genre_matrix = pd.DataFrame(index=movies_data.index, columns=sorted(all_genres))
        genre_matrix = genre_matrix.fillna(0)
        
        for idx, row in movies_data.iterrows():
            if pd.notna(row['genres']) and row['genres'] != '(no genres listed)':
                genres = row['genres'].split('|')
                for genre in genres:
                    if genre in genre_matrix.columns:
                        genre_matrix.loc[idx, genre] = 1
        
        return genre_matrix.astype(int)
    
    def fit(self, movies_data: pd.DataFrame, ratings_data: pd.DataFrame = None) -> None:
        """
        Entraîne le modèle basé sur les genres
        
        Args:
            movies_data: Données des films
            ratings_data: Données des ratings (optionnel)
        """
        logger.info("Entraînement du modèle basé sur les genres...")
        
        # Créer les caractéristiques de genre
        self.genre_features = self.create_genre_features(movies_data)
        
        # Calculer la similarité cosinus entre les films
        self.similarity_matrix = cosine_similarity(self.genre_features)
        
        # Créer le mapping des IDs
        self.movie_indices = pd.Series(
            movies_data.index, 
            index=movies_data['movieId']
        ).to_dict()
        
        self.is_trained = True
        logger.info(f"Modèle genre entraîné avec {len(self.genre_features.columns)} genres")
    
    def get_similar_movies(self, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Trouve les films similaires basés sur les genres
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        if movie_id not in self.movie_indices:
            return []
        
        movie_idx = self.movie_indices[movie_id]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_similar+1]
        
        similar_movies = []
        movie_id_list = list(self.movie_indices.keys())
        
        for idx, score in similarity_scores:
            similar_movie_id = movie_id_list[idx]
            similar_movies.append((similar_movie_id, score))
        
        return similar_movies
    
    def recommend_for_user(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Recommande des films basés sur les préférences de genre de l'utilisateur
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        user_ratings = ratings_data[ratings_data['userId'] == user_id]
        
        if user_ratings.empty:
            return []
        
        # Calculer les préférences de genre de l'utilisateur
        genre_preferences = np.zeros(len(self.genre_features.columns))
        total_weight = 0
        
        for _, rating_row in user_ratings.iterrows():
            movie_id = rating_row['movieId']
            rating = rating_row['rating']
            
            if movie_id in self.movie_indices:
                movie_idx = self.movie_indices[movie_id]
                movie_genres = self.genre_features.iloc[movie_idx].values
                
                weight = rating / 5.0
                genre_preferences += weight * movie_genres
                total_weight += weight
        
        if total_weight > 0:
            genre_preferences /= total_weight
        
        # Calculer les scores pour tous les films
        movie_scores = np.dot(self.genre_features.values, genre_preferences)
        
        # Exclure les films déjà notés
        rated_movie_ids = set(user_ratings['movieId'].values)
        
        recommendations = []
        movie_id_list = list(self.movie_indices.keys())
        
        for idx, score in enumerate(movie_scores):
            movie_id = movie_id_list[idx]
            if movie_id not in rated_movie_ids:
                recommendations.append((movie_id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:n_recommendations]
    
    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Prédit le rating qu'un utilisateur donnerait à un film basé sur les genres
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            
        Returns:
            Rating prédit (entre 1 et 5)
        """
        if not self.is_trained:
            raise ValueError("Le modèle doit être entraîné")
        
        if movie_id not in self.movie_indices:
            return 3.5
        
        # Obtenir l'indice du film
        movie_idx = self.movie_indices[movie_id]
        
        # Calculer la similarité moyenne avec tous les autres films
        similarities = self.similarity_matrix[movie_idx]
        avg_similarity = np.mean(similarities)
        
        # Mapper la similarité vers un rating avec un peu de variation
        predicted_rating = 2.5 + (avg_similarity * 2.5) + np.random.normal(0, 0.15)
        
        # S'assurer que le rating est dans la plage valide
        return max(1.0, min(5.0, predicted_rating))

class ContentBasedManager:
    """
    Gestionnaire pour tous les modèles de filtrage basé sur le contenu
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
            'tfidf': TFIDFContentModel(self.config),
            'genre': GenreBasedModel(self.config)
        }
        
        self.trained_models = {}
    
    def train_model(self, model_name: str, movies_data: pd.DataFrame, ratings_data: pd.DataFrame = None) -> None:
        """
        Entraîne un modèle spécifique
        
        Args:
            model_name: Nom du modèle ('tfidf', 'genre')
            movies_data: Données des films
            ratings_data: Données des ratings (optionnel)
        """
        if model_name not in self.models:
            raise ValueError(f"Modèle inconnu: {model_name}")
        
        logger.info(f"Entraînement du modèle {model_name}...")
        self.models[model_name].fit(movies_data, ratings_data)
        self.trained_models[model_name] = self.models[model_name]
        logger.info(f"Modèle {model_name} entraîné avec succès")
    
    def train_all_models(self, movies_data: pd.DataFrame, ratings_data: pd.DataFrame = None) -> None:
        """
        Entraîne tous les modèles
        
        Args:
            movies_data: Données des films
            ratings_data: Données des ratings (optionnel)
        """
        for model_name in self.models.keys():
            self.train_model(model_name, movies_data, ratings_data)
    
    def get_similar_movies(self, model_name: str, movie_id: int, n_similar: int = 10) -> List[Tuple[int, float]]:
        """
        Obtient des films similaires d'un modèle spécifique
        
        Args:
            model_name: Nom du modèle
            movie_id: ID du film
            n_similar: Nombre de films similaires
            
        Returns:
            Liste de films similaires
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].get_similar_movies(movie_id, n_similar)
    
    def get_user_recommendations(self, model_name: str, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10) -> List[Tuple[int, float]]:
        """
        Obtient des recommandations pour un utilisateur
        
        Args:
            model_name: Nom du modèle
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            
        Returns:
            Liste de recommandations
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].recommend_for_user(user_id, ratings_data, n_recommendations)
    
    def save_models(self, save_dir: str) -> None:
        """
        Sauvegarde tous les modèles entraînés
        
        Args:
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for model_name, model in self.trained_models.items():
            filepath = save_path / f"content_{model_name}_model.pkl"
            model.save_model(str(filepath))
    
    def load_models(self, save_dir: str) -> None:
        """
        Charge tous les modèles sauvegardés
        
        Args:
            save_dir: Répertoire des modèles
        """
        save_path = Path(save_dir)
        
        for model_name in self.models.keys():
            filepath = save_path / f"content_{model_name}_model.pkl"
            if filepath.exists():
                self.models[model_name].load_model(str(filepath))
                self.trained_models[model_name] = self.models[model_name]
                logger.info(f"Modèle {model_name} chargé")
    
    def predict(self, user_id: int, movie_id: int, model_name: str = 'tfidf') -> float:
        """
        Prédit le rating qu'un utilisateur donnerait à un film
        
        Args:
            user_id: ID de l'utilisateur
            movie_id: ID du film
            model_name: Nom du modèle à utiliser ('tfidf' ou 'genre')
            
        Returns:
            Rating prédit (entre 1 et 5)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].predict(user_id, movie_id)
    
    def recommend(self, user_id: int, ratings_data: pd.DataFrame, n_recommendations: int = 10, model_name: str = 'tfidf') -> List[Tuple[int, float]]:
        """
        Recommande des films pour un utilisateur
        
        Args:
            user_id: ID de l'utilisateur
            ratings_data: Données des ratings
            n_recommendations: Nombre de recommandations
            model_name: Nom du modèle à utiliser
            
        Returns:
            Liste de recommandations (movie_id, score)
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Modèle {model_name} non entraîné")
        
        return self.trained_models[model_name].recommend_for_user(user_id, ratings_data, n_recommendations)

def main():
    """
    Fonction principale pour tester les modèles
    """
    # Exemple d'utilisation
    manager = ContentBasedManager()
    
    # Charger les données
    movies_data = pd.read_csv("data/processed/movies_processed.csv")
    ratings_data = pd.read_csv("data/processed/train_data.csv")
    
    # Entraîner tous les modèles
    manager.train_all_models(movies_data, ratings_data)
    
    # Tester la similarité entre films
    movie_id = 1  # Toy Story
    for model_name in ['tfidf', 'genre']:
        similar_movies = manager.get_similar_movies(model_name, movie_id, 5)
        print(f"\nFilms similaires à {movie_id} ({model_name}):")
        for similar_id, score in similar_movies:
            print(f"  Film {similar_id}: {score:.3f}")
    
    # Tester les recommandations utilisateur
    user_id = 1
    for model_name in ['tfidf', 'genre']:
        recommendations = manager.get_user_recommendations(model_name, user_id, ratings_data, 5)
        print(f"\nRecommandations {model_name} pour l'utilisateur {user_id}:")
        for movie_id, score in recommendations:
            print(f"  Film {movie_id}: {score:.3f}")
    
    # Sauvegarder les modèles
    manager.save_models("data/models")

if __name__ == "__main__":
    main()