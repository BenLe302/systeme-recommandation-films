#!/usr/bin/env python3
"""
Prétraitement des données MovieLens 20M
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MovieLensPreprocessor:
    """
    Classe pour le prétraitement des données MovieLens 20M
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le préprocesseur avec la configuration
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.config = self._load_config(config_path)
        self.data_path = Path(self.config['data']['raw_path'])
        self.processed_path = Path(self.config['data']['processed_path'])
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Encodeurs pour les IDs
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Charge la configuration depuis le fichier YAML
        
        Args:
            config_path: Chemin vers le fichier de configuration
            
        Returns:
            Configuration sous forme de dictionnaire
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Fichier de configuration non trouvé: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
    
    def load_raw_data(self) -> Tuple[pd.DataFrame, ...]:
        """
        Charge toutes les données brutes
        
        Returns:
            Tuple contenant tous les DataFrames
        """
        logger.info("Chargement des données brutes...")
        
        try:
            # Chargement des fichiers principaux
            ratings = pd.read_csv(self.data_path / self.config['data']['files']['ratings'])
            movies = pd.read_csv(self.data_path / self.config['data']['files']['movies'])
            tags = pd.read_csv(self.data_path / self.config['data']['files']['tags'])
            genome_scores = pd.read_csv(self.data_path / self.config['data']['files']['genome_scores'])
            genome_tags = pd.read_csv(self.data_path / self.config['data']['files']['genome_tags'])
            links = pd.read_csv(self.data_path / self.config['data']['files']['links'])
            
            logger.info(f"Données chargées:")
            logger.info(f"  - Ratings: {ratings.shape}")
            logger.info(f"  - Movies: {movies.shape}")
            logger.info(f"  - Tags: {tags.shape}")
            logger.info(f"  - Genome Scores: {genome_scores.shape}")
            logger.info(f"  - Genome Tags: {genome_tags.shape}")
            logger.info(f"  - Links: {links.shape}")
            
            return ratings, movies, tags, genome_scores, genome_tags, links
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données: {e}")
            raise
    
    def clean_ratings_data(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie et filtre les données de ratings
        
        Args:
            ratings: DataFrame des ratings
            
        Returns:
            DataFrame nettoyé
        """
        logger.info("Nettoyage des données de ratings...")
        
        # Supprimer les valeurs manquantes
        ratings_clean = ratings.dropna().copy()
        
        # Convertir le timestamp en datetime (gérer les formats multiples)
        try:
            # Essayer d'abord comme timestamp Unix
            ratings_clean.loc[:, 'timestamp'] = pd.to_datetime(ratings_clean['timestamp'], unit='s')
        except (ValueError, OSError):
            try:
                # Si ça échoue, essayer comme string datetime
                ratings_clean.loc[:, 'timestamp'] = pd.to_datetime(ratings_clean['timestamp'], errors='coerce')
                # Supprimer les lignes avec des timestamps invalides
                ratings_clean = ratings_clean.dropna(subset=['timestamp'])
            except Exception as e:
                logger.warning(f"Erreur de conversion timestamp: {e}")
                # En dernier recours, garder les timestamps comme ils sont
                pass
        
        # Filtrer selon les paramètres de configuration
        min_ratings_user = self.config['preprocessing']['min_ratings_per_user']
        min_ratings_movie = self.config['preprocessing']['min_ratings_per_movie']
        
        try:
            # Compter les ratings par utilisateur et par film
            logger.info("Comptage des ratings par utilisateur...")
            user_counts = ratings_clean['userId'].value_counts()
            logger.info("Comptage des ratings par film...")
            movie_counts = ratings_clean['movieId'].value_counts()
            
            # Filtrer les utilisateurs et films avec suffisamment de ratings
            logger.info("Filtrage des utilisateurs valides...")
            valid_users = user_counts[user_counts >= min_ratings_user].index
            logger.info("Filtrage des films valides...")
            valid_movies = movie_counts[movie_counts >= min_ratings_movie].index
            
            logger.info("Application des filtres...")
            ratings_filtered = ratings_clean[
                (ratings_clean['userId'].isin(valid_users)) &
                (ratings_clean['movieId'].isin(valid_movies))
            ]
        except Exception as e:
            logger.error(f"Erreur lors du filtrage: {e}")
            # En cas d'erreur, retourner les données nettoyées sans filtrage
            ratings_filtered = ratings_clean
        
        logger.info(f"Filtrage terminé:")
        logger.info(f"  - Ratings originaux: {len(ratings_clean)}")
        logger.info(f"  - Ratings filtrés: {len(ratings_filtered)}")
        logger.info(f"  - Utilisateurs uniques: {ratings_filtered['userId'].nunique()}")
        logger.info(f"  - Films uniques: {ratings_filtered['movieId'].nunique()}")
        
        return ratings_filtered
    
    def process_movies_data(self, movies: pd.DataFrame) -> pd.DataFrame:
        """
        Traite les données des films
        
        Args:
            movies: DataFrame des films
            
        Returns:
            DataFrame traité
        """
        logger.info("Traitement des données de films...")
        
        movies_processed = movies.copy()
        
        # Extraire l'année du titre
        movies_processed['year'] = movies_processed['title'].str.extract(r'\((\d{4})\)$')
        movies_processed['year'] = pd.to_numeric(movies_processed['year'], errors='coerce')
        
        # Nettoyer le titre (enlever l'année)
        movies_processed['title_clean'] = movies_processed['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)
        
        # Traiter les genres
        movies_processed['genres_list'] = movies_processed['genres'].str.split('|')
        
        # Créer des colonnes binaires pour chaque genre
        all_genres = set()
        for genres_list in movies_processed['genres_list'].dropna():
            all_genres.update(genres_list)
        
        for genre in sorted(all_genres):
            if genre != '(no genres listed)':
                movies_processed[f'genre_{genre}'] = movies_processed['genres_list'].apply(
                    lambda x: 1 if x and genre in x else 0
                )
        
        logger.info(f"Genres identifiés: {sorted(all_genres)}")
        
        return movies_processed
    
    def create_user_movie_matrix(self, ratings: pd.DataFrame) -> pd.DataFrame:
        """
        Crée la matrice utilisateur-film
        
        Args:
            ratings: DataFrame des ratings
            
        Returns:
            Matrice utilisateur-film
        """
        logger.info("Création de la matrice utilisateur-film...")
        
        # Encoder les IDs
        ratings_encoded = ratings.copy()
        ratings_encoded['user_idx'] = self.user_encoder.fit_transform(ratings['userId'])
        ratings_encoded['movie_idx'] = self.movie_encoder.fit_transform(ratings['movieId'])
        
        # Créer la matrice pivot
        user_movie_matrix = ratings_encoded.pivot_table(
            index='user_idx',
            columns='movie_idx',
            values='rating',
            fill_value=0
        )
        
        logger.info(f"Matrice créée: {user_movie_matrix.shape}")
        logger.info(f"Sparsité: {(user_movie_matrix == 0).sum().sum() / user_movie_matrix.size:.4f}")
        
        return user_movie_matrix, ratings_encoded
    
    def split_data(self, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """
        Divise les données en ensembles d'entraînement, validation et test
        
        Args:
            ratings: DataFrame des ratings
            
        Returns:
            Tuple des ensembles train, validation, test
        """
        logger.info("Division des données...")
        
        # Paramètres de division
        train_ratio = self.config['preprocessing']['train_ratio']
        val_ratio = self.config['preprocessing']['val_ratio']
        test_ratio = self.config['preprocessing']['test_ratio']
        
        # Division temporelle (plus réaliste pour les systèmes de recommandation)
        ratings_sorted = ratings.sort_values('timestamp')
        
        n_total = len(ratings_sorted)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = ratings_sorted.iloc[:n_train]
        val_data = ratings_sorted.iloc[n_train:n_train + n_val]
        test_data = ratings_sorted.iloc[n_train + n_val:]
        
        logger.info(f"Division terminée:")
        logger.info(f"  - Train: {len(train_data)} ({len(train_data)/n_total:.2%})")
        logger.info(f"  - Validation: {len(val_data)} ({len(val_data)/n_total:.2%})")
        logger.info(f"  - Test: {len(test_data)} ({len(test_data)/n_total:.2%})")
        
        return train_data, val_data, test_data
    
    def save_processed_data(self, **datasets) -> None:
        """
        Sauvegarde les données traitées
        
        Args:
            **datasets: Dictionnaire des datasets à sauvegarder
        """
        logger.info("Sauvegarde des données traitées...")
        
        for name, data in datasets.items():
            if isinstance(data, pd.DataFrame):
                filepath = self.processed_path / f"{name}.csv"
                data.to_csv(filepath, index=False)
                logger.info(f"Sauvegardé: {filepath}")
            else:
                filepath = self.processed_path / f"{name}.pkl"
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Sauvegardé: {filepath}")
        
        # Sauvegarder les encodeurs
        encoders = {
            'user_encoder': self.user_encoder,
            'movie_encoder': self.movie_encoder
        }
        
        with open(self.processed_path / 'encoders.pkl', 'wb') as f:
            pickle.dump(encoders, f)
        
        logger.info("Sauvegarde terminée.")
    
    def run_preprocessing(self) -> None:
        """
        Exécute tout le pipeline de prétraitement
        """
        logger.info("Début du prétraitement des données MovieLens 20M")
        
        try:
            # 1. Charger les données brutes
            ratings, movies, tags, genome_scores, genome_tags, links = self.load_raw_data()
            
            # 2. Nettoyer les ratings
            ratings_clean = self.clean_ratings_data(ratings)
            
            # 3. Traiter les films
            movies_processed = self.process_movies_data(movies)
            
            # 4. Créer la matrice utilisateur-film
            user_movie_matrix, ratings_encoded = self.create_user_movie_matrix(ratings_clean)
            
            # 5. Diviser les données
            train_data, val_data, test_data = self.split_data(ratings_clean)
            
            # 6. Sauvegarder tout
            self.save_processed_data(
                ratings_clean=ratings_clean,
                movies_processed=movies_processed,
                tags=tags,
                genome_scores=genome_scores,
                genome_tags=genome_tags,
                links=links,
                user_movie_matrix=user_movie_matrix,
                ratings_encoded=ratings_encoded,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data
            )
            
            logger.info("Prétraitement terminé avec succès !")
            
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement: {e}")
            raise

def main():
    """
    Fonction principale
    """
    preprocessor = MovieLensPreprocessor()
    preprocessor.run_preprocessing()

if __name__ == "__main__":
    main()