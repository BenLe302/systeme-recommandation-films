#!/usr/bin/env python3
"""
Module d'évaluation pour le système de recommandation
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import pandas as pd
import numpy as np
import yaml
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Métriques de base
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecommendationMetrics:
    """
    Classe pour calculer toutes les métriques d'évaluation des systèmes de recommandation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise le calculateur de métriques
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
        
        # Paramètres d'évaluation
        eval_config = self.config.get('evaluation', {})
        self.k_values = eval_config.get('k_values', [5, 10, 20])
        self.rating_threshold = eval_config.get('rating_threshold', 4.0)
        
    def rmse(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule le Root Mean Square Error (RMSE)
        
        Args:
            y_true: Vraies valeurs
            y_pred: Valeurs prédites
            
        Returns:
            RMSE
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    def mae(self, y_true: List[float], y_pred: List[float]) -> float:
        """
        Calcule le Mean Absolute Error (MAE)
        
        Args:
            y_true: Vraies valeurs
            y_pred: Valeurs prédites
            
        Returns:
            MAE
        """
        return mean_absolute_error(y_true, y_pred)
    
    def precision_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calcule la précision à k
        
        Args:
            recommended_items: Items recommandés (ordonnés par score)
            relevant_items: Items pertinents pour l'utilisateur
            k: Nombre d'items à considérer
            
        Returns:
            Précision à k
        """
        if k == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / k
    
    def recall_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calcule le rappel à k
        
        Args:
            recommended_items: Items recommandés (ordonnés par score)
            relevant_items: Items pertinents pour l'utilisateur
            k: Nombre d'items à considérer
            
        Returns:
            Rappel à k
        """
        if len(relevant_items) == 0:
            return 0.0
        
        recommended_k = recommended_items[:k]
        relevant_recommended = len(set(recommended_k) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)
    
    def f1_score_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calcule le F1-score à k
        
        Args:
            recommended_items: Items recommandés
            relevant_items: Items pertinents
            k: Nombre d'items à considérer
            
        Returns:
            F1-score à k
        """
        precision = self.precision_at_k(recommended_items, relevant_items, k)
        recall = self.recall_at_k(recommended_items, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(self, recommended_items: List[int], relevant_items: List[int]) -> float:
        """
        Calcule l'Average Precision (AP)
        
        Args:
            recommended_items: Items recommandés (ordonnés par score)
            relevant_items: Items pertinents
            
        Returns:
            Average Precision
        """
        if len(relevant_items) == 0:
            return 0.0
        
        relevant_set = set(relevant_items)
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_items)
    
    def dcg_at_k(self, recommended_items: List[int], relevant_items: Dict[int, float], k: int) -> float:
        """
        Calcule le Discounted Cumulative Gain à k
        
        Args:
            recommended_items: Items recommandés (ordonnés par score)
            relevant_items: Dictionnaire {item_id: relevance_score}
            k: Nombre d'items à considérer
            
        Returns:
            DCG à k
        """
        dcg = 0.0
        for i, item in enumerate(recommended_items[:k]):
            relevance = relevant_items.get(item, 0.0)
            if i == 0:
                dcg += relevance
            else:
                dcg += relevance / np.log2(i + 1)
        
        return dcg
    
    def ndcg_at_k(self, recommended_items: List[int], relevant_items: Dict[int, float], k: int) -> float:
        """
        Calcule le Normalized Discounted Cumulative Gain à k
        
        Args:
            recommended_items: Items recommandés (ordonnés par score)
            relevant_items: Dictionnaire {item_id: relevance_score}
            k: Nombre d'items à considérer
            
        Returns:
            NDCG à k
        """
        dcg = self.dcg_at_k(recommended_items, relevant_items, k)
        
        # Calculer l'IDCG (DCG idéal)
        ideal_items = sorted(relevant_items.keys(), key=lambda x: relevant_items[x], reverse=True)
        idcg = self.dcg_at_k(ideal_items, relevant_items, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def hit_rate_at_k(self, recommended_items: List[int], relevant_items: List[int], k: int) -> float:
        """
        Calcule le taux de succès à k (1 si au moins un item pertinent, 0 sinon)
        
        Args:
            recommended_items: Items recommandés
            relevant_items: Items pertinents
            k: Nombre d'items à considérer
            
        Returns:
            Hit rate à k (0 ou 1)
        """
        recommended_k = set(recommended_items[:k])
        relevant_set = set(relevant_items)
        
        return 1.0 if len(recommended_k & relevant_set) > 0 else 0.0
    
    def coverage(self, all_recommendations: List[List[int]], total_items: int) -> float:
        """
        Calcule la couverture du catalogue
        
        Args:
            all_recommendations: Liste de toutes les recommandations pour tous les utilisateurs
            total_items: Nombre total d'items dans le catalogue
            
        Returns:
            Pourcentage d'items couverts
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return len(recommended_items) / total_items
    
    def diversity(self, recommendations: List[int], item_features: pd.DataFrame) -> float:
        """
        Calcule la diversité intra-liste des recommandations
        
        Args:
            recommendations: Liste des items recommandés
            item_features: DataFrame avec les caractéristiques des items
            
        Returns:
            Score de diversité (distance moyenne entre items)
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Filtrer les features pour les items recommandés
        rec_features = item_features[item_features.index.isin(recommendations)]
        
        if len(rec_features) < 2:
            return 0.0
        
        # Calculer la distance moyenne entre tous les pairs d'items
        distances = []
        for i in range(len(rec_features)):
            for j in range(i + 1, len(rec_features)):
                # Utiliser la distance cosinus
                dist = cosine(rec_features.iloc[i].values, rec_features.iloc[j].values)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def novelty(self, recommendations: List[int], item_popularity: Dict[int, float]) -> float:
        """
        Calcule la nouveauté des recommandations
        
        Args:
            recommendations: Liste des items recommandés
            item_popularity: Dictionnaire {item_id: popularité}
            
        Returns:
            Score de nouveauté (moyenne des -log(popularité))
        """
        if not recommendations:
            return 0.0
        
        novelty_scores = []
        for item in recommendations:
            popularity = item_popularity.get(item, 1e-6)  # Éviter log(0)
            novelty_scores.append(-np.log2(popularity + 1e-6))
        
        return np.mean(novelty_scores)
    
    def serendipity(self, recommendations: List[int], user_profile: List[int], item_similarity: Dict[Tuple[int, int], float]) -> float:
        """
        Calcule la sérendipité des recommandations
        
        Args:
            recommendations: Items recommandés
            user_profile: Items que l'utilisateur a déjà aimés
            item_similarity: Dictionnaire de similarités entre items
            
        Returns:
            Score de sérendipité
        """
        if not recommendations or not user_profile:
            return 0.0
        
        serendipity_scores = []
        
        for rec_item in recommendations:
            # Calculer la similarité moyenne avec le profil utilisateur
            similarities = []
            for profile_item in user_profile:
                sim = item_similarity.get((rec_item, profile_item), 0.0)
                similarities.append(sim)
            
            avg_similarity = np.mean(similarities) if similarities else 0.0
            # Sérendipité = 1 - similarité (plus c'est différent, plus c'est sérendipiteux)
            serendipity_scores.append(1 - avg_similarity)
        
        return np.mean(serendipity_scores)

class ModelEvaluator:
    """
    Classe pour évaluer les modèles de recommandation
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialise l'évaluateur de modèles
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        self.metrics = RecommendationMetrics(config_path)
        with open(config_path, 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)
    
    def evaluate_rating_prediction(self, model, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Évalue la prédiction de ratings
        
        Args:
            model: Modèle à évaluer
            test_data: Données de test
            
        Returns:
            Dictionnaire des métriques
        """
        logger.info("Évaluation de la prédiction de ratings...")
        
        y_true = []
        y_pred = []
        
        for _, row in test_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            true_rating = row['rating']
            
            try:
                predicted_rating = model.predict(user_id, movie_id)
                y_true.append(true_rating)
                y_pred.append(predicted_rating)
            except Exception as e:
                logger.warning(f"Erreur de prédiction pour user {user_id}, movie {movie_id}: {e}")
                continue
        
        if not y_true:
            logger.error("Aucune prédiction valide")
            return {}
        
        # Calculer les métriques
        results = {
            'rmse': self.metrics.rmse(y_true, y_pred),
            'mae': self.metrics.mae(y_true, y_pred),
            'n_predictions': len(y_true)
        }
        
        # Corrélations
        try:
            pearson_corr, _ = pearsonr(y_true, y_pred)
            spearman_corr, _ = spearmanr(y_true, y_pred)
            results['pearson_correlation'] = pearson_corr
            results['spearman_correlation'] = spearman_corr
        except Exception as e:
            logger.warning(f"Erreur calcul corrélation: {e}")
            results['pearson_correlation'] = 0.0
            results['spearman_correlation'] = 0.0
        
        return results
    
    def evaluate_recommendation_quality(self, model, test_data: pd.DataFrame, ratings_data: pd.DataFrame, k_values: List[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Évalue la qualité des recommandations
        
        Args:
            model: Modèle à évaluer
            test_data: Données de test
            ratings_data: Toutes les données de ratings
            k_values: Valeurs de k à tester
            
        Returns:
            Dictionnaire des métriques par k
        """
        if k_values is None:
            k_values = self.metrics.k_values
        
        logger.info("Évaluation de la qualité des recommandations...")
        
        # Grouper les données de test par utilisateur
        user_test_items = test_data.groupby('userId')['movieId'].apply(list).to_dict()
        
        # Identifier les items pertinents (rating >= seuil)
        relevant_items_by_user = {}
        for user_id, items in user_test_items.items():
            user_test_ratings = test_data[test_data['userId'] == user_id]
            relevant_items = user_test_ratings[user_test_ratings['rating'] >= self.metrics.rating_threshold]['movieId'].tolist()
            relevant_items_by_user[user_id] = relevant_items
        
        # Calculer les métriques pour chaque k
        results = {}
        
        for k in k_values:
            logger.info(f"Évaluation pour k={k}...")
            
            precision_scores = []
            recall_scores = []
            f1_scores = []
            hit_rates = []
            ap_scores = []
            
            for user_id in user_test_items.keys():
                try:
                    # Obtenir les recommandations
                    recommendations = model.recommend(user_id, ratings_data, k * 2)  # Demander plus pour avoir assez
                    recommended_items = [item_id for item_id, _ in recommendations]
                    
                    relevant_items = relevant_items_by_user.get(user_id, [])
                    
                    if not relevant_items:  # Ignorer les utilisateurs sans items pertinents
                        continue
                    
                    # Calculer les métriques
                    precision = self.metrics.precision_at_k(recommended_items, relevant_items, k)
                    recall = self.metrics.recall_at_k(recommended_items, relevant_items, k)
                    f1 = self.metrics.f1_score_at_k(recommended_items, relevant_items, k)
                    hit_rate = self.metrics.hit_rate_at_k(recommended_items, relevant_items, k)
                    ap = self.metrics.average_precision(recommended_items, relevant_items)
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    hit_rates.append(hit_rate)
                    ap_scores.append(ap)
                    
                except Exception as e:
                    logger.warning(f"Erreur recommandation pour user {user_id}: {e}")
                    continue
            
            # Moyenner les scores
            results[f'k_{k}'] = {
                'precision': np.mean(precision_scores) if precision_scores else 0.0,
                'recall': np.mean(recall_scores) if recall_scores else 0.0,
                'f1_score': np.mean(f1_scores) if f1_scores else 0.0,
                'hit_rate': np.mean(hit_rates) if hit_rates else 0.0,
                'map': np.mean(ap_scores) if ap_scores else 0.0,  # Mean Average Precision
                'n_users_evaluated': len(precision_scores)
            }
        
        return results
    
    def evaluate_diversity_and_novelty(self, model, test_users: List[int], ratings_data: pd.DataFrame, movies_data: pd.DataFrame, k: int = 10) -> Dict[str, float]:
        """
        Évalue la diversité et la nouveauté des recommandations
        
        Args:
            model: Modèle à évaluer
            test_users: Liste des utilisateurs de test
            ratings_data: Données des ratings
            movies_data: Données des films
            k: Nombre de recommandations
            
        Returns:
            Métriques de diversité et nouveauté
        """
        logger.info("Évaluation de la diversité et nouveauté...")
        
        # Calculer la popularité des items
        item_popularity = ratings_data['movieId'].value_counts(normalize=True).to_dict()
        
        # Préparer les features des films pour la diversité
        # Utiliser les genres comme features simples
        if 'genres' in movies_data.columns:
            # One-hot encoding des genres
            genres_expanded = movies_data['genres'].str.get_dummies(sep='|')
            item_features = genres_expanded
        else:
            # Utiliser des features aléatoires si pas de genres
            item_features = pd.DataFrame(np.random.rand(len(movies_data), 10), index=movies_data['movieId'])
        
        diversity_scores = []
        novelty_scores = []
        coverage_items = set()
        
        for user_id in test_users:
            try:
                recommendations = model.recommend(user_id, ratings_data, k)
                recommended_items = [item_id for item_id, _ in recommendations]
                
                if not recommended_items:
                    continue
                
                # Diversité
                diversity = self.metrics.diversity(recommended_items, item_features)
                diversity_scores.append(diversity)
                
                # Nouveauté
                novelty = self.metrics.novelty(recommended_items, item_popularity)
                novelty_scores.append(novelty)
                
                # Collecter pour la couverture
                coverage_items.update(recommended_items)
                
            except Exception as e:
                logger.warning(f"Erreur évaluation diversité pour user {user_id}: {e}")
                continue
        
        # Couverture
        total_items = len(movies_data)
        coverage = len(coverage_items) / total_items if total_items > 0 else 0.0
        
        return {
            'diversity': np.mean(diversity_scores) if diversity_scores else 0.0,
            'novelty': np.mean(novelty_scores) if novelty_scores else 0.0,
            'coverage': coverage,
            'n_users_evaluated': len(diversity_scores)
        }
    
    def comprehensive_evaluation(self, model, test_data: pd.DataFrame, ratings_data: pd.DataFrame, movies_data: pd.DataFrame, model_name: str = "Model") -> Dict[str, Any]:
        """
        Évaluation complète d'un modèle
        
        Args:
            model: Modèle à évaluer
            test_data: Données de test
            ratings_data: Toutes les données de ratings
            movies_data: Données des films
            model_name: Nom du modèle
            
        Returns:
            Résultats complets d'évaluation
        """
        logger.info(f"Évaluation complète du modèle {model_name}...")
        
        results = {
            'model_name': model_name,
            'rating_prediction': {},
            'recommendation_quality': {},
            'diversity_novelty': {}
        }
        
        # 1. Évaluation de la prédiction de ratings
        try:
            results['rating_prediction'] = self.evaluate_rating_prediction(model, test_data)
        except Exception as e:
            logger.error(f"Erreur évaluation prédiction ratings: {e}")
            results['rating_prediction'] = {}
        
        # 2. Évaluation de la qualité des recommandations
        try:
            results['recommendation_quality'] = self.evaluate_recommendation_quality(model, test_data, ratings_data)
        except Exception as e:
            logger.error(f"Erreur évaluation qualité recommandations: {e}")
            results['recommendation_quality'] = {}
        
        # 3. Évaluation de la diversité et nouveauté
        try:
            test_users = test_data['userId'].unique()[:100]  # Limiter pour la performance
            results['diversity_novelty'] = self.evaluate_diversity_and_novelty(model, test_users, ratings_data, movies_data)
        except Exception as e:
            logger.error(f"Erreur évaluation diversité/nouveauté: {e}")
            results['diversity_novelty'] = {}
        
        return results
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare les résultats de plusieurs modèles
        
        Args:
            evaluation_results: Liste des résultats d'évaluation
            
        Returns:
            DataFrame de comparaison
        """
        comparison_data = []
        
        for result in evaluation_results:
            model_name = result.get('model_name', 'Unknown')
            row = {'Model': model_name}
            
            # Métriques de prédiction
            rating_pred = result.get('rating_prediction', {})
            row['RMSE'] = rating_pred.get('rmse', np.nan)
            row['MAE'] = rating_pred.get('mae', np.nan)
            row['Pearson_Corr'] = rating_pred.get('pearson_correlation', np.nan)
            
            # Métriques de recommandation (k=10)
            rec_quality = result.get('recommendation_quality', {})
            k10_metrics = rec_quality.get('k_10', {})
            row['Precision@10'] = k10_metrics.get('precision', np.nan)
            row['Recall@10'] = k10_metrics.get('recall', np.nan)
            row['F1@10'] = k10_metrics.get('f1_score', np.nan)
            row['Hit_Rate@10'] = k10_metrics.get('hit_rate', np.nan)
            row['MAP'] = k10_metrics.get('map', np.nan)
            
            # Métriques de diversité
            div_nov = result.get('diversity_novelty', {})
            row['Diversity'] = div_nov.get('diversity', np.nan)
            row['Novelty'] = div_nov.get('novelty', np.nan)
            row['Coverage'] = div_nov.get('coverage', np.nan)
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_comparison(self, comparison_df: pd.DataFrame, save_path: str = None) -> None:
        """
        Crée des graphiques de comparaison des modèles
        
        Args:
            comparison_df: DataFrame de comparaison
            save_path: Chemin de sauvegarde (optionnel)
        """
        # Configuration du style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comparaison des Modèles de Recommandation', fontsize=16, fontweight='bold')
        
        # Métriques à visualiser
        metrics = [
            ('RMSE', 'RMSE (plus bas = meilleur)'),
            ('Precision@10', 'Précision@10'),
            ('Recall@10', 'Rappel@10'),
            ('Diversity', 'Diversité'),
            ('Novelty', 'Nouveauté'),
            ('Coverage', 'Couverture')
        ]
        
        for i, (metric, title) in enumerate(metrics):
            ax = axes[i // 3, i % 3]
            
            if metric in comparison_df.columns:
                # Filtrer les valeurs non-NaN
                valid_data = comparison_df[comparison_df[metric].notna()]
                
                if not valid_data.empty:
                    bars = ax.bar(valid_data['Model'], valid_data[metric], 
                                 color=plt.cm.Set3(np.linspace(0, 1, len(valid_data))))
                    
                    # Ajouter les valeurs sur les barres
                    for bar, value in zip(bars, valid_data[metric]):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                               f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    ax.set_title(title, fontweight='bold')
                    ax.set_ylabel(metric)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    ax.text(0.5, 0.5, 'Pas de données', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(title, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Métrique non disponible', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graphique sauvegardé: {save_path}")
        
        plt.show()
    
    def save_results(self, evaluation_results: List[Dict[str, Any]], save_dir: str) -> None:
        """
        Sauvegarde les résultats d'évaluation
        
        Args:
            evaluation_results: Résultats d'évaluation
            save_dir: Répertoire de sauvegarde
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder les résultats détaillés
        import json
        with open(save_path / 'detailed_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Sauvegarder le tableau de comparaison
        comparison_df = self.compare_models(evaluation_results)
        comparison_df.to_csv(save_path / 'model_comparison.csv', index=False)
        
        # Créer et sauvegarder les graphiques
        self.plot_comparison(comparison_df, str(save_path / 'model_comparison.png'))
        
        logger.info(f"Résultats sauvegardés dans {save_dir}")

def main():
    """
    Fonction principale pour tester l'évaluation
    """
    # Exemple d'utilisation
    evaluator = ModelEvaluator()
    
    # Charger les données (exemple)
    test_data = pd.read_csv("data/processed/test_data.csv")
    ratings_data = pd.read_csv("data/processed/train_data.csv")
    movies_data = pd.read_csv("data/processed/movies_processed.csv")
    
    # Simuler des résultats d'évaluation
    mock_results = [
        {
            'model_name': 'SVD',
            'rating_prediction': {'rmse': 0.85, 'mae': 0.67, 'pearson_correlation': 0.72},
            'recommendation_quality': {
                'k_10': {'precision': 0.15, 'recall': 0.08, 'f1_score': 0.11, 'hit_rate': 0.65, 'map': 0.12}
            },
            'diversity_novelty': {'diversity': 0.45, 'novelty': 8.2, 'coverage': 0.23}
        },
        {
            'model_name': 'NMF',
            'rating_prediction': {'rmse': 0.92, 'mae': 0.71, 'pearson_correlation': 0.68},
            'recommendation_quality': {
                'k_10': {'precision': 0.12, 'recall': 0.06, 'f1_score': 0.08, 'hit_rate': 0.58, 'map': 0.09}
            },
            'diversity_novelty': {'diversity': 0.52, 'novelty': 8.8, 'coverage': 0.31}
        }
    ]
    
    # Comparer les modèles
    comparison_df = evaluator.compare_models(mock_results)
    print("\nComparaison des modèles:")
    print(comparison_df)
    
    # Créer les graphiques
    evaluator.plot_comparison(comparison_df)
    
    # Sauvegarder les résultats
    evaluator.save_results(mock_results, "results/evaluation")

if __name__ == "__main__":
    main()