#!/usr/bin/env python3
"""
Interface utilisateur Streamlit pour le système de recommandation de films
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Any
import yaml
from pathlib import Path
import sys

# Configuration de la page
st.set_page_config(
    page_title="Système de Recommandation de Films",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.recommendation-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #e0e0e0;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.error-message {
    background-color: #ffebee;
    color: #c62828;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #c62828;
}
.success-message {
    background-color: #e8f5e8;
    color: #2e7d32;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #2e7d32;
}
</style>
""", unsafe_allow_html=True)

# Configuration de l'API
API_BASE_URL = "http://localhost:8000"

# Cache pour les données
@st.cache_data(ttl=300)  # Cache pendant 5 minutes
def load_config():
    """Charge la configuration"""
    try:
        config_path = Path("../../config/config.yaml")
        if not config_path.exists():
            config_path = Path("config/config.yaml")
        
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {e}")
    
    return {}

@st.cache_data(ttl=300)
def get_api_health():
    """Vérifie l'état de santé de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        return {"status": "error", "error": str(e)}
    
    return {"status": "unavailable"}

@st.cache_data(ttl=300)
def get_available_models():
    """Récupère la liste des modèles disponibles"""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json().get("available_models", [])
    except Exception as e:
        st.error(f"Erreur lors de la récupération des modèles: {e}")
    
    return []

def get_recommendations(user_id: int, model_type: str, n_recommendations: int = 10):
    """Obtient des recommandations pour un utilisateur"""
    try:
        payload = {
            "user_id": user_id,
            "n_recommendations": n_recommendations,
            "model_type": model_type,
            "exclude_seen": True
        }
        
        response = requests.post(
            f"{API_BASE_URL}/recommendations",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération des recommandations: {e}")
        return None

def predict_rating(user_id: int, movie_id: int, model_type: str):
    """Prédit la note qu'un utilisateur donnerait à un film"""
    try:
        payload = {
            "user_id": user_id,
            "movie_id": movie_id,
            "model_type": model_type
        }
        
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {e}")
        return None

def add_rating(user_id: int, movie_id: int, rating: float):
    """Ajoute un nouveau rating"""
    try:
        payload = {
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
            "timestamp": int(datetime.now().timestamp())
        }
        
        response = requests.post(
            f"{API_BASE_URL}/ratings",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de l'ajout du rating: {e}")
        return None

def get_movie_info(movie_id: int):
    """Récupère les informations d'un film"""
    try:
        response = requests.get(f"{API_BASE_URL}/movies/{movie_id}", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération du film: {e}")
        return None

def get_user_profile(user_id: int):
    """Récupère le profil d'un utilisateur"""
    try:
        response = requests.get(f"{API_BASE_URL}/users/{user_id}/profile", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"Erreur API: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Erreur lors de la récupération du profil: {e}")
        return None

def get_metrics():
    """Récupère les métriques de performance"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
            
    except Exception as e:
        return None

def main():
    """Fonction principale de l'application Streamlit"""
    
    # En-tête principal
    st.markdown('<h1 class="main-header">🎬 Système de Recommandation de Films</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["🏠 Accueil", "🎯 Recommandations", "🔮 Prédictions", "⭐ Ajouter un Rating", 
         "👤 Profil Utilisateur", "📊 Métriques", "ℹ️ À propos"]
    )
    
    # Vérification de l'état de l'API
    health = get_api_health()
    if health.get("status") == "healthy":
        st.sidebar.success("✅ API connectée")
    elif health.get("status") == "degraded":
        st.sidebar.warning("⚠️ API en mode dégradé")
    else:
        st.sidebar.error("❌ API non disponible")
        st.error("L'API n'est pas disponible. Veuillez vérifier que le serveur est démarré.")
        return
    
    # Affichage des pages
    if page == "🏠 Accueil":
        show_home_page(health)
    elif page == "🎯 Recommandations":
        show_recommendations_page()
    elif page == "🔮 Prédictions":
        show_predictions_page()
    elif page == "⭐ Ajouter un Rating":
        show_add_rating_page()
    elif page == "👤 Profil Utilisateur":
        show_user_profile_page()
    elif page == "📊 Métriques":
        show_metrics_page()
    elif page == "ℹ️ À propos":
        show_about_page()

def show_home_page(health: Dict[str, Any]):
    """Page d'accueil"""
    st.header("Bienvenue dans le Système de Recommandation de Films")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Recommandations</h3>
            <p>Obtenez des recommandations personnalisées basées sur vos préférences</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔮 Prédictions</h3>
            <p>Prédisez la note que vous donneriez à un film</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>⭐ Ratings</h3>
            <p>Ajoutez vos propres évaluations de films</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # État du système
    st.subheader("État du Système")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if health.get("status") == "healthy" else "🟡" if health.get("status") == "degraded" else "🔴"
        st.metric("Statut API", f"{status_color} {health.get('status', 'unknown').title()}")
    
    with col2:
        models_loaded = health.get("models_loaded", {})
        loaded_count = sum(1 for loaded in models_loaded.values() if loaded)
        st.metric("Modèles Chargés", f"{loaded_count}/{len(models_loaded)}")
    
    with col3:
        db_status = "✅ Connectée" if health.get("database_connected") else "❌ Déconnectée"
        st.metric("Base de Données", db_status)
    
    with col4:
        uptime = health.get("uptime_seconds", 0)
        uptime_str = f"{uptime//3600:.0f}h {(uptime%3600)//60:.0f}m"
        st.metric("Uptime", uptime_str)
    
    # Modèles disponibles
    if models_loaded:
        st.subheader("Modèles Disponibles")
        
        for model_type, is_loaded in models_loaded.items():
            status_icon = "✅" if is_loaded else "❌"
            st.write(f"{status_icon} **{model_type.title()}**: {'Chargé' if is_loaded else 'Non disponible'}")

def show_recommendations_page():
    """Page des recommandations"""
    st.header("🎯 Recommandations Personnalisées")
    
    # Paramètres de recommandation
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.number_input("ID Utilisateur", min_value=1, max_value=100000, value=1, step=1)
    
    with col2:
        available_models = get_available_models()
        if available_models:
            model_type = st.selectbox("Type de Modèle", available_models)
        else:
            st.error("Aucun modèle disponible")
            return
    
    with col3:
        n_recommendations = st.slider("Nombre de Recommandations", min_value=1, max_value=50, value=10)
    
    # Bouton pour obtenir les recommandations
    if st.button("Obtenir des Recommandations", type="primary"):
        with st.spinner("Génération des recommandations..."):
            recommendations = get_recommendations(user_id, model_type, n_recommendations)
        
        if recommendations:
            st.success(f"Recommandations générées en {recommendations['execution_time_ms']:.1f}ms")
            
            # Affichage des recommandations
            st.subheader(f"Recommandations pour l'utilisateur {user_id}")
            
            for i, rec in enumerate(recommendations['recommendations'], 1):
                with st.container():
                    col1, col2, col3 = st.columns([1, 3, 1])
                    
                    with col1:
                        st.write(f"**#{i}**")
                    
                    with col2:
                        st.write(f"**{rec['title']}**")
                        if rec.get('genres'):
                            st.write(f"Genres: {', '.join(rec['genres'])}")
                        if rec.get('year'):
                            st.write(f"Année: {rec['year']}")
                    
                    with col3:
                        score = rec['recommendation_score']
                        st.metric("Score", f"{score:.2f}")
                    
                    st.markdown("---")
            
            # Informations sur la recommandation
            st.info(f"Modèle utilisé: {recommendations['model_used']} | Généré le: {recommendations['timestamp']}")

def show_predictions_page():
    """Page des prédictions"""
    st.header("🔮 Prédiction de Notes")
    
    # Paramètres de prédiction
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_id = st.number_input("ID Utilisateur", min_value=1, max_value=100000, value=1, step=1, key="pred_user")
    
    with col2:
        movie_id = st.number_input("ID Film", min_value=1, max_value=200000, value=1, step=1)
    
    with col3:
        available_models = get_available_models()
        if available_models:
            model_type = st.selectbox("Type de Modèle", available_models, key="pred_model")
        else:
            st.error("Aucun modèle disponible")
            return
    
    # Informations sur le film
    if st.button("Voir les informations du film"):
        movie_info = get_movie_info(movie_id)
        if movie_info:
            st.info(f"**{movie_info['title']}** | Genres: {', '.join(movie_info['genres'])} | Année: {movie_info.get('year', 'N/A')}")
        else:
            st.warning("Film non trouvé")
    
    # Bouton pour prédire
    if st.button("Prédire la Note", type="primary"):
        with st.spinner("Prédiction en cours..."):
            prediction = predict_rating(user_id, movie_id, model_type)
        
        if prediction:
            st.success(f"Prédiction générée en {prediction['execution_time_ms']:.1f}ms")
            
            # Affichage de la prédiction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Note Prédite", f"{prediction['predicted_rating']:.2f}/5.0")
            
            with col2:
                # Barre de progression pour visualiser la note
                progress = prediction['predicted_rating'] / 5.0
                st.progress(progress)
            
            with col3:
                # Interprétation de la note
                rating = prediction['predicted_rating']
                if rating >= 4.0:
                    interpretation = "😍 Excellent"
                elif rating >= 3.5:
                    interpretation = "😊 Très bien"
                elif rating >= 3.0:
                    interpretation = "🙂 Bien"
                elif rating >= 2.5:
                    interpretation = "😐 Moyen"
                else:
                    interpretation = "😞 Décevant"
                
                st.write(f"**{interpretation}**")
            
            st.info(f"Modèle utilisé: {prediction['model_used']} | Généré le: {prediction['timestamp']}")

def show_add_rating_page():
    """Page pour ajouter un rating"""
    st.header("⭐ Ajouter une Évaluation")
    
    # Formulaire d'ajout de rating
    with st.form("add_rating_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            user_id = st.number_input("ID Utilisateur", min_value=1, max_value=100000, value=1, step=1, key="rating_user")
        
        with col2:
            movie_id = st.number_input("ID Film", min_value=1, max_value=200000, value=1, step=1, key="rating_movie")
        
        with col3:
            rating = st.slider("Note", min_value=0.5, max_value=5.0, value=3.0, step=0.5)
        
        # Informations sur le film
        movie_info = None
        if movie_id:
            movie_info = get_movie_info(movie_id)
            if movie_info:
                st.info(f"Film sélectionné: **{movie_info['title']}** | Genres: {', '.join(movie_info['genres'])}")
        
        submitted = st.form_submit_button("Ajouter l'Évaluation", type="primary")
        
        if submitted:
            if movie_info is None:
                st.error("Film non trouvé. Veuillez vérifier l'ID du film.")
            else:
                with st.spinner("Ajout de l'évaluation..."):
                    result = add_rating(user_id, movie_id, rating)
                
                if result:
                    st.success(f"✅ Évaluation ajoutée avec succès! Note: {rating}/5.0 pour '{movie_info['title']}'")
                    
                    # Suggestion de recommandations
                    if st.button("Voir mes nouvelles recommandations"):
                        st.experimental_rerun()

def show_user_profile_page():
    """Page du profil utilisateur"""
    st.header("👤 Profil Utilisateur")
    
    user_id = st.number_input("ID Utilisateur", min_value=1, max_value=100000, value=1, step=1, key="profile_user")
    
    if st.button("Charger le Profil"):
        with st.spinner("Chargement du profil..."):
            profile = get_user_profile(user_id)
        
        if profile:
            st.subheader(f"Profil de l'utilisateur {user_id}")
            
            # Métriques du profil
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Note Moyenne", f"{profile['average_rating']:.2f}/5.0")
            
            with col2:
                st.metric("Total d'Évaluations", profile['total_ratings'])
            
            with col3:
                st.metric("Films Uniques Évalués", profile['unique_movies_rated'])
            
            # Graphique de la distribution des notes (simulé)
            st.subheader("Analyse du Profil")
            
            # Simulation de données pour la démonstration
            rating_distribution = {
                '0.5-1.0': np.random.randint(0, 10),
                '1.5-2.0': np.random.randint(0, 15),
                '2.5-3.0': np.random.randint(5, 25),
                '3.5-4.0': np.random.randint(10, 30),
                '4.5-5.0': np.random.randint(5, 20)
            }
            
            fig = px.bar(
                x=list(rating_distribution.keys()),
                y=list(rating_distribution.values()),
                title="Distribution des Notes",
                labels={'x': 'Plage de Notes', 'y': 'Nombre de Films'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Utilisateur non trouvé ou aucune donnée disponible.")

def show_metrics_page():
    """Page des métriques"""
    st.header("📊 Métriques de Performance")
    
    # Bouton de rafraîchissement
    if st.button("🔄 Actualiser les Métriques"):
        st.cache_data.clear()
    
    metrics = get_metrics()
    
    if metrics:
        # Métriques système
        if 'system_metrics' in metrics and metrics['system_metrics'].get('enabled'):
            st.subheader("Métriques Système")
            
            col1, col2, col3, col4 = st.columns(4)
            
            sys_metrics = metrics['system_metrics']
            
            with col1:
                cpu = sys_metrics.get('cpu_percent', 0)
                st.metric("CPU", f"{cpu:.1f}%", delta=None)
            
            with col2:
                memory = sys_metrics.get('memory_percent', 0)
                st.metric("Mémoire", f"{memory:.1f}%", delta=None)
            
            with col3:
                disk = sys_metrics.get('disk_percent', 0)
                st.metric("Disque", f"{disk:.1f}%", delta=None)
            
            with col4:
                processes = sys_metrics.get('process_count', 0)
                st.metric("Processus", processes)
        
        # Statistiques des requêtes
        if 'request_stats' in metrics:
            st.subheader("Statistiques des Requêtes (Dernière Heure)")
            
            request_stats = metrics['request_stats'].get('last_hour', {})
            
            if request_stats:
                # Tableau des statistiques par endpoint
                stats_data = []
                for endpoint, stats in request_stats.items():
                    stats_data.append({
                        'Endpoint': endpoint,
                        'Requêtes': stats['request_count'],
                        'Temps Moyen (ms)': f"{stats['avg_duration_ms']:.1f}",
                        'P95 (ms)': f"{stats['p95_duration_ms']:.1f}",
                        'Max (ms)': f"{stats['max_duration_ms']:.1f}"
                    })
                
                if stats_data:
                    df_stats = pd.DataFrame(stats_data)
                    st.dataframe(df_stats, use_container_width=True)
                    
                    # Graphique des temps de réponse
                    fig = px.bar(
                        df_stats,
                        x='Endpoint',
                        y='Temps Moyen (ms)',
                        title="Temps de Réponse Moyen par Endpoint"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucune requête dans la dernière heure")
        
        # Performances des modèles
        if 'model_performance' in metrics and metrics['model_performance']:
            st.subheader("Performances des Modèles")
            
            model_perf = metrics['model_performance']
            
            for model_name, perf_metrics in model_perf.items():
                st.write(f"**{model_name}**")
                
                if perf_metrics:
                    cols = st.columns(len(perf_metrics))
                    for i, (metric_name, value) in enumerate(perf_metrics.items()):
                        with cols[i % len(cols)]:
                            st.metric(metric_name.upper(), f"{value:.3f}")
                else:
                    st.write("Aucune métrique disponible")
                
                st.markdown("---")
        
        # Informations de collecte
        if 'collection_info' in metrics:
            st.subheader("Informations de Collecte")
            
            col1, col2, col3 = st.columns(3)
            
            info = metrics['collection_info']
            
            with col1:
                st.metric("Points de Métriques", info.get('total_metric_points', 0))
            
            with col2:
                st.metric("Requêtes Enregistrées", info.get('total_request_records', 0))
            
            with col3:
                st.metric("Rétention (heures)", info.get('retention_hours', 0))
    
    else:
        st.warning("Métriques non disponibles")

def show_about_page():
    """Page à propos"""
    st.header("ℹ️ À Propos du Système")
    
    st.markdown("""
    ## Système de Recommandation de Films
    
    Ce système utilise des techniques avancées d'apprentissage automatique pour fournir des recommandations de films personnalisées.
    
    ### 🎯 Fonctionnalités
    
    - **Filtrage Collaboratif**: Recommandations basées sur les préférences d'utilisateurs similaires
    - **Filtrage par Contenu**: Recommandations basées sur les caractéristiques des films
    - **Système Hybride**: Combinaison des deux approches pour de meilleurs résultats
    - **Prédictions en Temps Réel**: Estimation des notes que vous donneriez aux films
    - **Interface Intuitive**: Interface web facile à utiliser
    
    ### 🛠️ Technologies Utilisées
    
    - **Backend**: FastAPI, Python
    - **Machine Learning**: Scikit-learn, Surprise
    - **Frontend**: Streamlit
    - **Base de Données**: SQLite
    - **Monitoring**: Métriques en temps réel
    
    ### 📊 Dataset
    
    Le système est entraîné sur le dataset MovieLens 20M qui contient:
    - 20 millions de ratings
    - 27,000 films
    - 138,000 utilisateurs
    
    ### 👨‍💻 Développeur
    
    **Dady Akrou Cyrille**  
    Data Scientist  
    Email: cyrilledady0501@gmail.com
    
    ### 📝 Version
    
    Version 1.0.0 - Décembre 2024
    """)
    
    # Configuration système
    config = load_config()
    if config:
        st.subheader("Configuration Système")
        
        with st.expander("Voir la configuration"):
            st.json(config)

if __name__ == "__main__":
    main()