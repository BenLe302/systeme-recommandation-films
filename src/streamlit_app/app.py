
import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
from datetime import datetime
import time

# Configuration de la page
st.set_page_config(
    page_title="🎬 CineRecommend - Système de Recommandation",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #FF6B6B;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    .search-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .genre-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 0.2rem 0.5rem;
        border-radius: 15px;
        margin: 0.1rem;
        font-size: 0.8rem;
        display: inline-block;
    }
    .rating-star {
        color: #ffd700;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Fonctions utilitaires
@st.cache_data
def load_movies_data():
    """Charge les données des films"""
    try:
        # Essayer de charger les données traitées
        movies_path = Path("data/processed/movies_processed.csv")
        if movies_path.exists():
            return pd.read_csv(movies_path)
        
        # Sinon charger les données brutes
        movies_path = Path("Dataset/movies.csv")
        if movies_path.exists():
            df = pd.read_csv(movies_path)
            # Traitement basique des données
            df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
            df['title_clean'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
            df['genres_list'] = df['genres'].str.split('|')
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return pd.DataFrame()

@st.cache_data
def load_ratings_data():
    """Charge les données des ratings"""
    try:
        # Essayer plusieurs chemins possibles
        for path in ["data/processed/train_data.csv", "Dataset/ratings.csv", "Dataset/rating.csv"]:
            if Path(path).exists():
                return pd.read_csv(path)
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def call_api(endpoint, data):
    """Appelle l'API de recommandation"""
    try:
        response = requests.post(
            f"http://localhost:8001/{endpoint}",
            json=data,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur API: {response.status_code}")
            return None
    except requests.exceptions.ConnectionError:
        st.warning("⚠️ API non disponible. Assurez-vous que l'API est lancée sur le port 8000.")
        return None
    except Exception as e:
        st.error(f"Erreur lors de l'appel API: {e}")
        return None

def check_api_status():
    """Vérifie le statut de l'API"""
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def display_movie_card(movie, rating_data=None):
    """Affiche une carte de film"""
    with st.container():
        st.markdown('<div class="movie-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            title = movie.get('title_clean', movie.get('title', 'Titre inconnu'))
            year = movie.get('year', 'Année inconnue')
            st.markdown(f"**🎬 {title}** ({year})")
            
            # Genres
            genres = movie.get('genres', '').split('|') if isinstance(movie.get('genres'), str) else []
            if genres and genres[0]:
                genre_html = ' '.join([f'<span class="genre-tag">{genre}</span>' for genre in genres])
                st.markdown(genre_html, unsafe_allow_html=True)
        
        with col2:
            # Affichage des ratings si disponibles
            if rating_data is not None and not rating_data.empty:
                movie_ratings = rating_data[rating_data['movieId'] == movie.get('movieId', 0)]
                if not movie_ratings.empty:
                    avg_rating = movie_ratings['rating'].mean()
                    num_ratings = len(movie_ratings)
                    stars = '⭐' * int(avg_rating)
                    st.markdown(f"**{avg_rating:.1f}/5** {stars}")
                    st.markdown(f"*{num_ratings} avis*")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Chargement des données
movies_df = load_movies_data()
ratings_df = load_ratings_data()

# Header principal
st.markdown('<h1 class="main-header">🎬 CineRecommend</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Votre assistant personnel pour découvrir les meilleurs films</p>', unsafe_allow_html=True)

# Vérification du statut de l'API
api_status = check_api_status()
if api_status:
    st.success("✅ API connectée et opérationnelle")
else:
    st.warning("⚠️ API non disponible - Certaines fonctionnalités seront limitées")

# Sidebar pour la navigation
st.sidebar.title("🎭 Navigation")
page = st.sidebar.selectbox(
    "Choisir une section",
    ["🏠 Accueil", "🔍 Recherche & Filtres", "⭐ Recommandations", "📊 Statistiques", "🎯 Découverte", "📈 Tendances"]
)

# === PAGE ACCUEIL ===
if page == "🏠 Accueil":
    st.header("🏠 Tableau de Bord")
    
    # Métriques principales
    if not movies_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Films disponibles", f"{len(movies_df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            unique_genres = set()
            for genres in movies_df['genres'].dropna():
                unique_genres.update(genres.split('|'))
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Genres", len(unique_genres))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            years = movies_df['year'].dropna()
            if not years.empty:
                year_range = f"{int(years.min())}-{int(years.max())}"
            else:
                year_range = "N/A"
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Période", year_range)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            if not ratings_df.empty:
                total_ratings = len(ratings_df)
            else:
                total_ratings = "N/A"
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Évaluations", f"{total_ratings:,}" if isinstance(total_ratings, int) else total_ratings)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Films populaires récents
    st.subheader("🔥 Films Populaires")
    if not movies_df.empty:
        recent_movies = movies_df[movies_df['year'] >= 2000].head(10) if 'year' in movies_df.columns else movies_df.head(10)
        for _, movie in recent_movies.iterrows():
            display_movie_card(movie, ratings_df)

# === PAGE RECHERCHE & FILTRES ===
elif page == "🔍 Recherche & Filtres":
    st.header("🔍 Recherche & Filtres Avancés")
    
    # Zone de recherche
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    search_term = st.text_input("🔍 Rechercher un film:", placeholder="Tapez le titre d'un film...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not movies_df.empty:
        # Filtres
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Filtre par genre
            all_genres = set()
            for genres in movies_df['genres'].dropna():
                all_genres.update(genres.split('|'))
            all_genres = sorted(list(all_genres))
            
            selected_genres = st.multiselect("🎭 Genres:", all_genres)
        
        with col2:
            # Filtre par année
            if 'year' in movies_df.columns:
                years = movies_df['year'].dropna()
                if not years.empty:
                    min_year, max_year = int(years.min()), int(years.max())
                    year_range = st.slider("📅 Période:", min_year, max_year, (min_year, max_year))
                else:
                    year_range = None
            else:
                year_range = None
        
        with col3:
            # Filtre par note (si disponible)
            if not ratings_df.empty and 'movieId' in ratings_df.columns:
                movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
                min_rating = st.slider("⭐ Note minimale:", 0.0, 5.0, 0.0, 0.1)
                min_reviews = st.number_input("📝 Nombre minimum d'avis:", 0, 1000, 0)
            else:
                min_rating = 0.0
                min_reviews = 0
        
        # Application des filtres
        filtered_df = movies_df.copy()
        
        # Filtre par terme de recherche
        if search_term:
            mask = filtered_df['title'].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]
        
        # Filtre par genres
        if selected_genres:
            mask = filtered_df['genres'].apply(
                lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False
            )
            filtered_df = filtered_df[mask]
        
        # Filtre par année
        if year_range and 'year' in filtered_df.columns:
            mask = (filtered_df['year'] >= year_range[0]) & (filtered_df['year'] <= year_range[1])
            filtered_df = filtered_df[mask]
        
        # Filtre par note
        if not ratings_df.empty and (min_rating > 0 or min_reviews > 0):
            movie_ratings = ratings_df.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
            valid_movies = movie_ratings[
                (movie_ratings['mean'] >= min_rating) & 
                (movie_ratings['count'] >= min_reviews)
            ]['movieId']
            filtered_df = filtered_df[filtered_df['movieId'].isin(valid_movies)]
        
        # Affichage des résultats
        st.subheader(f"📋 Résultats ({len(filtered_df)} films trouvés)")
        
        if len(filtered_df) > 0:
            # Pagination
            items_per_page = 10
            total_pages = (len(filtered_df) - 1) // items_per_page + 1
            
            if total_pages > 1:
                page_num = st.selectbox("Page:", range(1, total_pages + 1))
                start_idx = (page_num - 1) * items_per_page
                end_idx = start_idx + items_per_page
                page_df = filtered_df.iloc[start_idx:end_idx]
            else:
                page_df = filtered_df
            
            for _, movie in page_df.iterrows():
                display_movie_card(movie, ratings_df)
        else:
            st.info("Aucun film ne correspond à vos critères de recherche.")

# === PAGE RECOMMANDATIONS ===
elif page == "⭐ Recommandations":
    st.header("⭐ Recommandations Personnalisées")
    
    if api_status:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_id = st.number_input("👤 ID Utilisateur:", min_value=1, value=1, help="Entrez votre ID utilisateur")
            n_recommendations = st.slider("📊 Nombre de recommandations:", 1, 20, 10)
        
        with col2:
            st.info("💡 **Astuce:** Plus vous avez évalué de films, meilleures seront vos recommandations!")
        
        if st.button("🎯 Obtenir mes Recommandations", type="primary"):
            with st.spinner("🔄 Génération de vos recommandations personnalisées..."):
                recommendations = call_api("recommendations", {
                    "user_id": user_id,
                    "n_recommendations": n_recommendations
                })
                
                if recommendations:
                    st.success("✅ Recommandations générées avec succès!")
                    
                    rec_list = recommendations.get('recommendations', [])
                    if rec_list:
                        st.subheader(f"🎬 Vos {len(rec_list)} Recommandations")
                        
                        for i, rec in enumerate(rec_list, 1):
                            with st.expander(f"#{i} - {rec.get('title', 'Film inconnu')}", expanded=i<=3):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**Titre:** {rec.get('title', 'N/A')}")
                                    if 'genres' in rec:
                                        st.write(f"**Genres:** {rec['genres']}")
                                    if 'year' in rec:
                                        st.write(f"**Année:** {rec['year']}")
                                
                                with col2:
                                    if 'score' in rec:
                                        score = rec['score']
                                        st.metric("Score de recommandation", f"{score:.2f}")
                                    
                                    # Bouton pour prédire la note
                                    if st.button(f"🔮 Prédire ma note", key=f"predict_{i}"):
                                        prediction = call_api("predict", {
                                            "user_id": user_id,
                                            "movie_id": rec.get('movieId', rec.get('movie_id', 0))
                                        })
                                        if prediction:
                                            predicted_rating = prediction.get('predicted_rating', 0)
                                            st.success(f"Note prédite: ⭐ {predicted_rating:.1f}/5")
                    else:
                        st.warning("Aucune recommandation disponible.")
                else:
                    st.error("Impossible de générer des recommandations.")
    else:
        st.error("🚫 API non disponible. Impossible de générer des recommandations.")
        st.info("Pour utiliser cette fonctionnalité, lancez l'API avec: `python simple_api.py`")

# === PAGE STATISTIQUES ===
elif page == "📊 Statistiques":
    st.header("📊 Statistiques et Analyses")
    
    if not movies_df.empty:
        # Distribution des genres
        st.subheader("🎭 Distribution des Genres")
        genre_counts = {}
        for genres in movies_df['genres'].dropna():
            for genre in genres.split('|'):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if genre_counts:
            genre_df = pd.DataFrame(list(genre_counts.items()), columns=['Genre', 'Nombre'])
            genre_df = genre_df.sort_values('Nombre', ascending=False)
            
            fig = px.bar(genre_df.head(15), x='Genre', y='Nombre', 
                        title="Top 15 des Genres les Plus Populaires")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution par année
        if 'year' in movies_df.columns:
            st.subheader("📅 Films par Année")
            year_counts = movies_df['year'].value_counts().sort_index()
            
            fig = px.line(x=year_counts.index, y=year_counts.values,
                         title="Évolution du Nombre de Films par Année")
            fig.update_layout(xaxis_title="Année", yaxis_title="Nombre de Films")
            st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques des ratings
        if not ratings_df.empty:
            st.subheader("⭐ Statistiques des Évaluations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution des notes
                if 'rating' in ratings_df.columns:
                    fig = px.histogram(ratings_df, x='rating', nbins=20,
                                     title="Distribution des Notes")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top films les mieux notés
                movie_stats = ratings_df.groupby('movieId').agg({
                    'rating': ['mean', 'count']
                }).round(2)
                movie_stats.columns = ['avg_rating', 'num_ratings']
                movie_stats = movie_stats[movie_stats['num_ratings'] >= 50]  # Au moins 50 avis
                top_movies = movie_stats.sort_values('avg_rating', ascending=False).head(10)
                
                if not top_movies.empty:
                    # Joindre avec les titres
                    top_movies = top_movies.merge(
                        movies_df[['movieId', 'title']], 
                        left_index=True, 
                        right_on='movieId', 
                        how='left'
                    )
                    
                    st.write("**🏆 Top 10 Films les Mieux Notés**")
                    for _, movie in top_movies.iterrows():
                        title = movie['title'][:30] + "..." if len(str(movie['title'])) > 30 else movie['title']
                        st.write(f"⭐ {movie['avg_rating']:.1f} - {title} ({int(movie['num_ratings'])} avis)")

# === PAGE DÉCOUVERTE ===
elif page == "🎯 Découverte":
    st.header("🎯 Découverte de Films")
    
    # Sélection aléatoire
    st.subheader("🎲 Film Aléatoire")
    if st.button("🎲 Découvrir un Film au Hasard"):
        if not movies_df.empty:
            random_movie = movies_df.sample(1).iloc[0]
            display_movie_card(random_movie, ratings_df)
    
    # Films par décennie
    st.subheader("📅 Explorer par Décennie")
    if 'year' in movies_df.columns:
        decades = []
        for year in movies_df['year'].dropna():
            decade = int(year // 10 * 10)
            decades.append(f"{decade}s")
        
        unique_decades = sorted(set(decades), reverse=True)
        selected_decade = st.selectbox("Choisir une décennie:", unique_decades)
        
        if selected_decade:
            decade_start = int(selected_decade[:-1])
            decade_movies = movies_df[
                (movies_df['year'] >= decade_start) & 
                (movies_df['year'] < decade_start + 10)
            ].sample(min(5, len(movies_df)))
            
            st.write(f"**Films des années {selected_decade}:**")
            for _, movie in decade_movies.iterrows():
                display_movie_card(movie, ratings_df)
    
    # Quiz de recommandation
    st.subheader("🧠 Quiz de Recommandation")
    st.write("Répondez à quelques questions pour obtenir des recommandations personnalisées:")
    
    with st.form("quiz_form"):
        fav_genre = st.selectbox("Genre préféré:", 
                                ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller"])
        mood = st.radio("Humeur du moment:", 
                       ["Détendu", "Aventureux", "Romantique", "Intense", "Nostalgique"])
        duration_pref = st.slider("Durée préférée (minutes):", 60, 180, 120)
        
        if st.form_submit_button("🎯 Obtenir des Suggestions"):
            # Logique simple de recommandation basée sur les préférences
            filtered = movies_df[movies_df['genres'].str.contains(fav_genre, na=False)]
            suggestions = filtered.sample(min(3, len(filtered))) if not filtered.empty else movies_df.sample(3)
            
            st.success("Voici vos suggestions basées sur vos préférences:")
            for _, movie in suggestions.iterrows():
                display_movie_card(movie, ratings_df)

# === PAGE TENDANCES ===
elif page == "📈 Tendances":
    st.header("📈 Tendances et Analyses Avancées")
    
    if not movies_df.empty:
        # Évolution des genres dans le temps
        st.subheader("🎭 Évolution des Genres dans le Temps")
        
        if 'year' in movies_df.columns:
            # Créer un dataset pour l'analyse temporelle des genres
            genre_year_data = []
            for _, movie in movies_df.iterrows():
                if pd.notna(movie['year']) and pd.notna(movie['genres']):
                    year = int(movie['year'])
                    for genre in movie['genres'].split('|'):
                        genre_year_data.append({'year': year, 'genre': genre})
            
            if genre_year_data:
                genre_year_df = pd.DataFrame(genre_year_data)
                
                # Top genres par décennie
                genre_year_df['decade'] = (genre_year_df['year'] // 10) * 10
                decade_genre = genre_year_df.groupby(['decade', 'genre']).size().reset_index(name='count')
                
                # Sélection de la décennie
                decades = sorted(decade_genre['decade'].unique(), reverse=True)
                selected_decade = st.selectbox("Analyser la décennie:", decades)
                
                decade_data = decade_genre[decade_genre['decade'] == selected_decade]
                decade_data = decade_data.sort_values('count', ascending=False).head(10)
                
                fig = px.pie(decade_data, values='count', names='genre',
                           title=f"Répartition des Genres dans les années {int(selected_decade)}s")
                st.plotly_chart(fig, use_container_width=True)
        
        # Analyse de la popularité
        if not ratings_df.empty:
            st.subheader("📊 Analyse de Popularité")
            
            # Films les plus évalués
            movie_popularity = ratings_df['movieId'].value_counts().head(20)
            popular_movies = movies_df[movies_df['movieId'].isin(movie_popularity.index)]
            
            if not popular_movies.empty:
                # Joindre avec les comptes
                popular_movies = popular_movies.merge(
                    movie_popularity.to_frame('num_ratings'),
                    left_on='movieId',
                    right_index=True
                )
                
                fig = px.bar(popular_movies.head(10), 
                           x='num_ratings', 
                           y='title',
                           orientation='h',
                           title="Top 10 des Films les Plus Évalués")
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Matrice de corrélation des genres (si suffisamment de données)
        st.subheader("🔗 Relations entre Genres")
        
        # Créer une matrice de co-occurrence des genres
        genre_matrix = {}
        all_genres = set()
        
        for genres in movies_df['genres'].dropna():
            genre_list = genres.split('|')
            all_genres.update(genre_list)
            
            for i, genre1 in enumerate(genre_list):
                for j, genre2 in enumerate(genre_list):
                    if i != j:
                        key = tuple(sorted([genre1, genre2]))
                        genre_matrix[key] = genre_matrix.get(key, 0) + 1
        
        if genre_matrix:
            # Afficher les combinaisons les plus fréquentes
            top_combinations = sorted(genre_matrix.items(), key=lambda x: x[1], reverse=True)[:10]
            
            st.write("**🔗 Combinaisons de Genres les Plus Fréquentes:**")
            for (genre1, genre2), count in top_combinations:
                st.write(f"• {genre1} + {genre2}: {count} films")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🎬 CineRecommend**")
    st.markdown("Système de recommandation intelligent")

with col2:
    st.markdown("**🔧 Fonctionnalités**")
    st.markdown("• Recherche avancée\n• Recommandations IA\n• Analyses statistiques")

with col3:
    st.markdown("**👨‍💻 Développé par**")
    st.markdown("Dady Akrou Cyrille")
    st.markdown(f"*Dernière mise à jour: {datetime.now().strftime('%d/%m/%Y')}*")

# Message de statut en bas
if api_status:
    st.success("🟢 Toutes les fonctionnalités sont disponibles")
else:
    st.warning("🟡 Mode hors ligne - Fonctionnalités limitées")
