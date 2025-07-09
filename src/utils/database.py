#!/usr/bin/env python3
"""
Gestionnaire de base de données pour le système de recommandation
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import asyncio
import logging
import sqlite3
import aiosqlite
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données pour le système de recommandation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de base de données
        
        Args:
            config: Configuration de l'application
        """
        self.config = config
        self.db_path = config.get('database', {}).get('path', 'data/recommendations.db')
        self.connection = None
        
        # Créer le répertoire de la base de données si nécessaire
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self):
        """Initialise la base de données et crée les tables"""
        try:
            self.connection = await aiosqlite.connect(self.db_path)
            await self._create_tables()
            logger.info(f"Base de données initialisée: {self.db_path}")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de la base de données: {e}")
            raise
    
    async def _create_tables(self):
        """Crée les tables nécessaires"""
        tables = {
            'ratings': '''
                CREATE TABLE IF NOT EXISTS ratings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    rating REAL NOT NULL,
                    timestamp INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, movie_id)
                )
            ''',
            'recommendations': '''
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    score REAL NOT NULL,
                    model_type TEXT NOT NULL,
                    rank_position INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'user_sessions': '''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id TEXT NOT NULL,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    actions_count INTEGER DEFAULT 0
                )
            ''',
            'api_logs': '''
                CREATE TABLE IF NOT EXISTS api_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    user_id INTEGER,
                    request_data TEXT,
                    response_status INTEGER,
                    execution_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'model_performance': '''
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'feedback': '''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    movie_id INTEGER NOT NULL,
                    recommendation_id INTEGER,
                    feedback_type TEXT NOT NULL, -- 'like', 'dislike', 'not_interested'
                    feedback_value INTEGER, -- 1 for positive, -1 for negative, 0 for neutral
                    comment TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (recommendation_id) REFERENCES recommendations (id)
                )
            '''
        }
        
        for table_name, create_sql in tables.items():
            try:
                await self.connection.execute(create_sql)
                logger.debug(f"Table {table_name} créée ou vérifiée")
            except Exception as e:
                logger.error(f"Erreur lors de la création de la table {table_name}: {e}")
                raise
        
        await self.connection.commit()
        
        # Créer les index pour améliorer les performances
        await self._create_indexes()
    
    async def _create_indexes(self):
        """Crée les index pour améliorer les performances"""
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id)',
            'CREATE INDEX IF NOT EXISTS idx_ratings_timestamp ON ratings(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_user_id ON recommendations(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_model_type ON recommendations(model_type)',
            'CREATE INDEX IF NOT EXISTS idx_recommendations_created_at ON recommendations(created_at)',
            'CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_logs(endpoint)',
            'CREATE INDEX IF NOT EXISTS idx_api_logs_timestamp ON api_logs(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id)',
            'CREATE INDEX IF NOT EXISTS idx_feedback_movie_id ON feedback(movie_id)'
        ]
        
        for index_sql in indexes:
            try:
                await self.connection.execute(index_sql)
            except Exception as e:
                logger.warning(f"Erreur lors de la création d'un index: {e}")
        
        await self.connection.commit()
    
    async def save_rating(self, rating_data: Dict[str, Any]) -> int:
        """Sauvegarde un rating utilisateur"""
        try:
            sql = '''
                INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
                VALUES (?, ?, ?, ?)
            '''
            
            cursor = await self.connection.execute(
                sql,
                (rating_data['user_id'], rating_data['movie_id'], 
                 rating_data['rating'], rating_data['timestamp'])
            )
            await self.connection.commit()
            
            logger.debug(f"Rating sauvegardé: user {rating_data['user_id']}, movie {rating_data['movie_id']}")
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du rating: {e}")
            raise
    
    async def save_recommendations(self, user_id: int, recommendations: List[Tuple[int, float]], 
                                 model_type: str) -> List[int]:
        """Sauvegarde les recommandations pour un utilisateur"""
        try:
            # Supprimer les anciennes recommandations pour ce modèle
            await self.connection.execute(
                'DELETE FROM recommendations WHERE user_id = ? AND model_type = ?',
                (user_id, model_type)
            )
            
            # Insérer les nouvelles recommandations
            sql = '''
                INSERT INTO recommendations (user_id, movie_id, score, model_type, rank_position)
                VALUES (?, ?, ?, ?, ?)
            '''
            
            recommendation_ids = []
            for rank, (movie_id, score) in enumerate(recommendations, 1):
                cursor = await self.connection.execute(
                    sql, (user_id, movie_id, score, model_type, rank)
                )
                recommendation_ids.append(cursor.lastrowid)
            
            await self.connection.commit()
            
            logger.debug(f"Sauvegardé {len(recommendations)} recommandations pour user {user_id}")
            return recommendation_ids
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des recommandations: {e}")
            raise
    
    async def get_user_ratings(self, user_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Récupère les ratings d'un utilisateur"""
        try:
            sql = 'SELECT * FROM ratings WHERE user_id = ? ORDER BY timestamp DESC'
            if limit:
                sql += f' LIMIT {limit}'
            
            async with self.connection.execute(sql, (user_id,)) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            return pd.DataFrame(rows, columns=columns)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ratings: {e}")
            return pd.DataFrame()
    
    async def get_movie_ratings(self, movie_id: int, limit: Optional[int] = None) -> pd.DataFrame:
        """Récupère les ratings d'un film"""
        try:
            sql = 'SELECT * FROM ratings WHERE movie_id = ? ORDER BY timestamp DESC'
            if limit:
                sql += f' LIMIT {limit}'
            
            async with self.connection.execute(sql, (movie_id,)) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            return pd.DataFrame(rows, columns=columns)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des ratings du film: {e}")
            return pd.DataFrame()
    
    async def get_user_recommendations(self, user_id: int, model_type: Optional[str] = None, 
                                     limit: Optional[int] = None) -> pd.DataFrame:
        """Récupère les recommandations d'un utilisateur"""
        try:
            sql = 'SELECT * FROM recommendations WHERE user_id = ?'
            params = [user_id]
            
            if model_type:
                sql += ' AND model_type = ?'
                params.append(model_type)
            
            sql += ' ORDER BY rank_position ASC'
            
            if limit:
                sql += f' LIMIT {limit}'
            
            async with self.connection.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            return pd.DataFrame(rows, columns=columns)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des recommandations: {e}")
            return pd.DataFrame()
    
    async def log_api_request(self, endpoint: str, method: str, user_id: Optional[int] = None,
                            request_data: Optional[Dict] = None, response_status: int = 200,
                            execution_time_ms: float = 0.0):
        """Enregistre une requête API"""
        try:
            sql = '''
                INSERT INTO api_logs (endpoint, method, user_id, request_data, response_status, execution_time_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            
            request_json = json.dumps(request_data) if request_data else None
            
            await self.connection.execute(
                sql, (endpoint, method, user_id, request_json, response_status, execution_time_ms)
            )
            await self.connection.commit()
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du log API: {e}")
    
    async def save_model_performance(self, model_type: str, metrics: Dict[str, float]):
        """Sauvegarde les métriques de performance d'un modèle"""
        try:
            sql = '''
                INSERT INTO model_performance (model_type, metric_name, metric_value)
                VALUES (?, ?, ?)
            '''
            
            for metric_name, metric_value in metrics.items():
                await self.connection.execute(sql, (model_type, metric_name, metric_value))
            
            await self.connection.commit()
            
            logger.debug(f"Métriques sauvegardées pour {model_type}: {len(metrics)} métriques")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {e}")
            raise
    
    async def save_feedback(self, user_id: int, movie_id: int, feedback_type: str,
                          feedback_value: int, comment: Optional[str] = None,
                          recommendation_id: Optional[int] = None) -> int:
        """Sauvegarde le feedback utilisateur"""
        try:
            sql = '''
                INSERT INTO feedback (user_id, movie_id, recommendation_id, feedback_type, feedback_value, comment)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            
            cursor = await self.connection.execute(
                sql, (user_id, movie_id, recommendation_id, feedback_type, feedback_value, comment)
            )
            await self.connection.commit()
            
            logger.debug(f"Feedback sauvegardé: user {user_id}, movie {movie_id}, type {feedback_type}")
            return cursor.lastrowid
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du feedback: {e}")
            raise
    
    async def get_api_stats(self, hours: int = 24) -> Dict[str, Any]:
        """Récupère les statistiques d'utilisation de l'API"""
        try:
            sql = '''
                SELECT 
                    endpoint,
                    COUNT(*) as request_count,
                    AVG(execution_time_ms) as avg_execution_time,
                    MAX(execution_time_ms) as max_execution_time,
                    MIN(execution_time_ms) as min_execution_time
                FROM api_logs 
                WHERE timestamp >= datetime('now', '-{} hours')
                GROUP BY endpoint
                ORDER BY request_count DESC
            '''.format(hours)
            
            async with self.connection.execute(sql) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            stats = [dict(zip(columns, row)) for row in rows]
            
            # Statistiques globales
            global_sql = '''
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(execution_time_ms) as avg_response_time
                FROM api_logs 
                WHERE timestamp >= datetime('now', '-{} hours')
            '''.format(hours)
            
            async with self.connection.execute(global_sql) as cursor:
                global_row = await cursor.fetchone()
                global_stats = dict(zip([desc[0] for desc in cursor.description], global_row))
            
            return {
                'period_hours': hours,
                'global_stats': global_stats,
                'endpoint_stats': stats,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques API: {e}")
            return {}
    
    async def get_model_performance_history(self, model_type: Optional[str] = None, 
                                          days: int = 30) -> pd.DataFrame:
        """Récupère l'historique des performances des modèles"""
        try:
            sql = '''
                SELECT * FROM model_performance 
                WHERE evaluation_date >= datetime('now', '-{} days')
            '''.format(days)
            
            params = []
            if model_type:
                sql += ' AND model_type = ?'
                params.append(model_type)
            
            sql += ' ORDER BY evaluation_date DESC'
            
            async with self.connection.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            return pd.DataFrame(rows, columns=columns)
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique des performances: {e}")
            return pd.DataFrame()
    
    async def get_user_feedback_stats(self, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Récupère les statistiques de feedback"""
        try:
            sql = '''
                SELECT 
                    feedback_type,
                    COUNT(*) as count,
                    AVG(feedback_value) as avg_value
                FROM feedback
            '''
            
            params = []
            if user_id:
                sql += ' WHERE user_id = ?'
                params.append(user_id)
            
            sql += ' GROUP BY feedback_type'
            
            async with self.connection.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                columns = [description[0] for description in cursor.description]
            
            feedback_stats = [dict(zip(columns, row)) for row in rows]
            
            return {
                'user_id': user_id,
                'feedback_stats': feedback_stats,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques de feedback: {e}")
            return {}
    
    async def cleanup_old_data(self, days: int = 90):
        """Nettoie les anciennes données"""
        try:
            tables_to_clean = [
                ('api_logs', 'timestamp'),
                ('recommendations', 'created_at'),
                ('user_sessions', 'start_time')
            ]
            
            total_deleted = 0
            
            for table, date_column in tables_to_clean:
                sql = f'''
                    DELETE FROM {table} 
                    WHERE {date_column} < datetime('now', '-{days} days')
                '''
                
                cursor = await self.connection.execute(sql)
                deleted_count = cursor.rowcount
                total_deleted += deleted_count
                
                logger.info(f"Supprimé {deleted_count} enregistrements de {table}")
            
            await self.connection.commit()
            
            # Optimiser la base de données
            await self.connection.execute('VACUUM')
            
            logger.info(f"Nettoyage terminé: {total_deleted} enregistrements supprimés")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            raise
    
    async def is_connected(self) -> bool:
        """Vérifie si la connexion à la base de données est active"""
        try:
            if self.connection is None:
                return False
            
            await self.connection.execute('SELECT 1')
            return True
            
        except Exception:
            return False
    
    async def close(self):
        """Ferme la connexion à la base de données"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logger.info("Connexion à la base de données fermée")
    
    async def backup_database(self, backup_path: str):
        """Crée une sauvegarde de la base de données"""
        try:
            backup_conn = await aiosqlite.connect(backup_path)
            await self.connection.backup(backup_conn)
            await backup_conn.close()
            
            logger.info(f"Sauvegarde créée: {backup_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            raise
    
    async def get_database_info(self) -> Dict[str, Any]:
        """Récupère les informations sur la base de données"""
        try:
            info = {
                'database_path': self.db_path,
                'connected': await self.is_connected(),
                'tables': {},
                'total_size_mb': 0
            }
            
            # Taille du fichier
            if Path(self.db_path).exists():
                info['total_size_mb'] = Path(self.db_path).stat().st_size / (1024 * 1024)
            
            # Informations sur les tables
            tables = ['ratings', 'recommendations', 'user_sessions', 'api_logs', 
                     'model_performance', 'feedback']
            
            for table in tables:
                try:
                    async with self.connection.execute(f'SELECT COUNT(*) FROM {table}') as cursor:
                        count = (await cursor.fetchone())[0]
                    info['tables'][table] = {'row_count': count}
                except Exception:
                    info['tables'][table] = {'row_count': 0, 'error': 'Table not accessible'}
            
            return info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations de la base: {e}")
            return {'error': str(e)}

# Fonction utilitaire pour créer une instance de DatabaseManager
def create_database_manager(config: Dict[str, Any]) -> DatabaseManager:
    """Crée et retourne une instance de DatabaseManager"""
    return DatabaseManager(config)

if __name__ == "__main__":
    # Test du gestionnaire de base de données
    import asyncio
    import yaml
    
    async def test_database():
        # Configuration de test
        config = {
            'database': {
                'path': 'test_recommendations.db'
            }
        }
        
        db = DatabaseManager(config)
        await db.initialize()
        
        # Test d'ajout de rating
        rating_data = {
            'user_id': 1,
            'movie_id': 100,
            'rating': 4.5,
            'timestamp': int(datetime.now().timestamp())
        }
        
        rating_id = await db.save_rating(rating_data)
        print(f"Rating sauvegardé avec ID: {rating_id}")
        
        # Test de récupération
        user_ratings = await db.get_user_ratings(1)
        print(f"Ratings de l'utilisateur 1: {len(user_ratings)} ratings")
        
        # Test des statistiques
        stats = await db.get_api_stats(24)
        print(f"Statistiques API: {stats}")
        
        # Informations sur la base
        info = await db.get_database_info()
        print(f"Informations base: {info}")
        
        await db.close()
        
        # Nettoyer le fichier de test
        Path('test_recommendations.db').unlink(missing_ok=True)
    
    asyncio.run(test_database())