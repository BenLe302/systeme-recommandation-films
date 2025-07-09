#!/usr/bin/env python3
"""
Module de monitoring et collecte de métriques pour le système de recommandation
Auteur: Dady Akrou Cyrille
Email: cyrilledady0501@gmail.com
"""

import time
import asyncio
import logging
import psutil
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MetricPoint:
    """Point de métrique avec timestamp"""
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)

@dataclass
class RequestMetric:
    """Métrique de requête"""
    endpoint: str
    method: str
    status_code: int
    duration_ms: float
    timestamp: datetime
    user_id: Optional[int] = None
    model_type: Optional[str] = None

class MetricsCollector:
    """Collecteur de métriques pour le système de recommandation"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le collecteur de métriques
        
        Args:
            config: Configuration de l'application
        """
        self.config = config
        self.metrics = defaultdict(lambda: deque(maxlen=1000))  # Garder les 1000 derniers points
        self.request_metrics = deque(maxlen=10000)  # Garder les 10000 dernières requêtes
        self.error_counts = defaultdict(int)
        self.model_performance = defaultdict(dict)
        
        # Configuration du monitoring
        self.monitoring_config = config.get('monitoring', {})
        self.retention_hours = self.monitoring_config.get('retention_hours', 24)
        self.collection_interval = self.monitoring_config.get('collection_interval_seconds', 60)
        
        # Thread pour la collecte automatique
        self._collection_thread = None
        self._stop_collection = threading.Event()
        
        # Métriques système
        self.system_metrics_enabled = self.monitoring_config.get('system_metrics', True)
        
        # Démarrer la collecte automatique
        self.start_collection()
    
    def start_collection(self):
        """Démarre la collecte automatique de métriques"""
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_collection.clear()
            self._collection_thread = threading.Thread(target=self._collect_system_metrics_loop)
            self._collection_thread.daemon = True
            self._collection_thread.start()
            logger.info("Collecte de métriques démarrée")
    
    def stop_collection(self):
        """Arrête la collecte automatique de métriques"""
        self._stop_collection.set()
        if self._collection_thread:
            self._collection_thread.join(timeout=5)
        logger.info("Collecte de métriques arrêtée")
    
    def _collect_system_metrics_loop(self):
        """Boucle de collecte des métriques système"""
        while not self._stop_collection.is_set():
            try:
                if self.system_metrics_enabled:
                    self._collect_system_metrics()
                self._cleanup_old_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Erreur lors de la collecte de métriques: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self):
        """Collecte les métriques système"""
        try:
            now = datetime.now()
            
            # Métriques CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('system_cpu_percent', cpu_percent, now)
            
            # Métriques mémoire
            memory = psutil.virtual_memory()
            self.record_metric('system_memory_percent', memory.percent, now)
            self.record_metric('system_memory_used_mb', memory.used / (1024 * 1024), now)
            self.record_metric('system_memory_available_mb', memory.available / (1024 * 1024), now)
            
            # Métriques disque
            disk = psutil.disk_usage('/')
            self.record_metric('system_disk_percent', disk.percent, now)
            self.record_metric('system_disk_used_gb', disk.used / (1024 * 1024 * 1024), now)
            
            # Métriques réseau (si disponibles)
            try:
                net_io = psutil.net_io_counters()
                self.record_metric('system_network_bytes_sent', net_io.bytes_sent, now)
                self.record_metric('system_network_bytes_recv', net_io.bytes_recv, now)
            except Exception:
                pass  # Métriques réseau non disponibles sur certains systèmes
            
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des métriques système: {e}")
    
    def record_metric(self, name: str, value: float, timestamp: Optional[datetime] = None, 
                     labels: Optional[Dict[str, str]] = None):
        """Enregistre une métrique"""
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_point = MetricPoint(
            value=value,
            timestamp=timestamp,
            labels=labels or {}
        )
        
        self.metrics[name].append(metric_point)
    
    async def record_request(self, endpoint: str, model_type: Optional[str] = None, 
                           user_id: Optional[int] = None):
        """Enregistre une requête"""
        try:
            metric = RequestMetric(
                endpoint=endpoint,
                method="POST",  # La plupart de nos endpoints sont POST
                status_code=200,  # Sera mis à jour si erreur
                duration_ms=0,  # Sera calculé lors de la finalisation
                timestamp=datetime.now(),
                user_id=user_id,
                model_type=model_type
            )
            
            # Compter les requêtes par endpoint
            self.record_metric(f'requests_total_{endpoint}', 1)
            
            # Compter les requêtes par modèle
            if model_type:
                self.record_metric(f'requests_by_model_{model_type}', 1)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la requête: {e}")
    
    async def record_latency(self, endpoint: str, duration_ms: float):
        """Enregistre la latence d'une requête"""
        try:
            self.record_metric(f'latency_{endpoint}_ms', duration_ms)
            self.record_metric('latency_all_ms', duration_ms)
            
            # Enregistrer dans l'historique des requêtes
            metric = RequestMetric(
                endpoint=endpoint,
                method="POST",
                status_code=200,
                duration_ms=duration_ms,
                timestamp=datetime.now()
            )
            self.request_metrics.append(metric)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de la latence: {e}")
    
    async def record_error(self, endpoint: str, error_message: str):
        """Enregistre une erreur"""
        try:
            self.error_counts[f'{endpoint}_errors'] += 1
            self.record_metric(f'errors_{endpoint}', 1)
            self.record_metric('errors_total', 1)
            
            logger.warning(f"Erreur enregistrée pour {endpoint}: {error_message}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement de l'erreur: {e}")
    
    def record_model_performance(self, model_type: str, metrics: Dict[str, float]):
        """Enregistre les performances d'un modèle"""
        try:
            self.model_performance[model_type].update(metrics)
            
            # Enregistrer chaque métrique individuellement
            for metric_name, value in metrics.items():
                self.record_metric(f'model_{model_type}_{metric_name}', value)
            
            logger.info(f"Performances enregistrées pour {model_type}: {len(metrics)} métriques")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement des performances: {e}")
    
    def _cleanup_old_metrics(self):
        """Nettoie les anciennes métriques"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            
            for metric_name, points in self.metrics.items():
                # Supprimer les points trop anciens
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
            
            # Nettoyer les métriques de requêtes
            while self.request_metrics and self.request_metrics[0].timestamp < cutoff_time:
                self.request_metrics.popleft()
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des métriques: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Récupère toutes les métriques"""
        try:
            now = datetime.now()
            
            # Métriques actuelles
            current_metrics = {}
            for name, points in self.metrics.items():
                if points:
                    latest = points[-1]
                    current_metrics[name] = {
                        'value': latest.value,
                        'timestamp': latest.timestamp.isoformat(),
                        'labels': latest.labels
                    }
            
            # Statistiques des requêtes
            request_stats = self._calculate_request_stats()
            
            # Métriques système actuelles
            system_metrics = self._get_current_system_metrics()
            
            # Performances des modèles
            model_metrics = dict(self.model_performance)
            
            # Statistiques d'erreurs
            error_stats = dict(self.error_counts)
            
            return {
                'timestamp': now.isoformat(),
                'current_metrics': current_metrics,
                'request_stats': request_stats,
                'system_metrics': system_metrics,
                'model_performance': model_metrics,
                'error_counts': error_stats,
                'collection_info': {
                    'retention_hours': self.retention_hours,
                    'total_metric_points': sum(len(points) for points in self.metrics.values()),
                    'total_request_records': len(self.request_metrics)
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def _calculate_request_stats(self) -> Dict[str, Any]:
        """Calcule les statistiques des requêtes"""
        try:
            if not self.request_metrics:
                return {}
            
            # Statistiques par endpoint
            endpoint_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0, 'durations': []})
            
            # Statistiques temporelles (dernière heure)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_requests = [r for r in self.request_metrics if r.timestamp > one_hour_ago]
            
            for request in recent_requests:
                stats = endpoint_stats[request.endpoint]
                stats['count'] += 1
                stats['total_duration'] += request.duration_ms
                stats['durations'].append(request.duration_ms)
            
            # Calculer les moyennes et percentiles
            processed_stats = {}
            for endpoint, stats in endpoint_stats.items():
                if stats['count'] > 0:
                    durations = sorted(stats['durations'])
                    processed_stats[endpoint] = {
                        'request_count': stats['count'],
                        'avg_duration_ms': stats['total_duration'] / stats['count'],
                        'p50_duration_ms': self._percentile(durations, 50),
                        'p95_duration_ms': self._percentile(durations, 95),
                        'p99_duration_ms': self._percentile(durations, 99),
                        'max_duration_ms': max(durations),
                        'min_duration_ms': min(durations)
                    }
            
            return {
                'last_hour': processed_stats,
                'total_requests_tracked': len(self.request_metrics),
                'requests_last_hour': len(recent_requests)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des statistiques de requêtes: {e}")
            return {}
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calcule un percentile"""
        if not data:
            return 0.0
        
        k = (len(data) - 1) * percentile / 100
        f = int(k)
        c = k - f
        
        if f + 1 < len(data):
            return data[f] * (1 - c) + data[f + 1] * c
        else:
            return data[f]
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Récupère les métriques système actuelles"""
        try:
            if not self.system_metrics_enabled:
                return {'enabled': False}
            
            return {
                'enabled': True,
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'process_count': len(psutil.pids()),
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métriques système: {e}")
            return {'enabled': True, 'error': str(e)}
    
    async def get_metric_history(self, metric_name: str, hours: int = 1) -> List[Dict[str, Any]]:
        """Récupère l'historique d'une métrique"""
        try:
            if metric_name not in self.metrics:
                return []
            
            cutoff_time = datetime.now() - timedelta(hours=hours)
            points = self.metrics[metric_name]
            
            history = []
            for point in points:
                if point.timestamp > cutoff_time:
                    history.append({
                        'value': point.value,
                        'timestamp': point.timestamp.isoformat(),
                        'labels': point.labels
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []
    
    async def export_metrics(self, format_type: str = 'json') -> str:
        """Exporte les métriques dans différents formats"""
        try:
            metrics_data = await self.get_metrics()
            
            if format_type.lower() == 'json':
                return json.dumps(metrics_data, indent=2, default=str)
            elif format_type.lower() == 'prometheus':
                return self._export_prometheus_format(metrics_data)
            else:
                raise ValueError(f"Format non supporté: {format_type}")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export des métriques: {e}")
            return f"# Erreur lors de l'export: {e}"
    
    def _export_prometheus_format(self, metrics_data: Dict[str, Any]) -> str:
        """Exporte les métriques au format Prometheus"""
        try:
            lines = []
            lines.append("# Métriques du système de recommandation")
            lines.append(f"# Généré le {datetime.now().isoformat()}")
            lines.append("")
            
            # Métriques actuelles
            for name, data in metrics_data.get('current_metrics', {}).items():
                # Nettoyer le nom pour Prometheus
                clean_name = name.replace('-', '_').replace('.', '_')
                lines.append(f"# TYPE {clean_name} gauge")
                lines.append(f"{clean_name} {data['value']}")
                lines.append("")
            
            # Statistiques de requêtes
            request_stats = metrics_data.get('request_stats', {}).get('last_hour', {})
            for endpoint, stats in request_stats.items():
                clean_endpoint = endpoint.replace('/', '_').replace('-', '_')
                lines.append(f"# TYPE requests_total_{clean_endpoint} counter")
                lines.append(f"requests_total_{clean_endpoint} {stats['request_count']}")
                lines.append(f"# TYPE avg_duration_ms_{clean_endpoint} gauge")
                lines.append(f"avg_duration_ms_{clean_endpoint} {stats['avg_duration_ms']}")
                lines.append("")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export Prometheus: {e}")
            return f"# Erreur: {e}"
    
    async def save_metrics_to_file(self, file_path: str, format_type: str = 'json'):
        """Sauvegarde les métriques dans un fichier"""
        try:
            metrics_export = await self.export_metrics(format_type)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(metrics_export)
            
            logger.info(f"Métriques sauvegardées dans {file_path}")
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métriques: {e}")
            raise
    
    def get_health_status(self) -> Dict[str, Any]:
        """Récupère le statut de santé du système de monitoring"""
        try:
            now = datetime.now()
            
            # Vérifier si la collecte fonctionne
            collection_healthy = (
                self._collection_thread is not None and 
                self._collection_thread.is_alive() and 
                not self._stop_collection.is_set()
            )
            
            # Vérifier les métriques récentes
            recent_metrics = 0
            five_minutes_ago = now - timedelta(minutes=5)
            
            for points in self.metrics.values():
                if points and points[-1].timestamp > five_minutes_ago:
                    recent_metrics += 1
            
            # Statut global
            status = "healthy" if collection_healthy and recent_metrics > 0 else "degraded"
            
            return {
                'status': status,
                'timestamp': now.isoformat(),
                'collection_thread_alive': collection_healthy,
                'recent_metrics_count': recent_metrics,
                'total_metrics_tracked': len(self.metrics),
                'total_requests_tracked': len(self.request_metrics),
                'system_metrics_enabled': self.system_metrics_enabled,
                'retention_hours': self.retention_hours
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de la vérification du statut: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Nettoyage lors de la destruction de l'objet"""
        try:
            self.stop_collection()
        except Exception:
            pass

# Fonction utilitaire pour créer une instance de MetricsCollector
def create_metrics_collector(config: Dict[str, Any]) -> MetricsCollector:
    """Crée et retourne une instance de MetricsCollector"""
    return MetricsCollector(config)

if __name__ == "__main__":
    # Test du collecteur de métriques
    import asyncio
    
    async def test_metrics_collector():
        config = {
            'monitoring': {
                'retention_hours': 1,
                'collection_interval_seconds': 5,
                'system_metrics': True
            }
        }
        
        collector = MetricsCollector(config)
        
        # Enregistrer quelques métriques de test
        await collector.record_request('recommendations', 'hybrid_weighted', 123)
        await collector.record_latency('recommendations', 150.5)
        
        collector.record_model_performance('hybrid_weighted', {
            'rmse': 0.85,
            'mae': 0.65,
            'precision_at_10': 0.75
        })
        
        # Attendre un peu pour la collecte
        await asyncio.sleep(2)
        
        # Récupérer les métriques
        metrics = await collector.get_metrics()
        print("Métriques collectées:")
        print(json.dumps(metrics, indent=2, default=str))
        
        # Test de l'export
        prometheus_export = await collector.export_metrics('prometheus')
        print("\nExport Prometheus:")
        print(prometheus_export[:500] + "...")
        
        # Statut de santé
        health = collector.get_health_status()
        print(f"\nStatut de santé: {health}")
        
        # Arrêter la collecte
        collector.stop_collection()
    
    asyncio.run(test_metrics_collector())