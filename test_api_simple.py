#!/usr/bin/env python3
"""
Test simple de l'API de recommandation
"""

import requests
import json
import time
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_api_health():
    """
    Test de santé de l'API
    """
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API accessible")
            return True
        else:
            logger.error(f"❌ API retourne le code {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Impossible de contacter l'API: {e}")
        return False

def test_recommendations():
    """
    Test des recommandations
    """
    try:
        # Test avec un utilisateur fictif
        payload = {
            "user_id": 1,
            "n_recommendations": 5,
            "model_type": "svd",
            "exclude_seen": True
        }
        response = requests.post(
            "http://localhost:8001/recommendations",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Recommandations reçues: {len(data.get('recommendations', []))}")
            return True
        else:
            logger.error(f"❌ Erreur recommandations: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors du test de recommandations: {e}")
        return False

def test_prediction():
    """
    Test de prédiction
    """
    try:
        # Test avec des IDs fictifs
        payload = {
            "user_id": 1,
            "movie_id": 1,
            "model_type": "svd"
        }
        response = requests.post(
            "http://localhost:8001/predict",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"✅ Prédiction reçue: {data.get('predicted_rating')}")
            return True
        else:
            logger.error(f"❌ Erreur prédiction: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Erreur lors du test de prédiction: {e}")
        return False

def main():
    """
    Test principal de l'API
    """
    logger.info("🧪 DÉBUT DES TESTS API")
    logger.info("=" * 30)
    
    # Vérifier si l'API est accessible
    if not test_api_health():
        logger.error("❌ L'API n'est pas accessible. Assurez-vous qu'elle est démarrée.")
        logger.info("💡 Pour démarrer l'API: python -m uvicorn src.api.main:app --reload")
        return False
    
    # Attendre un peu pour que l'API soit prête
    time.sleep(2)
    
    # Tests des fonctionnalités
    tests_passed = 0
    total_tests = 2
    
    logger.info("🔍 Test des recommandations...")
    if test_recommendations():
        tests_passed += 1
    
    logger.info("🔍 Test des prédictions...")
    if test_prediction():
        tests_passed += 1
    
    # Résumé
    logger.info("\n" + "=" * 30)
    logger.info(f"📊 RÉSULTATS: {tests_passed}/{total_tests} tests réussis")
    
    if tests_passed == total_tests:
        logger.info("✅ Tous les tests sont passés !")
        return True
    else:
        logger.warning(f"⚠️  {total_tests - tests_passed} test(s) ont échoué")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)