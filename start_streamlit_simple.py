#!/usr/bin/env python3
"""
Script de démarrage rapide de l'interface Streamlit
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_streamlit_app_exists():
    """
    Vérifie si l'application Streamlit existe
    """
    app_path = Path("src/streamlit_app/app.py")
    
    if not app_path.exists():
        logger.error(f"❌ Application Streamlit non trouvée: {app_path}")
        return False
    
    logger.info("✅ Application Streamlit trouvée")
    return True

def check_models_exist():
    """
    Vérifie si les modèles existent
    """
    models_dir = Path("data/models")
    
    if not models_dir.exists():
        logger.warning("❌ Répertoire des modèles non trouvé")
        return False
    
    # Vérifier quelques fichiers de modèles
    required_files = [
        "content_based/tfidf_model.pkl",
        "collaborative/svd_model.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = models_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"❌ Fichiers de modèles manquants: {missing_files}")
        return False
    
    logger.info("✅ Modèles trouvés")
    return True

def run_simple_pipeline_if_needed():
    """
    Lance le pipeline simplifié si les modèles n'existent pas
    """
    if not check_models_exist():
        logger.info("🔄 Lancement du pipeline simplifié pour créer les modèles...")
        
        try:
            result = subprocess.run(
                [sys.executable, "run_simple_pipeline.py"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes max
            )
            
            if result.returncode == 0:
                logger.info("✅ Pipeline simplifié terminé avec succès")
                return True
            else:
                logger.error(f"❌ Erreur dans le pipeline: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Timeout du pipeline simplifié")
            return False
        except Exception as e:
            logger.error(f"❌ Erreur lors du lancement du pipeline: {e}")
            return False
    
    return True

def start_streamlit():
    """
    Démarre l'application Streamlit
    """
    logger.info("🎨 Démarrage de l'interface Streamlit...")
    
    try:
        # Commande pour démarrer Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app/app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ]
        
        logger.info("🌐 Interface Streamlit démarrée sur http://localhost:8501")
        logger.info("🛑 Appuyez sur Ctrl+C pour arrêter")
        
        # Démarrer Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt de Streamlit")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage de Streamlit: {e}")

def main():
    """
    Script principal
    """
    logger.info("🎨 DÉMARRAGE RAPIDE DE STREAMLIT")
    logger.info("=" * 40)
    
    # 1. Vérifier l'application Streamlit
    if not check_streamlit_app_exists():
        logger.error("❌ Application Streamlit non trouvée")
        return False
    
    # 2. Vérifier/créer les modèles
    if not run_simple_pipeline_if_needed():
        logger.error("❌ Impossible de préparer les modèles")
        return False
    
    # 3. Démarrer Streamlit
    start_streamlit()
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n👋 Au revoir !")
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        sys.exit(1)