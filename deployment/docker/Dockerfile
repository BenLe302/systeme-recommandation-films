# Dockerfile pour le système de recommandation de films
# Auteur: Dady Akrou Cyrille
# Email: cyrilledady0501@gmail.com

# Image de base Python 3.9 slim pour réduire la taille
FROM python:3.9-slim

# Métadonnées
LABEL maintainer="cyrilledady0501@gmail.com"
LABEL description="Système de Recommandation de Films - API FastAPI"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Création d'un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Définition du répertoire de travail
WORKDIR /app

# Copie des fichiers de requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

# Création des répertoires nécessaires
RUN mkdir -p logs models temp && \
    chown -R appuser:appuser /app

# Changement vers l'utilisateur non-root
USER appuser

# Exposition du port
EXPOSE 8000

# Variables d'environnement pour l'application
ENV PYTHONPATH=/app \
    CONFIG_PATH=/app/config/config.yaml \
    LOG_LEVEL=INFO

# Commande de santé
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Point d'entrée
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]