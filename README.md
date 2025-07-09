# 🎬 Système de Recommandation de Films

Un système de recommandation de films avancé utilisant des techniques de filtrage collaboratif, de filtrage basé sur le contenu et des approches hybrides.

## 🚀 Fonctionnalités

- **Filtrage Collaboratif** : Recommandations basées sur les préférences des utilisateurs similaires
- **Filtrage Basé sur le Contenu** : Recommandations basées sur les caractéristiques des films
- **Système Hybride** : Combinaison des deux approches pour des recommandations optimales
- **API REST** : Interface API complète pour l'intégration
- **Interface Streamlit** : Interface utilisateur interactive et moderne
- **Évaluation des Modèles** : Métriques de performance détaillées

## 📋 Prérequis

- Python 3.8+
- pip
- Git

## 🛠️ Installation

1. **Cloner le repository**
```bash
git clone <repository-url>
cd "Projet 10  Systèmes de Recommandation"
```

2. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

3. **Préparer les données**
```bash
python src/data_processing/preprocess.py
```

## 🚀 Démarrage Rapide

### Option 1: Démarrage automatique
```bash
# Windows
start_project.bat

# Ou manuellement
python start_streamlit_simple.py
```

### Option 2: Démarrage manuel

1. **Démarrer l'API**
```bash
python -c "import uvicorn; from test_api_mock import app; uvicorn.run(app, host='0.0.0.0', port=8001)"
```

2. **Démarrer Streamlit**
```bash
streamlit run src/streamlit_app/app.py --server.port 8504
```

## 📁 Structure du Projet

```
├── Dataset/                    # Données brutes
├── config/                     # Fichiers de configuration
├── data/
│   ├── models/                # Modèles entraînés
│   ├── processed/             # Données traitées
│   └── raw/                   # Données brutes
├── deployment/                # Configuration de déploiement
│   ├── docker/               # Fichiers Docker
│   └── kubernetes/           # Manifestes Kubernetes
├── notebooks/                 # Notebooks Jupyter
├── src/
│   ├── api/                  # API REST
│   ├── data_processing/      # Traitement des données
│   ├── evaluation/           # Métriques d'évaluation
│   ├── models/               # Modèles de recommandation
│   ├── streamlit_app/        # Interface Streamlit
│   └── utils/                # Utilitaires
└── tests/                    # Tests unitaires
```

## 🔧 API Endpoints

- `GET /health` - Vérification de l'état de l'API
- `POST /recommend` - Obtenir des recommandations pour un utilisateur
- `POST /predict` - Prédire la note d'un utilisateur pour un film

### Exemple d'utilisation de l'API

```python
import requests

# Recommandations
response = requests.post(
    "http://localhost:8001/recommend",
    json={"user_id": 1, "num_recommendations": 10}
)
recommendations = response.json()

# Prédiction
response = requests.post(
    "http://localhost:8001/predict",
    json={"user_id": 1, "movie_id": 123}
)
prediction = response.json()
```

## 📊 Modèles Disponibles

### 1. Filtrage Collaboratif
- **Algorithme** : SVD (Singular Value Decomposition)
- **Avantages** : Capture les patterns complexes dans les données utilisateur
- **Utilisation** : Recommandations basées sur les utilisateurs similaires

### 2. Filtrage Basé sur le Contenu
- **Algorithme** : TF-IDF + Similarité Cosinus
- **Avantages** : Recommandations basées sur les caractéristiques des films
- **Utilisation** : Recommandations pour nouveaux utilisateurs

### 3. Système Hybride
- **Combinaison** : Moyenne pondérée des deux approches
- **Avantages** : Meilleure performance globale
- **Utilisation** : Recommandations optimales

## 🧪 Tests

```bash
# Tests de l'API
python test_api_simple.py

# Tests avec API mock
python test_api_mock.py
```

## 📈 Métriques d'Évaluation

- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Précision@K**
- **Rappel@K**
- **F1-Score@K**

## 🐳 Déploiement

### Docker
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f deployment/kubernetes/
```

## 📝 Configuration

Le fichier `config/config.yaml` contient toutes les configurations du projet :

```yaml
data:
  raw_path: "data/raw/"
  processed_path: "data/processed/"
  
models:
  collaborative:
    algorithm: "SVD"
    n_factors: 100
  content_based:
    vectorizer: "TfidfVectorizer"
    max_features: 5000
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit les changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Votre Nom** - *Développement initial*

## 🙏 Remerciements

- MovieLens pour le dataset
- Scikit-learn pour les algorithmes de machine learning
- Streamlit pour l'interface utilisateur
- FastAPI pour l'API REST

---

**Note** : Ce projet fait partie d'un système de recommandation avancé développé pour démontrer l'utilisation de différentes techniques de machine learning dans le domaine des systèmes de recommandation.