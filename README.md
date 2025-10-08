# 🤖 ML Demo - Application Interactive de Machine Learning

Une application web interactive construite avec Streamlit pour apprendre et expérimenter avec les concepts de Machine Learning, sans écrire une seule ligne de code.

## 📋 Description

Cette application permet d'explorer, de manière interactive et pédagogique, les concepts fondamentaux du Machine Learning à travers le célèbre dataset Titanic. Elle offre une interface intuitive pour :

- **Explorer les données** : Visualiser les distributions et les corrélations entre les caractéristiques
- **Prétraiter les données** : Gérer les valeurs manquantes, normaliser les features, encoder les variables catégorielles
- **Entraîner des modèles** : Tester et comparer jusqu'à 19 algorithmes différents
- **Évaluer les performances** : Analyser les métriques de performance, les matrices de confusion, l'importance des features
- **Prédire en temps réel** : Créer des profils de passagers personnalisés et obtenir des prédictions instantanées

## ✨ Fonctionnalités

### 🔍 Exploration de Données
- Visualisations interactives des distributions
- Analyse de la corrélation entre les features et la survie
- Statistiques descriptives en temps réel

### 🔧 Prétraitement des Données
- Gestion intelligente des valeurs manquantes (médiane, moyenne, suppression)
- Normalisation optionnelle des features
- Encodage automatique des variables catégorielles
- Sélection personnalisée des features à utiliser

### 🤖 Entraînement de Modèles

#### Classification (10 algorithmes)
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gradient Boosting
- AdaBoost
- Naive Bayes
- Neural Network (MLP)
- Ridge Classifier

#### Régression (9 algorithmes)
- Linear Regression
- Random Forest Regressor
- Support Vector Regression (SVR)
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Gradient Boosting Regressor
- Ridge Regression
- Lasso Regression
- Neural Network Regressor (MLP)

### 📊 Visualisations Avancées
- Frontières de décision en 2D (classification)
- Surfaces de régression en 3D
- Matrices de confusion interactives
- Graphiques de comparaison des performances
- Analyse de l'importance des features
- Graphiques de résidus (régression)

### ⚙️ Réglage des Hyperparamètres
Interface intuitive pour ajuster les hyperparamètres de chaque modèle :
- Profondeur des arbres
- Nombre d'estimateurs
- Taux d'apprentissage
- Paramètres de régularisation
- Et bien plus...

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Étapes d'installation

1. **Cloner ou télécharger le projet**
```bash
cd /home/lopilo/code/aib-ml-assignments
```

2. **(Optionnel) Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Linux/Mac
# ou
venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

## 📦 Dépendances

L'application utilise les bibliothèques suivantes :
- `streamlit` >= 1.24.0 - Framework web interactif
- `pandas` >= 1.5.3 - Manipulation de données
- `numpy` >= 1.26.0 - Calculs numériques
- `matplotlib` >= 3.7.1 - Visualisations statiques
- `seaborn` >= 0.12.2 - Visualisations statistiques
- `scikit-learn` >= 1.3.0 - Algorithmes de Machine Learning
- `plotly` >= 5.16.0 - Visualisations interactives

## 🎮 Lancement de l'application

### Lancement Standard (1-10 utilisateurs)

Une fois les dépendances installées, lancez l'application avec :

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur par défaut à l'adresse :
```
http://localhost:8501
```

Si elle ne s'ouvre pas automatiquement, copiez simplement cette URL dans votre navigateur.

### Déploiement Optimisé (50-150 utilisateurs simultanés)

Pour supporter un grand nombre d'utilisateurs simultanés (usage pédagogique en classe), l'application a été optimisée :

**Optimisations incluses :**
- Multiprocessing intelligent pour Random Forest (utilise 4 cores max par utilisateur)
- Réservation de 3 cores CPU pour la stabilité système
- Visualisations optimisées (6x plus rapides)
- Gestion automatique de la charge CPU

**Matériel recommandé pour 100-150 élèves :**
- CPU : i7-12700KF (20 threads) ou équivalent
- RAM : 32 GB minimum
- Connexion réseau stable

**Lancement optimisé :**
```bash
streamlit run app.py --server.maxMessageSize=200
```

**Configuration réseau locale :**
Pour permettre aux élèves de se connecter depuis d'autres ordinateurs du réseau local :

```bash
streamlit run app.py --server.address=0.0.0.0 --server.maxMessageSize=200
```

Les élèves pourront ensuite accéder à l'application via :
```
http://[VOTRE_IP_LOCALE]:8501
```

**Recommandations pédagogiques :**
- Éviter que tous les élèves entraînent des modèles exactement au même moment
- Pour 150 élèves, envisager 2-3 sessions échelonnées de 50-75 élèves
- Privilégier des exercices guidés avec des pauses entre les entraînements

## 📖 Guide d'utilisation

### 1. Choisir le type de problème
Dans la barre latérale, sélectionnez entre :
- **Classification** : Prédire la survie (survécu / non survécu)
- **Régression** : Prédire des valeurs continues (âge, prix du billet, etc.)

### 2. Explorer les données
- Consultez les statistiques générales du dataset
- Sélectionnez une feature à explorer
- Observez les visualisations de distribution et de corrélation

### 3. Prétraiter les données
- Choisissez comment gérer les valeurs manquantes
- Activez/désactivez la normalisation
- Sélectionnez les features à utiliser pour l'entraînement

### 4. Entraîner et comparer les modèles
- Sélectionnez un ou plusieurs modèles à comparer
- Ajustez la taille du jeu de test (10-40%)
- Configurez les hyperparamètres via les sliders interactifs
- Comparez les performances via les métriques et graphiques

### 5. Analyser les résultats
- Examinez la matrice de confusion (classification)
- Visualisez les frontières de décision en 2D
- Consultez l'importance des features
- Analysez les graphiques de résidus (régression)

### 6. Faire des prédictions
- Créez un profil de passager personnalisé
- Obtenez une prédiction instantanée
- Visualisez la confiance du modèle

## 🎯 Cas d'usage

Cette application est idéale pour :
- **Étudiants** découvrant le Machine Learning
- **Enseignants** souhaitant démontrer des concepts ML
- **Data Scientists** explorant rapidement différents algorithmes
- **Curieux** voulant comprendre l'IA de manière interactive

## 🛠️ Structure du projet

```
aib-ml-assignments/
├── app.py              # Application principale Streamlit
├── requirements.txt    # Dépendances Python
└── README.md          # Ce fichier
```

## 📊 Dataset

L'application utilise le dataset Titanic, chargé automatiquement via `seaborn`. Ce dataset contient :
- **891 passagers** avec leurs caractéristiques
- **12 features** incluant l'âge, le sexe, la classe, le prix du billet, etc.
- **Cible** : survie (0 = non survécu, 1 = survécu)

## 🔧 Personnalisation

L'application est conçue pour être facilement extensible. Pour ajouter :
- **Nouveaux modèles** : Ajoutez-les dans la section de sélection et instanciation des modèles
- **Nouveaux datasets** : Modifiez la fonction `load_titanic_data()`
- **Nouvelles visualisations** : Créez de nouvelles fonctions de plotting

## 📝 Notes techniques

- Les modèles sont entraînés avec `random_state=42` pour la reproductibilité
- La stratification est appliquée lors du split train/test en classification
- Le cache Streamlit (`@st.cache_data`) optimise le chargement des données
- Les warnings scikit-learn sont filtrés pour une interface propre

### Optimisations de performance

- **Multiprocessing intelligent** : Random Forest utilise jusqu'à 4 cores CPU (paramètre `n_jobs=4`)
- **Réservation CPU** : 3 cores réservés au système pour éviter les freezes
- **Visualisations optimisées** :
  - Frontières de décision : résolution réduite de 0.02 à 0.05 (6x plus rapide)
  - Surfaces 3D : résolution réduite de 50×50 à 30×30 (3x plus rapide)
- **Architecture** : Chaque utilisateur = 1 session Python indépendante
- **Charge CPU** : Automatiquement répartie entre les cores disponibles

## 🐛 Résolution de problèmes

### L'application ne démarre pas
- Vérifiez que toutes les dépendances sont installées
- Assurez-vous d'utiliser Python 3.8+
- Réinstallez les dépendances : `pip install -r requirements.txt --force-reinstall`

### Erreurs de chargement des données
- Vérifiez votre connexion internet (seaborn télécharge le dataset)
- Le dataset Titanic peut parfois être indisponible, attendez quelques instants

### Performances lentes
- Réduisez le nombre de modèles comparés simultanément
- Diminuez le nombre d'estimateurs pour les modèles d'ensemble
- Utilisez moins de features

## 📄 Licence

Ce projet est open-source et disponible pour un usage éducatif.

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
- Signaler des bugs
- Proposer de nouvelles fonctionnalités
- Améliorer la documentation
- Ajouter de nouveaux modèles ou visualisations

---

**Développé avec ❤️ pour rendre le Machine Learning accessible à tous**

