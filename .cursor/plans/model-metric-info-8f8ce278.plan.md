<!-- 8f8ce278-8314-4abf-85c0-304637590b9e 01e43d76-4986-492f-b602-7417e0133695 -->
# Ajout de Cartes Informatives pour Modèles et Métriques

## Vue d'ensemble
Ajouter des cartes visuelles et informatives qui s'affichent dynamiquement en fonction des modèles et métriques sélectionnés, avec un design moderne et une animation au survol.

## Étapes d'implémentation

### 1. Créer les données de description des métriques
- Ajouter une fonction `get_metric_descriptions()` dans `src/utils/model_factory.py`
- Inclure pour chaque métrique :
  - Emoji représentatif
  - Description simple
  - Formule mathématique (si applicable)
  - Utilité/Cas d'usage
  - Avantages
  - Inconvénients
- Séparer Classification (Accuracy, Precision, Recall, F1) et Regression (R², RMSE, MAE)

### 2. Créer le composant de cartes pour modèles
- Ajouter fonction `render_model_info_cards()` dans `src/utils/ui_components.py`
- Paramètres : `selected_models`, `problem_type`
- Utiliser `st.columns(4)` pour affichage en grille 4 colonnes
- Récupérer les descriptions depuis `get_model_descriptions()`
- Afficher pour chaque modèle :
  - Emoji + Nom
  - Description
  - Avantages (liste à puces)
  - Meilleur usage

### 3. Créer le composant de cartes pour métriques
- Ajouter fonction `render_metric_info_cards()` dans `src/utils/ui_components.py`
- Paramètres : `selected_metrics`, `problem_type`
- Utiliser `st.columns(4)` pour affichage en grille 4 colonnes
- Récupérer les descriptions depuis `get_metric_descriptions()`
- Afficher pour chaque métrique :
  - Emoji + Nom
  - Formule (en LaTeX si applicable)
  - Description
  - Utilité
  - Avantages/Inconvénients

### 4. Ajouter le CSS pour le design moderne
- Créer fonction `inject_card_styles()` dans `src/utils/ui_components.py`
- Style des cartes :
  - Bordure arrondie
  - Ombre portée légère
  - Fond blanc/gris clair
  - Padding harmonieux
- Animation au survol :
  - Transition douce (transform + box-shadow)
  - Surélévation légère (`translateY(-5px)`)
  - Ombre plus prononcée
  - Effet scale subtil

### 5. Intégrer dans la page Classification
Modifier `src/pages/classification_page.py` :
- Après le multiselect de sélection des modèles (ligne ~67), appeler `render_model_info_cards(selected_models, "Classification")`
- Après le multiselect de sélection des métriques (ligne ~86), appeler `render_metric_info_cards(selected_metrics, "Classification")`
- Injecter le CSS au début de la page avec `inject_card_styles()`

### 6. Intégrer dans la page Regression
Modifier `src/pages/regression_page.py` :
- Après le multiselect de sélection des modèles (ligne ~67), appeler `render_model_info_cards(selected_models, "Regression")`
- Après le multiselect de sélection des métriques (ligne ~86), appeler `render_metric_info_cards(selected_metrics, "Regression")`
- Injecter le CSS au début de la page avec `inject_card_styles()`

## Fichiers à modifier
- `src/utils/model_factory.py` - Ajouter descriptions métriques
- `src/utils/ui_components.py` - Ajouter composants de cartes + CSS
- `src/pages/classification_page.py` - Intégrer les cartes
- `src/pages/regression_page.py` - Intégrer les cartes


### To-dos

- [ ] Créer la fonction get_metric_descriptions() avec descriptions détaillées des métriques
- [ ] Créer render_model_info_cards() pour afficher les cartes de modèles
- [ ] Créer render_metric_info_cards() pour afficher les cartes de métriques
- [ ] Créer inject_card_styles() avec CSS moderne et animation au survol
- [ ] Intégrer les cartes dans classification_page.py
- [ ] Intégrer les cartes dans regression_page.py