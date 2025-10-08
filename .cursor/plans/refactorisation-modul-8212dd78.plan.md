<!-- 8212dd78-2fb7-4ee7-b240-8de2d35f1c82 c848e07e-495a-4c03-9a49-1197cf99a407 -->
# Refactorisation modulaire de l'application ML

## Structure cible

```
aib-ml-assignments/
├── app.py                              # Point d'entrée (~50 lignes)
├── src/
│   ├── __init__.py
│   ├── pages/
│   │   ├── __init__.py
│   │   ├── preprocessing_page.py      # Page preprocessing & exploration (~200 lignes)
│   │   ├── regression_page.py         # Page régression (~250 lignes)
│   │   └── classification_page.py     # Page classification (~250 lignes)
│   └── utils/
│       ├── __init__.py
│       ├── config.py                  # Configuration & styles CSS (~80 lignes)
│       ├── data_loader.py             # Chargement données Titanic (~30 lignes)
│       ├── preprocessing.py           # Logique de preprocessing (~80 lignes)
│       ├── model_factory.py           # Factory pour instancier les modèles (~200 lignes)
│       ├── model_training.py          # Entraînement & évaluation (~150 lignes)
│       ├── visualization.py           # Graphiques (decision boundary, scatter, etc.) (~150 lignes)
│       └── ui_components.py           # Composants UI réutilisables (~100 lignes)
```

## Découpage détaillé

### 1. `app.py` - Point d'entrée minimal

- Import configuration Streamlit depuis `src.utils.config`
- Chargement données via `src.utils.data_loader`
- Affichage header et overview
- Navigation et dispatch vers pages
- Appel des fonctions render des pages selon sélection
- Footer avec learning summary

### 2. `src/utils/config.py` - Configuration centralisée

- `setup_page_config()`: Configuration Streamlit
- `get_css_styles()`: Retourne le CSS personnalisé
- Constantes de configuration (couleurs, tailles, etc.)

### 3. `src/utils/data_loader.py` - Gestion des données

- `load_titanic_data()`: Charge le dataset Titanic (déjà en cache)
- Fonction simple, isolée pour le chargement

### 4. `src/utils/preprocessing.py` - Logique de preprocessing

- `preprocess_data(df, selected_features, missing_age_option, normalize)`: Preprocessing complet
- `encode_categorical_features(df)`: Encodage des features catégorielles
- `get_feature_mapping()`: Retourne le mapping des noms de features
- Retourne X, y, feature_names, scaler, métriques (before_count, after_count)

### 5. `src/utils/model_factory.py` - Factory de modèles

- `get_available_models(problem_type)`: Retourne dict des modèles disponibles
- `create_model(model_name, problem_type, hyperparams)`: Instancie un modèle configuré
- `get_model_descriptions(problem_type)`: Descriptions des modèles pour UI
- Centralise toute la logique d'instanciation des modèles (Classification + Régression)

### 6. `src/utils/model_training.py` - Entraînement et évaluation

- `train_models(X, y, selected_models, hyperparams, test_size, problem_type)`: Entraîne tous les modèles
- `evaluate_classification_model(model, X_train, y_train, X_test, y_test)`: Métriques classification
- `evaluate_regression_model(model, X_train, y_train, X_test, y_test)`: Métriques régression
- Retourne models dict, results list, X_train, X_test, y_train, y_test

### 7. `src/utils/visualization.py` - Visualisations

- `create_decision_boundary_plot(...)`: Decision boundary 2D (déjà existant)
- `create_regression_surface_plot(...)`: Surface 3D régression (déjà existant)
- `create_2d_scatter_plot(...)`: Scatter plot 2D (déjà existant)
- `create_exploration_plots(df, feature)`: Plots pour exploration données
- `create_comparison_chart(results_df, problem_type)`: Chart de comparaison modèles
- `create_confusion_matrix_plot(cm)`: Heatmap confusion matrix
- `create_feature_importance_plot(feature_names, importances, model_name)`: Bar plot importance

### 8. `src/utils/ui_components.py` - Composants UI réutilisables

- `display_metrics_row(results, problem_type)`: Affiche ligne de métriques
- `display_dataset_overview(df)`: Affiche overview du dataset (4 metrics)
- `display_preprocessing_results(before_count, after_count, n_features, feature_list)`: Résultats preprocessing
- `render_hyperparameter_controls(model_name, problem_type, key_prefix)`: Retourne dict hyperparams
- `render_prediction_inputs(selected_features, key_prefix)`: Inputs pour prédiction interactive
- `display_prediction_result(prediction, probability, y, problem_type)`: Affiche résultat prédiction

### 9. `src/pages/preprocessing_page.py` - Page Preprocessing & Exploration

- `render_preprocessing_page(df)`: Fonction principale
  - Exploration interactive (sélection feature, plots, insights)
  - Contrôles de preprocessing (missing values, normalization, feature selection, test size)
  - Appel `preprocess_data()` de utils
  - Stockage dans `st.session_state`
  - Affichage résultats via `display_preprocessing_results()`

### 10. `src/pages/regression_page.py` - Page Régression

- `render_regression_page()`: Fonction principale
  - Vérification preprocessing fait
  - Récupération données depuis session_state
  - Sélection modèles disponibles
  - Hyperparameter tuning via `render_hyperparameter_controls()`
  - Entraînement via `train_models()`
  - Affichage comparaison via `create_comparison_chart()`
  - Analyse best model (plots régression, feature importance)
  - Prédiction interactive via `render_prediction_inputs()` et `display_prediction_result()`

### 11. `src/pages/classification_page.py` - Page Classification

- `render_classification_page()`: Fonction principale
  - Même structure que régression
  - Spécifique: confusion matrix, decision boundary
  - Métriques: accuracy, precision, recall

## Principes de refactorisation

1. **Aucun changement de comportement**: L'application doit fonctionner exactement pareil
2. **Fail-fast**: Pas de try/except sauf API externes, assertions claires
3. **Réutilisabilité**: Fonctions utils peuvent être appelées par n'importe quelle page
4. **Taille fichiers**: Tous < 300 lignes
5. **Imports propres**: Chaque module importe uniquement ce dont il a besoin
6. **Session state**: Continuer à utiliser `st.session_state` pour partager données entre pages

## Fichiers à créer

Total: 13 nouveaux fichiers (2 `__init__.py`, 3 pages, 8 utils) + modification de `app.py`

### To-dos

- [ ] Créer l'arborescence src/ avec sous-dossiers pages/ et utils/ et fichiers __init__.py
- [ ] Créer src/utils/config.py avec configuration Streamlit et styles CSS
- [ ] Créer src/utils/data_loader.py avec fonction de chargement des données
- [ ] Créer src/utils/preprocessing.py avec logique de preprocessing et encodage
- [ ] Créer src/utils/model_factory.py avec factory pattern pour instancier les modèles
- [ ] Créer src/utils/model_training.py avec logique d'entraînement et évaluation
- [ ] Créer src/utils/visualization.py avec toutes les fonctions de visualisation
- [ ] Créer src/utils/ui_components.py avec composants UI réutilisables
- [ ] Créer src/pages/preprocessing_page.py avec la page de preprocessing et exploration
- [ ] Créer src/pages/regression_page.py avec la page de régression
- [ ] Créer src/pages/classification_page.py avec la page de classification
- [ ] Refactoriser app.py pour qu'il soit un simple point d'entrée chargeant et orchestrant les pages
- [ ] Vérifier que l'application fonctionne correctement après refactorisation