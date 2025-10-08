<!-- e99bf1d3-5dfb-43bf-9bbe-244e562421bf 9ba88954-97d6-4c9c-ab0c-7832d5af8f25 -->
# Réorganisation en 3 onglets Streamlit

## Structure cible

L'application sera divisée en 3 onglets via `st.tabs()` :

1. **Onglet "Preprocessing & Exploration"** : Exploration des données + préprocessing des features
2. **Onglet "Régression"** : Entraînement + validation + prédictions pour modèles de régression
3. **Onglet "Classification"** : Entraînement + validation + prédictions pour modèles de classification

## Changements principaux dans `app.py`

### 1. Suppression du bouton radio problem_type

- Supprimer lignes 264-268 (sélection du type de problème)
- Les onglets remplacent ce choix

### 2. Création de la structure en onglets

Après l'affichage des métriques générales (lignes 273-281), ajouter :

```python
tab1, tab2, tab3 = st.tabs(["📊 Preprocessing & Exploration", "📈 Régression", "🎯 Classification"])
```

### 3. Onglet 1 - Preprocessing & Exploration

Contenu (lignes 283-406 actuelles) :

- Section "Interactive Data Exploration" (lignes 288-333)
- Section "Data Preprocessing" (lignes 335-406)
- Stocker les données préprocessées dans `st.session_state` pour partage entre onglets :
  - `st.session_state.X` : features préprocessées
  - `st.session_state.y` : target (survived)
  - `st.session_state.feature_names` : noms des features
  - `st.session_state.scaler` : scaler utilisé (si normalisation)
  - `st.session_state.normalize_features` : booléen

### 4. Onglet 2 - Régression

Contenu adapté pour la régression uniquement :

- Sidebar : sélection des modèles de régression (lignes 428-440)
- Hyperparamètres : uniquement pour modèles de régression (lignes 547-571)
- Entraînement des modèles : lignes 758-818 (partie régression)
- Métriques : R², MSE, MAE, RMSE (lignes 843-859)
- Visualisations : Actual vs Predicted, Residuals, Surface 3D (lignes 981-1017)
- Feature importance/coefficients (lignes 1036-1067)
- Section "Make Your Own Predictions" adaptée régression (lignes 1068-1168)

### 5. Onglet 3 - Classification

Contenu adapté pour la classification uniquement :

- Sidebar : sélection des modèles de classification (lignes 414-427)
- Hyperparamètres : uniquement pour modèles de classification (lignes 469-545)
- Entraînement des modèles : lignes 686-757 (partie classification)
- Métriques : Accuracy, Precision, Recall (lignes 828-842)
- Visualisations : Decision boundaries, Confusion matrix (lignes 927-979, 963-1035)
- Feature importance/coefficients (lignes 1036-1067)
- Section "Make Your Own Predictions" adaptée classification (lignes 1068-1168)

### 6. Gestion du state avec session_state

Ajouter vérifications pour éviter erreurs si preprocessing non fait :

```python
if 'X' not in st.session_state:
    st.warning("⚠️ Veuillez d'abord configurer le preprocessing dans l'onglet 'Preprocessing & Exploration'")
    st.stop()
```

## Détails techniques

### Fonctions helpers à conserver

- `create_decision_boundary_plot()` (lignes 69-119) : pour classification
- `create_regression_surface_plot()` (lignes 121-171) : pour régression  
- `create_2d_scatter_plot()` (lignes 173-231) : pour les deux

### Sidebar dynamique

La sidebar doit s'adapter selon l'onglet actif :

- Onglet 1 : Options de preprocessing uniquement
- Onglet 2 : Sélection modèles régression + hyperparamètres
- Onglet 3 : Sélection modèles classification + hyperparamètres

### Target variable

Pour le moment : `y = survived` pour les deux types (même si c'est une valeur binaire)

- La régression prédira des valeurs continues proches de 0 ou 1
- À terme : permettre de choisir la target (age, fare, etc.)

## Structure finale du code

```
main()
  ├─ Header + Info box
  ├─ Load data
  ├─ Dataset overview (métriques)
  └─ Tabs
      ├─ Tab 1: Preprocessing & Exploration
      │   ├─ Interactive Data Exploration
      │   ├─ Data Preprocessing
      │   └─ Save to session_state
      ├─ Tab 2: Régression
      │   ├─ Check session_state
      │   ├─ Model selection (sidebar)
      │   ├─ Training & metrics
      │   ├─ Visualizations
      │   └─ Make predictions
      └─ Tab 3: Classification
          ├─ Check session_state
          ├─ Model selection (sidebar)
          ├─ Training & metrics
          ├─ Visualizations
          └─ Make predictions
```

### To-dos

- [ ] Créer la structure principale avec st.tabs() et supprimer le bouton radio problem_type
- [ ] Implémenter l'onglet Preprocessing & Exploration avec session_state pour partager les données
- [ ] Implémenter l'onglet Régression avec entraînement, validation et prédictions
- [ ] Implémenter l'onglet Classification avec entraînement, validation et prédictions
- [ ] Adapter la sidebar pour qu'elle soit contextuelle selon l'onglet actif