# 🏠 Prédiction de Prix Immobilier - Californie

## 📌 Description
Application web de prédiction de prix de maisons en Californie utilisant un modèle **XGBoost** optimisé.

Ce projet a été réalisé dans le cadre d'un apprentissage du Machine Learning, de la Data Science et du déploiement d'applications avec Streamlit. Il permet à un utilisateur de saisir les caractéristiques d'un logement et d'obtenir instantanément une estimation de son prix.

---

## 📊 Dataset
- **Source** : California Housing (scikit-learn)
- **Taille** : 20 556 lignes (après nettoyage)
- **Caractéristiques** : 8 features numériques
  - `MedInc` : Revenu médian du district (en dizaines de milliers de $)
  - `HouseAge` : Âge moyen des maisons (années)
  - `AveRooms` : Nombre moyen de pièces par maison
  - `AveBedrms` : Nombre moyen de chambres par maison
  - `Population` : Population du district
  - `AveOccup` : Nombre moyen d'occupants par maison
  - `Latitude` : Latitude du district
  - `Longitude` : Longitude du district
- **Cible** : `MedHouseVal` (prix médian en centaines de milliers de $)

---

## 🏆 Performances du Modèle

| Métrique | Valeur |
|----------|--------|
| **R² Score** | **0.852** |
| RMSE | 45 110 $ |
| MAE | 29 280 $ |

### 📈 Comparaison des Modèles

| Modèle | R² Test | RMSE | MAE |
|--------|---------|------|-----|
| Régression Linéaire | 0.660 | 68 350 $ | 49 380 $ |
| XGBoost (base) | 0.839 | 47 070 $ | 31 460 $ |
| **XGBoost (optimisé)** | **0.852** | **45 110 $** | **29 280 $** |

---

## 🚀 Installation et Utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/MOUAD642207/california-housing-prediction.git
cd california-housing-prediction

```bash
streamlit run app.py

🔄 Améliorations Futures
□ Déploiement sur Streamlit Cloud ou Heroku
□ Ajout de graphiques interactifs (distribution du prix, carte géographique)
□ Historique des prédictions sauvegardé
□ Ajout d'autres modèles (LightGBM, Random Forest)
□ API REST avec FastAPI ou Flask
□ Interface utilisateur multilingue


