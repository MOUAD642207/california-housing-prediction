import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Prix Immobilier",
    page_icon="🏠",
    layout="wide"
)

# Titre
st.title("🏠 Prédiction du Prix d'une Maison en Californie")
st.markdown("---")

# Chargement du modèle et du scaler
@st.cache_resource
def load_model():
    model = joblib.load('housing_price_model.pkl')
    scaler = joblib.load('scaler.pkl')
    with open('feature_names.json', 'r') as f:
        features = json.load(f)
    with open('model_metrics.json', 'r') as f:
        metrics = json.load(f)
    return model, scaler, features, metrics

model, scaler, features, metrics = load_model()

# Affichage des métriques
col1, col2, col3, col4 = st.columns(4)
col1.metric("R² Score", f"{metrics['r2']:.3f}")
col2.metric("RMSE", f"{metrics['rmse']:.3f} (x100k $)")
col3.metric("MAE", f"{metrics['mae']:.3f} (x100k $)")
col4.metric("Meilleur Modèle", "XGBoost")

st.markdown("---")

# Interface utilisateur
st.subheader("📊 Caractéristiques du logement")

col_left, col_right = st.columns(2)

with col_left:
    med_inc = st.number_input(
        "Revenu Médian (en dizaines de milliers de $)",
        min_value=0.5, max_value=15.0, value=3.0, step=0.1,
        help="Revenu médian du district"
    )
    
    house_age = st.number_input(
        "Âge Moyen des Maisons (années)",
        min_value=1, max_value=52, value=20, step=1
    )
    
    avg_rooms = st.number_input(
        "Nombre Moyen de Pièces",
        min_value=1.0, max_value=20.0, value=5.0, step=0.1,
        help="Nombre moyen de pièces par maison"
    )
    
    avg_bedrooms = st.number_input(
        "Nombre Moyen de Chambres",
        min_value=0.5, max_value=5.0, value=1.0, step=0.05
    )

with col_right:
    population = st.number_input(
        "Population du District",
        min_value=100, max_value=20000, value=1000, step=100
    )
    
    avg_occup = st.number_input(
        "Nombre Moyen d'Occupants par Maison",
        min_value=0.5, max_value=20.0, value=3.0, step=0.1
    )
    
    latitude = st.number_input(
        "Latitude",
        min_value=32.5, max_value=42.0, value=35.0, step=0.01
    )
    
    longitude = st.number_input(
        "Longitude",
        min_value=-124.5, max_value=-114.0, value=-120.0, step=0.01
    )

# Bouton de prédiction
st.markdown("---")
if st.button("🔮 Prédire le Prix", type="primary"):
    # Créer le DataFrame
    input_data = pd.DataFrame([[
        med_inc, house_age, avg_rooms, avg_bedrooms,
        population, avg_occup, latitude, longitude
    ]], columns=features)
    
    # Normaliser
    input_scaled = scaler.transform(input_data)
    
    # Prédire
    prediction = model.predict(input_scaled)[0]
    
    # Prix en dollars
    price_dollars = prediction * 100000
    
    # Afficher le résultat
    st.markdown("---")
    st.subheader("💰 Résultat de la Prédiction")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: #d4edda;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
            ">
                <h2 style="color: #155724; margin: 0;">
                    ${price_dollars:,.0f}
                </h2>
                <p style="color: #155724; margin: 5px 0 0 0;">
                    Prix Médian Prédit (en dollars)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Affichage de l'intervalle de confiance
    st.markdown("---")
    st.caption(f"📌 Modèle entraîné sur des données de Californie • R² = {metrics['r2']:.3f}")

# Pied de page
st.markdown("---")
st.caption("🔧 Modèle XGBoost • Données California Housing • 2026")