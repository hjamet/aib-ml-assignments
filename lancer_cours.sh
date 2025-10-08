#!/bin/bash

# Script de lancement optimisé pour cours avec 100-150 élèves
# Usage: ./lancer_cours.sh

echo "🚀 Lancement de l'application ML Streamlit pour cours"
echo "=================================================="
echo ""

# Afficher l'IP locale
echo "📡 Votre IP locale :"
if command -v hostname &> /dev/null; then
    IP=$(hostname -I | awk '{print $1}')
    echo "   $IP"
else
    echo "   Utilisez 'ipconfig' (Windows) ou 'ip addr' (Linux) pour trouver votre IP"
fi

echo ""
echo "🌐 Les élèves pourront se connecter sur :"
if [ ! -z "$IP" ]; then
    echo "   http://$IP:8501"
else
    echo "   http://[VOTRE_IP]:8501"
fi

echo ""
echo "⚙️  Configuration optimisée :"
echo "   - Multiprocessing : 4 cores max par modèle"
echo "   - Cores réservés : 3 pour le système"
echo "   - Visualisations : Résolution optimisée"
echo "   - Capacité : 80-150 utilisateurs simultanés"

echo ""
echo "📊 Pour surveiller les performances, ouvrir un autre terminal et lancer :"
echo "   htop   # ou 'top' si htop n'est pas installé"

echo ""
echo "⏱️  Démarrage dans 3 secondes..."
sleep 3

echo ""
echo "🎯 Application en cours de démarrage..."
echo "   (Appuyez Ctrl+C pour arrêter)"
echo ""

# Lancer Streamlit avec configuration optimisée
streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.maxMessageSize=200 \
    --server.headless=true

echo ""
echo "✅ Application arrêtée proprement"

