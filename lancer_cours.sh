#!/bin/bash

# Script de lancement optimisé pour cours avec 100-150 élèves
# Usage: ./lancer_cours.sh

# Cleanup function to properly stop both processes
cleanup() {
    echo ""
    echo "🛑 Arrêt des services en cours..."
    
    if [ ! -z "$CLOUDFLARED_PID" ]; then
        kill $CLOUDFLARED_PID 2>/dev/null
        echo "   ✓ Cloudflared tunnel arrêté"
    fi
    
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo "   ✓ Streamlit arrêté"
    fi
    
    # Clean up temporary log file
    rm -f /tmp/cloudflared_output.log 2>/dev/null
    
    echo "✅ Application arrêtée proprement"
    exit 0
}

# Trap SIGINT (Ctrl+C) to cleanup properly
trap cleanup SIGINT

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
echo "   - Cloudflared tunnel : Accès distant sécurisé"

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

# Start cloudflared tunnel in background
echo "🌐 Démarrage du tunnel Cloudflared..."
cloudflared tunnel --config ~/.cloudflared/config.yml run aib-ml-assignment > /tmp/cloudflared_output.log 2>&1 &
CLOUDFLARED_PID=$!
echo "   ✓ Tunnel démarré (PID: $CLOUDFLARED_PID)"

# Wait a moment for cloudflared to establish the tunnel
sleep 2

# Try to extract and display the cloudflared URL
if [ -f /tmp/cloudflared_output.log ]; then
    TUNNEL_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' /tmp/cloudflared_output.log | head -1)
    if [ ! -z "$TUNNEL_URL" ]; then
        echo "   🔗 URL du tunnel : $TUNNEL_URL"
    fi
fi

echo ""
echo "🚀 Démarrage de Streamlit..."
# Start Streamlit in background (production mode)
streamlit run app.py \
    --server.address=0.0.0.0 \
    --server.maxMessageSize=200 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=true \
    --global.developmentMode=false &
STREAMLIT_PID=$!
echo "   ✓ Streamlit démarré en mode production (PID: $STREAMLIT_PID)"

echo ""
echo "✅ Services actifs - Appuyez Ctrl+C pour arrêter"
echo ""

# Wait for both processes
wait

