# 🚀 Démarrage Rapide - Déploiement pour 150 Élèves

## Préparation (À faire une fois)

### 1. Vérifier les Optimisations

```bash
cd /home/lopilo/code/aib-ml-assignments
python utils/performance_test.py
```

Vous devriez voir :
```
✅ Model training speedup: ...
✅ Visualization speedup: ~6x
✅ Estimated capacity: 80-150 concurrent users
```

### 2. Tester en Local

```bash
streamlit run app.py
```

Ouvrez http://localhost:8501 et vérifiez que tout fonctionne.

### 3. Trouver Votre IP Local

**Sur Linux/Mac :**
```bash
hostname -I | awk '{print $1}'
```

**Sur Windows :**
```bash
ipconfig
```

Notez l'adresse IPv4 (ex: 192.168.1.100)

## Jour du Cours

### Option 1 : Session Unique (150 élèves)

**1. Démarrer l'application**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.maxMessageSize=200
```

**2. Donner l'URL aux élèves**
```
http://[VOTRE_IP]:8501
```

**3. Organiser les activités**
- Groupe A : Commence par "Preprocessing & Exploration"
- Groupe B : Commence par "Classification" avec modèle léger
- Groupe C : Commence par visualiser les données
- Après 15 min : Rotation

Cela évite que tous entraînent des modèles en même temps.

### Option 2 : Sessions Échelonnées (RECOMMANDÉ)

**Session 1 : 14h00-14h45**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.maxMessageSize=200
```
50 premiers élèves

**Session 2 : 14h45-15h30**
50 élèves suivants (redémarrer l'app pour rafraîchir la mémoire)

**Session 3 : 15h30-16h15**
50 derniers élèves

## Monitoring Pendant le Cours

### Surveiller la Charge CPU

**Terminal 2 (pendant que Streamlit tourne) :**
```bash
htop  # ou top
```

Vérifiez que :
- CPU ne dépasse pas 85-90% en moyenne
- Quelques cores restent libres (grâce aux 3 réservés)

### Surveiller la RAM

```bash
free -h
```

Avec 32GB, vous devriez rester sous 50% d'utilisation.

### Surveiller les Connexions

Dans les logs Streamlit, vous verrez :
```
New connection from 192.168.1.X
```

## En Cas de Problème

### L'application est très lente

**Solution rapide :**
1. Demander aux élèves de rafraîchir leur page (F5)
2. Demander de fermer les onglets inutilisés
3. Réduire le nombre d'élèves actifs (faire une pause pour un groupe)

### L'application ne répond plus

**Redémarrage rapide :**
1. Ctrl+C dans le terminal
2. Attendre 5 secondes
3. Relancer : `streamlit run app.py --server.address=0.0.0.0 --server.maxMessageSize=200`

Les élèves devront juste rafraîchir leur navigateur.

### Les élèves ne peuvent pas se connecter

**Vérifications :**
1. Firewall désactivé ou port 8501 ouvert
2. Tous sur le même réseau WiFi/Ethernet
3. IP correcte (pas 127.0.0.1 mais 192.168.x.x)

**Ouvrir le port sur Linux :**
```bash
sudo ufw allow 8501
```

## Consignes pour les Élèves

**À distribuer en début de cours :**

```
🎓 Application ML Titanic

URL : http://[VOTRE_IP]:8501

Consignes :
1. Un seul onglet ouvert par personne
2. Si l'application est lente, patientez 5-10 secondes
3. Ne pas spammer les boutons (cliquer une fois puis attendre)
4. Fermer l'onglet en fin de TP

En cas de problème : F5 (rafraîchir)
```

## Checklist Pré-Cours

- [ ] Application testée en local
- [ ] IP local notée
- [ ] Port 8501 ouvert dans le firewall
- [ ] Ordinateur branché secteur (pas sur batterie)
- [ ] Connexion Internet stable (pour charger dataset)
- [ ] Terminal prêt avec la commande
- [ ] `htop` installé pour monitoring

## Commandes Utiles

**Lancement standard :**
```bash
streamlit run app.py
```

**Lancement réseau local :**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.maxMessageSize=200
```

**Lancement avec headless (pas de navigateur auto) :**
```bash
streamlit run app.py --server.address=0.0.0.0 --server.headless=true --server.maxMessageSize=200
```

**Trouver l'IP :**
```bash
hostname -I  # Linux
ip addr      # Linux détaillé
ipconfig     # Windows
```

**Tuer Streamlit si bloqué :**
```bash
pkill -f streamlit
```

## Performance Attendue

Avec votre configuration (i7-12700KF, 32GB) :

| Élèves Actifs | Temps Réponse | Qualité |
|---------------|---------------|---------|
| 0-30 | <1s | Excellent |
| 30-60 | 1-2s | Très bon |
| 60-100 | 2-5s | Bon |
| 100-150 | 5-10s | Acceptable |

**Note :** Les temps sont pour l'entraînement de modèles. La navigation est toujours fluide.

## Après le Cours

**Nettoyage (optionnel) :**
```bash
# Vider le cache Streamlit
streamlit cache clear

# Redémarrer proprement
pkill -f streamlit
```

## Contact Support

En cas de problème persistant :
1. Consulter `RESUME_OPTIMISATIONS.md`
2. Lire `OPTIMIZATIONS.md` (détails techniques)
3. Exécuter `python utils/performance_test.py`

---

**Bon cours ! 🎓🚀**

