
# Projet SDD1004 – Détection de Voitures avec YOLOv8 (Streamlit)

## 🎓 Informations Générales
- **Auteur :** cheikhouna kebe
- **Cours :** SDD1004 – Sciences des Données et Applications
- **Université :** Université du Québec à Trois-Rivières (UQTR)

## 🚀 Description
Cette application Streamlit permet de détecter automatiquement les **voitures** dans une image grâce au modèle **YOLOv8** préentraîné. L’interface est simple et efficace :

- Téléversement d’image (formats supportés : `.jpg`, `.jpeg`, `.png`)
- Affichage du résultat annoté
- Affichage des coordonnées de détection
- Téléchargement de l’image annotée
- Graphe camembert avec la répartition des classes détectées

## 📁 Contenu de l'archive

```
YOLO8_VOITURE_STREAMLIT/
├── app.py               # Code principal de l'application
├── requirements.txt     # Dépendances à installer
├── README.md            # (Ce fichier)
└── images/              # Dossier contenant les images de test
```

## ▶️ Instructions d'exécution

1. Crée un environnement virtuel :
   ```
   python -m venv env
   ```

2. Active-le :
   - Windows :
     ```
     env\Scripts\activate
     ```
   - macOS/Linux :
     ```
     source env/bin/activate
     ```

3. Installe les dépendances :
   ```
   pip install -r requirements.txt
   ```

4. Lance l'application :
   ```
   streamlit run app.py
   ```

## 📌 Remarques importantes
- Le modèle doit être placé localement dans le même dossier que `app.py` avant exécution.
- Le dossier `images/` contient des exemples d'images pour faciliter les tests.

## ✅ Fonctionnalités couvertes
- ✅ Téléversement d'image
- ✅ Détection avec YOLOv8
- ✅ Visualisation et téléchargement de l’image annotée
- ✅ Tableau des détections
- ✅ Graphe de répartition des classes détectées

