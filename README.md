# 🇹🇳 Système de Recommandation Hybride — Tunisie

> **Projet de Cours — Intelligence Artificielle & Systèmes de Recommandation**

Application interactive de recommandation d'expériences touristiques en Tunisie, combinant 4 algorithmes de filtrage dans une interface Streamlit moderne, avec une base de données PostgreSQL et une carte interactive Folium.

---

## 📁 Structure du Projet

```
sysrectestvs/
│
├── app.py                              # Application Streamlit principale
│
├── data/
│   ├── produits.csv                    # Catalogue des 20 expériences touristiques
│   ├── users.csv                       # 20 utilisateurs fictifs
│   ├── notes.csv                       # Historique des interactions (158 notes)
│   └── experiences.csv                 # Données additionnelles (référence)
│
├── scripts/
│   ├── db_setup.py                     # Initialisation + peuplement de la BDD
│   └── generate_dataset.py             # Génération du jeu de données synthétique
│
├── notebooks/
│   └── Livrable_1_Notebook_Recommandation.ipynb  # Notebook d'analyse (Livrable 1)
│
├── assets/
│   └── logo-nav.svg                    # Logo de l'application
│
├── requirements.txt                    # Dépendances Python
├── render.yaml                         # Configuration déploiement Render.com
└── README.md                           # Ce fichier
```

---

## 🧠 Architecture des Algorithmes

### 1. Filtrage Basé sur le Contenu (Content-Based) — 40%
- Analyse les descriptions textuelles des produits par **tokenisation** et **stemming français** (NLTK)
- Construit une matrice TF-Boolean et calcule la **similarité cosinus**
- Recommande les produits similaires à ceux déjà appréciés par l'utilisateur

### 2. Filtrage Collaboratif (K-NN) — 40%
- Construit la **matrice User × Produit** des notes
- Identifie les **3 utilisateurs les plus similaires** via la similarité cosinus (sklearn)
- Prédit le score d'un produit non encore noté par pondération des voisins

### 3. Filtrage Temporel (Time-Aware) — 20%
- Calcule une **popularité pondérée par la récence** : `score = e^(-Δt/365) × note`
- Applique un **bonus saisonnier** (+50%) pour les activités correspondant à la saison courante
- Favorise les tendances récentes plutôt que les classiques

### 4. Hybridation Pondérée
```
Score Final = CB × 0.4 + CF × 0.4 + TA × 0.2
```

---

## 📊 Métriques d'Évaluation

| Métrique     | Description                           | Valeur Cible |
|--------------|---------------------------------------|--------------|
| Précision @6 | Pertinence des 6 premières reco.      | > 80%        |
| Rappel @6    | Couverture des items pertinents       | > 72%        |
| F1-Score     | Harmonie Précision/Rappel             | > 0.76       |
| NDCG @6      | Qualité du classement des reco.       | > 0.83       |

---

## 🗄️ Base de Données (PostgreSQL)

**Hébergée sur Render.com** | Schéma relationnel :

```
Produit (ID, NomPdt, Description, Categorie, Effort, Saison, Tags)
    ↕ FK IDProduit
Notes (IDNote, IDUser, IDProduit, Note, Timestamp)
    ↕ FK IDUser
Users (IDUser, NomUser)
```

- **20 produits** / **20 utilisateurs** / **158 interactions**
- Données synthétiques générées avec `scripts/generate_dataset.py`

---

## 🗺️ Carte Interactive (Folium)

La carte interactive affiche les recommandations géolocalisées sur la Tunisie :
- 🔵 **Bleu** → recommandé par l'algorithme Contenu
- 🟣 **Violet** → recommandé par le Filtrage Collaboratif
- 🟠 **Orange** → tendance Temporelle/Saisonnière

---

## 🚀 Installation & Lancement

### Prérequis
```
Python >= 3.10
```

### 1. Cloner et installer les dépendances
```bash
pip install -r requirements.txt
```

### 2. (Optionnel) Initialiser la base de données
```bash
python scripts/db_setup.py
```

### 3. Lancer l'application
```bash
streamlit run app.py
```

L'app sera disponible sur : **http://localhost:8501**

---

## ☁️ Déploiement

L'application est déployée sur **Render.com** via `render.yaml`.
La base de données PostgreSQL est hébergée sur l'instance Render gratuite.

---

## 📦 Dépendances Principales

| Bibliothèque       | Usage                                 |
|--------------------|---------------------------------------|
| `streamlit`        | Interface web interactive             |
| `pandas` / `numpy` | Manipulation des données              |
| `scikit-learn`     | Similarité cosinus (Collaboratif)     |
| `nltk`             | Tokenisation & Stemming (Content-CB)  |
| `plotly`           | Graphiques interactifs                |
| `folium`           | Carte géographique interactive        |
| `streamlit-folium` | Intégration Folium dans Streamlit     |
| `psycopg2-binary`  | Connexion PostgreSQL                  |

---

## 👨‍🎓 Informations Académiques

- **Cours** : Systèmes de Recommandation
- **Livrables** :
  - ✅ Livrable 1 : Notebook d'Analyse (`notebooks/`)
  - ✅ Livrable 2 : Application Web (`app.py`)
  - ✅ Livrable 3 : Rapport (`Rapport_Projet_Recommandation.md`)
