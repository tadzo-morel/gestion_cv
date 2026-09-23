# Dossier CV — Classement automatique de CV par IA

TPE — Morel Tadzo, Licence 3 Informatique, Université de Douala.

Application web qui classe des CV par catégorie de métier (Random Forest +
TF-IDF, entraîné sur un jeu de ~2 500 CV réels) et permet de comparer
plusieurs CV entre eux via un score de qualité.

Le projet a deux parties séparées :

- **`app.py`** : API Flask (le "cerveau" — modèle ML, extraction PDF/DOCX,
  routes `/api/...`).
- **`app_choix_cv/`** : interface React (upload, résultats, comparaison).

Les deux tournent en parallèle sur deux ports différents (5000 et 3000) et
se parlent en HTTP.

## Installation

### 1. Backend (Python)

Depuis la racine du projet :

```bash
python -m venv venv
# Windows :
venv\Scripts\activate
# macOS / Linux :
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Frontend (React)

```bash
cd app_choix_cv
npm install
```

## Lancer l'application

Il faut deux terminaux ouverts en même temps.

**Terminal 1 — backend :**

```bash
python app.py
```

Au premier lancement, le modèle est entraîné à partir de
`archive_cv1/Resume/Resume1.csv` (~1 minute), puis sauvegardé dans
`models/`. Les lancements suivants sont quasi instantanés : le modèle
sauvegardé est simplement rechargé. Pour forcer un ré-entraînement,
supprime le contenu de `models/` ou passe `force_retrain=True` à
`load_and_prepare_dataset()`.

Le serveur écoute sur **http://localhost:5000**.

**Terminal 2 — frontend :**

```bash
cd app_choix_cv
npm start
```

L'interface s'ouvre sur **http://localhost:3000**. En développement, elle
transmet automatiquement les appels `/api/...` vers le backend sur le port
5000 (voir le champ `"proxy"` dans `app_choix_cv/package.json`) : pas de
souci de CORS à gérer.

### Vérifier que tout est en ordre

```bash
python check_setup.py   # vérifie les bibliothèques Python installées
python test_api.py      # teste les endpoints /api/... (backend démarré requis)
```

## Utilisation

- **Onglet Analyser** : dépose un CV (PDF, DOCX ou TXT) ou colle son texte.
  Résultat : catégorie prédite, top 3 des catégories probables, score de
  qualité sur 100, compétences détectées, expérience estimée.
- **Onglet Comparer** : dépose au moins deux CV (ou colle plusieurs textes
  séparés par une ligne `---`). Résultat : tableau classé par score de
  qualité décroissant.

## API

| Méthode | Route               | Description                                    |
| ------- | -------------------- | ----------------------------------------------- |
| GET     | `/api/stats`          | Statistiques du dataset d'entraînement           |
| GET     | `/api/categories`     | Liste des catégories reconnues                   |
| POST    | `/api/analyze`        | Analyse un CV envoyé en texte (`{ "cv_text": "…" }`) |
| POST    | `/api/analyze-file`   | Analyse un CV envoyé en fichier (`file`, form-data) |
| POST    | `/api/compare`        | Compare des CV en texte (`{ "cvs": ["…", "…"] }`) |
| POST    | `/api/compare-files`  | Compare des CV en fichiers (`files`, form-data, plusieurs) |

Formats de fichier acceptés : PDF, DOCX, TXT (10 Mo max par requête). Un PDF
scanné sans couche de texte (image pure) renvoie une erreur explicite plutôt
qu'un résultat vide.

## Configuration

Le chemin du dataset peut être surchargé sans toucher au code, via la
variable d'environnement `CV_DATASET_PATH` :

```bash
CV_DATASET_PATH=/chemin/vers/Resume1.csv python app.py
```

Par défaut, `app.py` utilise `archive_cv1/Resume/Resume1.csv`, relatif au
dossier du script (donc peu importe d'où la commande est lancée).

## Structure du projet

```
app_choix_cv_IA/
├── app.py                    # API Flask + ML
├── requirements.txt
├── check_setup.py            # vérifie les libs Python installées
├── test_api.py                # tests manuels des endpoints
├── choix_cv.ipynb            # notebook d'exploration du dataset
├── models/                   # modèle entraîné, sauvegardé (.pkl)
├── uploads/                  # dossier de travail (vide, non utilisé pour l'instant)
├── archive_cv1/
│   └── Resume/Resume1.csv    # dataset d'entraînement (~2 500 CV)
├── app_choix_cv/             # interface React
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.js
│       ├── api.js            # appels vers l'API Flask
│       └── components/
│           ├── AnalyzePanel.js
│           ├── ComparePanel.js
│           ├── AnalysisResult.js
│           └── FileDropzone.js
├── document_technique_de_cv.docx
├── rapport/RapportProjetGestionCV.docx
└── PowerPoint_de_CV.pptx
```

## Limites connues

- Le score de qualité et la détection de compétences reposent sur des
  règles simples (mots-clés, longueur du texte) plutôt que sur une analyse
  sémantique — c'est un indicateur, pas une vérité absolue.
- Un CV scanné en image (PDF sans couche de texte) n'est pas lisible sans
  OCR, qui n'est pas implémenté ici.
- Le dataset d'entraînement est en anglais ; un CV rédigé dans une autre
  langue sera moins bien classé.
