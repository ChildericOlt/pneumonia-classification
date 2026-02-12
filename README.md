# Pneumonia Detection using Convolutional Neural Networks

## Description

Ce projet implémente un modèle de classification binaire pour la détection de pneumonie à partir d'images de radiographies thoraciques (chest X-rays). Le modèle utilise un réseau de neurones convolutifs (CNN) entraîné sur un dataset de radiographies, avec des techniques de régularisation comme le dropout pour éviter l'overfitting.

Le notebook `main.ipynb` contient tout le code nécessaire : chargement des données, définition du modèle, entraînement, évaluation et visualisation des résultats.

## Données

Les données utilisées proviennent du dataset "Chest X-ray Classification" disponible sur Hugging Face (probablement `keremberke/chest-xray-classification` ou similaire). Ce dataset contient des images de radiographies thoraciques étiquetées pour la classification binaire : pneumonie (1) ou normal (0).

### Téléchargement et organisation locale

Les données ont été téléchargées localement en utilisant la bibliothèque `datasets` de Hugging Face. Elles sont organisées selon l'architecture suivante :

```
data/
├── train/
│   └── 0000.parquet  # Données d'entraînement (~5000+ images)
├── val/
│   └── 0000.parquet  # Données de validation (~1000+ images)
└── test/
    └── 0000.parquet  # Données de test (~1000+ images)
```

Chaque fichier Parquet contient :
- `image`: Les bytes de l'image (format PIL Image)
- `labels`: L'étiquette binaire (0 = normal, 1 = pneumonie)

### Prétraitement

- Redimensionnement des images à 224x224 pixels
- Conversion en tenseurs PyTorch
- Normalisation avec les moyennes et écarts-types ImageNet : `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

## Stack Technique

- **Langage** : Python 3.x
- **Framework de Deep Learning** : PyTorch (avec torchvision pour les transformations d'images)
- **Manipulation de données** : Pandas, PyArrow (pour les fichiers Parquet)
- **Traitement d'images** : Pillow (PIL)
- **Métriques et évaluation** : Scikit-learn
- **Visualisation** : Matplotlib
- **Téléchargement de données** : Hugging Face datasets
- **Calcul numérique** : NumPy

## Installation

1. Cloner le repository ou télécharger les fichiers.

2. Créer un environnement virtuel :
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Sur macOS/Linux
   ```

3. Installer les dépendances :
   ```bash
   pip install torch torchvision pandas pyarrow Pillow scikit-learn matplotlib datasets
   ```

4. Télécharger les données (si nécessaire) :
   - Utiliser la bibliothèque `datasets` pour télécharger le dataset Chest X-ray Classification
   - Sauvegarder les splits train/val/test dans le dossier `data/` comme indiqué ci-dessus

## Utilisation

1. Ouvrir le notebook `main.ipynb` dans Jupyter ou VS Code.

2. Exécuter les cellules dans l'ordre :
   - Installation des packages
   - Import des bibliothèques
   - Vérification du device (CPU/GPU)
   - Inspection des données
   - Définition du dataset et des transformations
   - Définition du modèle CNN
   - Entraînement du modèle
   - Évaluation sur le jeu de test

3. Le modèle sauvegardé (`best_model.pth`) peut être rechargé pour l'évaluation sans réentraînement.

## Architecture du Modèle

Le modèle `PneumoniaClassifier` est un CNN simple composé de :

- **Couches convolutives** :
  - Conv2D (3→16 canaux) + ReLU + MaxPool
  - Conv2D (16→32) + ReLU + MaxPool  
  - Conv2D (32→64) + ReLU + MaxPool

- **Couches fully connected** :
  - Linear (64*28*28 → 224) + ReLU + Dropout(0.5)
  - Linear (224 → 64) + ReLU + Dropout(0.5)
  - Linear (64 → 1) + Sigmoid

- **Régularisation** : Dropout à 0.5 appliqué après les deux premières couches fully connected pour éviter l'overfitting.

## Entraînement

- **Optimiseur** : Adam (lr=0.001)
- **Fonction de perte** : Binary Cross Entropy (BCELoss)
- **Batch size** : 32
- **Epochs** : 5
- **Sauvegarde** : Le modèle avec la meilleure loss de validation est sauvegardé automatiquement

## Résultats

Après ajout du dropout, les résultats d'entraînement montrent une réduction significative de l'overfitting :

- **Avant dropout** (overfitting) :
  - Epoch 5: Train Loss: 0.0477, Val Loss: 0.1851

- **Après dropout** :
  - Epoch 5: Train Loss: 0.1014, Val Loss: 0.1943 (mais meilleur modèle sauvegardé à Epoch 4 avec Val Loss: 0.1545)

### Métriques sur le jeu de test (avec le meilleur modèle)

- Précision globale (Accuracy)
- Capacité à détecter les malades (Recall) 
- Score de séparation (AUC ROC)
- Matrice de confusion
- Courbe ROC

## Améliorations Possibles

- **Data Augmentation** : Ajouter des transformations aléatoires (rotations, flips, crops) pour augmenter la diversité des données
- **Architecture plus complexe** : Utiliser des modèles pré-entraînés comme ResNet ou EfficientNet
- **Hyperparameter tuning** : Ajuster le taux de dropout, le learning rate, etc.
- **Early stopping** : Arrêter l'entraînement si la validation ne s'améliore plus
- **Cross-validation** : Pour une évaluation plus robuste

## Structure du Projet

```
pneumonia_detection/
├── main.ipynb          # Notebook principal avec tout le code
├── main.py            # Version script Python (optionnel)
├── best_model.pth     # Modèle entraîné sauvegardé
├── data/              # Données (non incluses dans le repo)
│   ├── train/
│   ├── val/
│   └── test/
└── README.md          # Ce fichier
```