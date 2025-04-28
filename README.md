# StatApp - détection de panneaux PV à Madagascar
Détection de panneaux solaires par CNN.

## Table des matières

- [Installation](#installation)
- [Fichiers](#fichiers)
- [Contact](#contact)

## Installation

Clonez le dépôt :

```bash
git clone https://github.com/maximechansat/StatApp.git
```

Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Fichiers

- **`dataset.py`** : Définit la classe `SolarPanelDataset`, qui gère le chargement, le prétraitement et l'augmentation du jeu de données des panneaux solaires. Elle prend en charge les tâches de classification et de segmentation.
- **`training.py`** : Implémente la boucle d'entraînement pour le modèle de segmentation, en utilisant le jeu de données défini dans `dataset.py`. Elle inclut des fonctions pour l'entraînement, l'évaluation et l'enregistrement du meilleur modèle basé sur la performance sur les tests.
- **`hands_on.ipynb`** : Charge un modèle de segmentation pré-entraîné à partir des poids enregistrés (`model_bdappv_seg.pth`) pour réaliser des prédictions sur de nouvelles images sans nécessiter de réentraînement. Ce fichier gère également l'importation des données nécessaires.
- **`StatDescriptives.ipynb`** : Fournit des statistiques descriptives des données. Permet de comprendre la structure et les caractéristiques des données utilisées pour l'entraînement.

## Contact
[mettre nos @ensae.fr ?]
