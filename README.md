# StatApp - détection de panneaux PV à Madagascar
Détection de panneaux solaires par CNN.

## Fichiers

- **`dataset.py`** : Définit la classe `SolarPanelDataset`, qui gère le chargement, le prétraitement et l'augmentation du jeu de données des panneaux solaires. Elle prend en charge les tâches de classification et de segmentation.
- **`training.py`** : Implémente la boucle d'entraînement pour le modèle de segmentation, en utilisant le jeu de données défini dans `dataset.py`. Elle inclut des fonctions pour l'entraînement, l'évaluation et l'enregistrement du meilleur modèle basé sur la performance sur les tests.
