# AIF-Movies - Partie 1

Cette partie a pour objectif de développer un modèle capable de prédire le genre d’un film à partir de son affiche. Elle comprend plusieurs étapes :

### 1) Développement et entraînement du modèle

Création d’un modèle de computer vision (MovieNet) qui prend en entrée des affiches de films et prédit parmi 10 genres.

Entraînement du modèle sur un dataset d’affiches de films.

### 2) Développement d’une API et d’une interface locale

Création d’une API Flask pour faire des prédictions sur une ou plusieurs images.

Développement d’une interface Gradio pour interagir avec le modèle en local.

### 3) Préparation au déploiement

Création d’une image Docker contenant l’API Flask et l’interface Gradio.

## Image et container Docker

Pour utiliser le modèle MovieNet déjà entraîné, il suffit de lancer la commande `sudo docker compose up` dans son terminal. Cela construit deux images Docker (une pour l'API, une pour l'interface Gradio), puis instancie deux conteneurs à partir de ces images. L'interface MovieNet est ensuite accessible à l'adresse suivante : `http://localhost:7860`

## Scripts Python 

### `MovieNet.py`
- Contient la définition du modèle de prédiction MovieNet. C'est un ResNet50 pré-entraîné sur le dataset ImageNet, auquel il a été ajouté une couche linéaire de classification pour l'adapter à la tâche de prédiction de genre de film

### `Train_MovieNet.py`

- Script d'entraînement du modèle MovieNet
- Si le dataset n'est pas présent en local, il est auotmatiquement téléchargé sur la machine depuis https://drive.google.com/file/d/1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf/view
- À la fin de l’entraînement, les poids sont sauvegardés dans `saved_models/` (ignoré par Git)

### `MovieNet_api.py`

- Fournit une API REST avec Flask pour faire des prédictions
- Si le fichier avec les poids du modèle ne sont pas présents en local, il est automatiquement téléchargé sur la machine depuis https://huggingface.co/qsenatore/MovieNet

### `MovieNet_gradio.py`

- Fournit un interface web Gradio pour utiliser le modèle facilement
