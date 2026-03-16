# AIF-Movies - Partie 1

Cette partie a pour objectif de développer un modèle capable de prédire le genre d’un film à partir de son affiche. Il comprend plusieurs étapes :

### 1) Développement et entraînement du modèle

Création d’un modèle de computer vision (MovieNet) qui prend en entrée les affiches de films et prédit parmi 10 genres.

Entraînement du modèle avec PyTorch sur un dataset d’affiches de films.

### 2) Développement d’une API et d’une interface locale

Création d’une API Flask pour faire des prédictions sur une ou plusieurs images.

Développement d’une interface Gradio pour interagir avec le modèle en local.

### 3) Préparation au déploiement

Création d’une image Docker contenant l’API Flask et l’interface Gradio.

## Scripts Python 

### `MovieNet.py`
- Contient la définition du modèle de prédiction MovieNet. C'est ResNet50 pré-entraîné sur le dataset ImageNet, auquel il a été ajouté une couche linéaire de classification pour l'adapter à la tâche de prédiction de genre de film.

### `Train_MovieNet.py`

- Script d'entraînement du modèle MovieNet
- Vu que le dataset n'est pas sur le répertoire, il faut modifier l'emplacement du dataset à la ligne 62 pour rediriger vers le chemin du dataset téléchargé en local
- À la fin de l’entraînement, les poids sont sauvegardés dans `saved_models/` (ignoré par Git)

Éxécution : `python Train_MovieNet.py --epochs 10 --batch_size 32 --lr 0.001`

### `MovieNet_api.py`

- Fournit une API REST avec Flask pour faire des prédictions
- L’API est disponible sur `http://127.0.0.1:5075`

Éxecution : `python MovieNet_api.py --model_path saved_models/movie_poster_model.pth`

### `MovieNet_gradio.py`

- Fournit un interface web Gradio pour tester le modèle facilement
- L'interface est accessible sur `http://127.0.0.1:7860`

Éxécution : `python app_gradio.py`
