# Artificial Intelligence Frameworks  
### Projet INSA 5ème année 🎬 

### Objectif
Intégration d’outils d’intelligence artificielle dans une plateforme de streaming de films.

## Lancer l'application avec des images et containers Docker

Pour utiliser les différentes fonctionnalités développées dans ce projet, il suffit de lancer la commande `sudo docker compose up` dans son terminal. Cela construit deux images Docker (une pour l'API, une pour l'interface Gradio), puis instancie deux conteneurs à partir de ces images. L'interface utilisateur est ensuite accessible à l'adresse suivante : `http://localhost:7860`

## Partie 1 — Classification de films à partir de leur poster

<details>
<summary><strong>Voir les détails</strong></summary>

Cette partie a pour objectif de développer un modèle capable de prédire le genre d’un film à partir de son affiche. Elle comprend plusieurs étapes :

### 1) Développement et entraînement du modèle

Création d’un modèle de computer vision (MovieNet) qui prend en entrée des affiches de films et prédit parmi 10 genres.

Entraînement du modèle sur un dataset d’affiches de films.

### 2) Développement d’une API et d’une interface locale

Création d’une API Flask pour faire des prédictions sur une ou plusieurs images.

Développement d’une interface Gradio pour interagir avec le modèle en local.

### 3) Préparation au déploiement

Création d’une image Docker contenant l’API Flask et l’interface Gradio.

## Scripts Python 

### `MovieNet.py`
- Contient la définition du modèle de prédiction MovieNet. C'est un ResNet50 pré-entraîné sur le dataset ImageNet, auquel il a été ajouté une couche linéaire de classification pour l'adapter à la tâche de prédiction de genre de film

### `Train_MovieNet.py` (Script bonus)

- Script d'entraînement du modèle MovieNet. N'est utile que si vous voulez réentraîner le modèle vous même.
- Si le dataset n'est pas présent en local, il est auotmatiquement téléchargé sur la machine depuis https://drive.google.com/file/d/1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf/view
- Sauvegarde les poids dans le fichier `movie_poster_model.pth`

</details>

## Partie 2 — Recommandation de films à partir de leur poster

<details>
<summary><strong>Voir les détails</strong><</summary>

Cette partie a pour objectif de développer un modèle capable de recommander d'autres films à partir de l'affiche d'un film.

</details>

## Partie 3 —

<details>
<summary><strong>Voir les détails</strong><</summary>



</details>

## Partie 4 — 

<details>
<summary><strong>Voir les détails</strong><</summary>



</details>

## Partie 5 — 

<details>
<summary><strong>Voir les détails</strong><</summary>



</details>

### Scripts Python utilisés dans les 5 parties

### `Movie_api.py`

- Fournit une API REST avec Flask
- Pour la prédiction de genre de films, si le fichier avec les poids du modèle n'est pas présent en local, il est automatiquement téléchargé sur la machine depuis https://huggingface.co/qsenatore/MovieNet

### `Movie_gradio.py`

- Fournit un interface web Gradio pour utiliser les différents modèles facilement
