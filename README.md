# Artificial Intelligence Frameworks  
### Projet INSA 5ème année 🎬 

### Objectif
Intégration d’outils d’intelligence artificielle dans une plateforme de streaming de films.

## Lancer l'application avec des images et containers Docker

Pour utiliser les différentes fonctionnalités développées dans ce projet, il suffit de lancer la commande `sudo docker compose up` dans son terminal. Cela construit deux images Docker (une pour l'API, une pour l'interface Gradio), puis instancie deux conteneurs à partir de ces images. L'interface utilisateur est ensuite accessible à l'adresse suivante : `http://localhost:7860`

## Partie 1 — Classification de films à partir de leur poster

<details>
<summary><strong>Voir les détails</strong></summary>

Cette partie a pour objectif de développer un modèle capable de prédire le genre d’un film à partir de son affiche. C'est un ResNet50 pré-entraîné sur le dataset ImageNet, auquel il a été ajouté une couche linéaire de classification pour l'adapter à la tâche de prédiction de genre de film.

</details>

## Partie 2 — Recommandation de films à partir de leur poster

<details>
<summary><strong>Voir les détails</strong></summary>

Cette partie a pour objectif de développer un modèle capable de recommander d'autres films à partir de l'affiche d'un film.

</details>

## Partie 3 — Classification et Recommandation de films par leur synopsis

<details>
<summary><strong>Voir les détails</strong></summary>

Cette partie a pour objectif de développer des modèles capables de prédire le genre d'un film et recommander d'autres films à partir du synopsis de ce film. Ce sont un modèle Bag-of-Words, un modèle LSTM et un modèle basé sur DistilBERT qui sont utilisés. 

Le modèle Bag-of-Words repose sur une moyenne des embeddings de mots suivie d’une couche linéaire de classification. 
Le modèle LSTM exploite une architecture récurrente afin de capturer les dépendances séquentielles du texte
Le modèle BERT utilise des représentations contextuelles pré-entraînées issues d’un transformer partiellement fine-tuné pour la tâche de classification. 

Dans les trois cas, les représentations textuelles apprises sont également utilisées pour construire un espace vectoriel des films, permettant de générer des recommandations via un index de similarité Annoy basé sur la distance angulaire entre embeddings.

</details>

## Partie 4 — 

<details>
<summary><strong>Voir les détails</strong></summary>



</details>

### Scripts Python utilisés dans les 4 parties

### `Movie_api.py`

- Fournit une API REST avec Flask

### `Movie_gradio.py`

- Fournit un interface web Gradio pour utiliser les différents modèles facilement

Dans le fichier `Codes d'entraînement` se trouvent les codes pour l'entraînement des différents modèles.
