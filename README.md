# AIF-Movies
Projet AI Framework : Intégration d'outils IA pour une plateforme de streaming

## Scripts Python 

### `MovieNet.py`
- Contient la définition du modèle de prédiction MovieNet (Pour l'instant, c'est un CNN)

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
