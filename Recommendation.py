import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

"""
Nécessite les fichiers features.npy & df.pkl calculés dans Recommendation_test.py
Fichier directement appeler dans l'API afin d'importer le modèle de prédiction
"""

class MovieRecommender:
    """
    Système de recommandation basé sur embeddings CNN pré-calculés.
    """

    def __init__(self, features_path="features.npy", df_path="df.pkl"):
        """
        Charge les features et les métadonnées.
        """

        self.features = np.load(features_path)
        self.df = pd.read_pickle(df_path)
        self.features = self._l2_normalize(self.features) # Normalisation

        #print(f"[INFO] Recommender chargé : {len(self.df)} films")

    # -----------------------------------------------------
    # UTILITAIRES
    # -----------------------------------------------------

    def _l2_normalize(self, x, eps=1e-8):
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

    # -----------------------------------------------------
    # RECOMMANDATION PAR INDEX
    # -----------------------------------------------------

    def recommend_from_index(self, idx, top_k=5):
        """
        Retourne les k films les plus similaires à une affiche donnée.
        Ne sera pas utilisé dans l'API.
        """

        query_vec = self.features[idx].reshape(1, -1)
        sims = cosine_similarity(query_vec, self.features)[0]
        top_idx = sims.argsort()[::-1][1:top_k + 1]
        return self.df.iloc[top_idx]["path"].tolist()

    # -----------------------------------------------------
    # RECOMMANDATION PAR VECTOR (API FUTURE)
    # -----------------------------------------------------

    def recommend_from_vector(self, vector, top_k=5):
        """
        Permet de faire de la reco depuis une image encodée (embedding CNN).
        """

        vector = vector.reshape(1, -1)
        vector = self._l2_normalize(vector)
        sims = cosine_similarity(vector, self.features)[0]
        top_idx = sims.argsort()[::-1][1:top_k+1]
        return self.df.iloc[top_idx]["path"].tolist()


    def get_path(self, idx):
        return self.df.iloc[idx]["path"]

    def get_features(self, idx):
        return self.features[idx]