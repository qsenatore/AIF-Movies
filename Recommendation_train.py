import torch
import pandas as pd
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.image as mpimg
import os
import zipfile
import gdown

"""
Fichier servant à construire et vérifier le modèle utiliser par la suite dans l'API.
Il n'est pas appeler par l'API mais permet de calculer les fichiers features.npy & df.pkl
"""


# Chemin local du dataset
DATASET_PATH = './content/sorted_movie_posters_paligema'
GDRIVE_FILE_ID = '1-1OSGlN2EOqyZuehBgpgI8FNOtK-caYf'
ZIP_PATH = './movie_posters.zip'


# Fonction pour télécharger le dataset
def download_dataset():
    """Télécharge et dézippe le dataset depuis Google Drive si absent."""
    if os.path.exists(DATASET_PATH):
        print(f"Dataset déjà présent : {DATASET_PATH}")
        return

    print("Dataset absent. Téléchargement depuis Google Drive...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    gdown.download(url, ZIP_PATH, quiet=False)

    print("Décompression du dataset...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('.')
    os.remove(ZIP_PATH)
    print(f"Dataset prêt dans : {DATASET_PATH}")


class ImageAndPathsDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, _ = self.samples[index]  # plus stable que self.imgs
        img, _ = super().__getitem__(index)
        return img, path
    

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean, std)

inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(mean, std)],
    std=[1/s for s in std]
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

def plot_image(path):
  img = mpimg.imread(path)
  plt.imshow(img)
  plt.axis('off')

def plot_images(paths_list):
    plt.figure(figsize=(20, 5))
    n = len(paths_list)

    for i, path in enumerate(paths_list):
        plt.subplot(1, n, i+1)
        img = mpimg.imread(path)
        plt.imshow(img)
        plt.axis('off')

    plt.show()

def plot_reco(idx, sim_matrix, df):
    plot_image(df['path'][idx])
    recos = sim_matrix[idx].argsort()[::-1][1:6]
    reco_posters = df.iloc[recos]['path'].tolist()
    plot_images(reco_posters)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    download_dataset()

    dataset = ImageAndPathsDataset(DATASET_PATH, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    weights = ResNet50_Weights.DEFAULT
    resnet = resnet50(weights=weights)

    model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)    
    model = model.eval()

    if os.path.exists("features.npy") and os.path.exists("df.pkl"):
        print("Chargement des features existantes...")
        features = np.load("features.npy")
        df = pd.read_pickle("df.pkl")
    else:
        features_list = []
        paths_list = []

        with torch.no_grad():
            for x, paths in tqdm(dataloader):
                embeddings = model(x.to(device))
                features_list.append(embeddings.cpu().numpy())
                paths_list.extend(paths)

        features = np.vstack(features_list)
        features = features.reshape(features.shape[0], -1)
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        df = pd.DataFrame({
            "features": list(features),
            "path": paths_list
        })

        np.save("features.npy", features)
        pd.to_pickle(df, "df.pkl")
            
    cosine_sim = cosine_similarity(features, features)

    idx = 19
    plot_reco(idx, cosine_sim, df)




