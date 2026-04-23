import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from flask import Flask, jsonify, request
from PIL import Image
import io
import os
import numpy as np
from huggingface_hub import hf_hub_download

from MovieNet import MovieNet
from Recommendation import MovieRecommender


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

# -------------------------
# Début partie CLASSIFICATION
# -------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='movie_poster_model.pth')
parser.add_argument('--hf_repo', type=str, default='qsenatore/MovieNet')
args = parser.parse_args()

# Téléchargement depuis HuggingFace si les poids sont absents en local
if not os.path.exists(args.model_path):
    print(f"Téléchargement du modèle MovieNet depuis HuggingFace ({args.hf_repo})...")
    args.model_path = hf_hub_download(
        repo_id=args.hf_repo,
        filename="movie_poster_model.pth",
    )
    print(f"Poids téléchargés : {args.model_path}")
else:
    print(f"Poids de MovieNet déjà présents en local : {args.model_path}")

# Load model
model = MovieNet(num_classes=10).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Transformations adaptées aux posters
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# mapping indices -> genres
genres = [
    "action",
    "animation",
    "documentary",
    "comedy",
    "drama",
    "fantasy",
    "horror",
    "romance",
    "science fiction",
    "thriller"
]
# -------------------------
# Fin partie CLASSIFICATION
# -------------------------


# -------------------------
# Début partie RECOMMANDATION
# -------------------------

# Télécharge les features extraites dans Recommendation.py
recommender = MovieRecommender(
    features_path="features.npy",
    df_path="df.pkl"
)

# Poids par défaut de ResNet50
weights = ResNet50_Weights.DEFAULT
resnet = resnet50(weights=weights)

resnet_model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device)
resnet_model.eval()

# -------------------------
# Fin partie RECOMMANDATION
# -------------------------



@app.route('/predict', methods=['POST'])
def predict():

    img_binary = request.data
    img_pil = Image.open(io.BytesIO(img_binary)).convert("RGB")

    tensor = transform(img_pil).to(device)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)

    probs = probabilities[0].cpu().numpy()

    result = {genres[i]: float(probs[i]) for i in range(len(genres))}

    return jsonify({
        "prediction_genre": genres[probs.argmax()],
        "probabilities": result
    })


@app.route('/batch_predict', methods=['POST'])
def batch_predict():

    images_binary = request.files.getlist("images[]")
    tensors = []

    for img_binary in images_binary:
        img_pil = Image.open(img_binary.stream).convert("RGB")
        tensor = transform(img_pil)
        tensors.append(tensor)

    batch_tensor = torch.stack(tensors, dim=0)

    with torch.no_grad():
        outputs = model(batch_tensor.to(device))
        _, predictions = outputs.max(1)

    genres_pred = [genres[p] for p in predictions.tolist()]

    return jsonify({
        "predictions_index": predictions.tolist(),
        "predictions_genre": genres_pred
    })

@app.route('/recommend', methods=['POST'])
def recommend():

    try:
        file = request.files["file"]
        img_pil = Image.open(file.stream).convert("RGB")

        tensor = transform(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet_model(tensor)
            embedding = torch.flatten(embedding, 1)
            embedding = embedding.cpu().numpy()[0]

        paths = recommender.recommend_from_vector(embedding, top_k=5)

        return jsonify({
            "recommendations": paths
        })

    except Exception as e:
        print("ERROR /recommend:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5075, debug=False)