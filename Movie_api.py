import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
import io
import os
import re
import string
import pickle
import json
import numpy as np

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from annoy import AnnoyIndex
from MovieNet import MovieNet

# ─── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

HF_REPO = "qsenatore/MovieNet"

# ─── Chemin (local ou HuggingFace) ────────────────────────────────────────────

def get_model(arg_path, filename):
    # Retourne le chemin local, télécharge depuis HF si absent
    if not os.path.exists(arg_path):
        print(f"Téléchargement {filename} depuis HuggingFace...")
        return hf_hub_download(repo_id=HF_REPO, filename=filename)
    print(f"{filename} trouvé en local : {arg_path}")
    return arg_path

# ─── Args ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument('--model_path',         type=str, default='saved_models/movie_poster_model.pth')
parser.add_argument('--bow_path',           type=str, default='saved_models/model_weights_bag_of_words.pth')
parser.add_argument('--lstm_path',          type=str, default='saved_models/model_weights_lstm.pth')
parser.add_argument('--bert_path',          type=str, default='saved_models/bert_model_weights_1.pth')
parser.add_argument('--vocab_path',         type=str, default='saved_models/vocab.pkl')
parser.add_argument('--label_encoder_path', type=str, default='saved_models/label_encoder.pkl')
parser.add_argument('--bert_annoy',         type=str, default='saved_models/plot_embeddings_bert_1.ann')
parser.add_argument('--metadata',           type=str, default='saved_models/annoy_metadata_bert_1.json')
args = parser.parse_args()

# Résolution : local si présent, sinon téléchargement HuggingFace
args.model_path         = get_model(args.model_path,         "movie_poster_model.pth")
args.bow_path           = get_model(args.bow_path,           "model_weights_bag_of_words.pth")
args.lstm_path          = get_model(args.lstm_path,          "model_weights_lstm.pth")
args.bert_path          = get_model(args.bert_path,          "bert_model_weights_1.pth")
args.vocab_path         = get_model(args.vocab_path,         "vocab.pkl")
args.label_encoder_path = get_model(args.label_encoder_path, "label_encoder.pkl")
args.bert_annoy         = get_model(args.bert_annoy,         "plot_embeddings_bert_1.ann")
args.metadata           = get_model(args.metadata,           "annoy_metadata_bert_1.json")

# ─── Genres ───────────────────────────────────────────────────────────────────
genres = [
    "action", "animation", "documentary", "comedy", "drama",
    "fantasy", "horror", "romance", "science fiction", "thriller"
]

# =============================================================================
# VISION — MovieNet
# =============================================================================

poster_model = MovieNet(num_classes=10).to(device)
poster_model.load_state_dict(torch.load(args.model_path, map_location=device))
poster_model.eval()
print("Modèle MovieNet chargé.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =============================================================================
# NLP — Utilitaires partagés
# =============================================================================

stop_words = set(stopwords.words('english'))

def improved_tokenizer(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [t for t in text.strip().split() if t not in stop_words and len(t) > 1]

# LabelEncoder
with open(args.label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)
print("LabelEncoder chargé.")

def decode_label(idx):
    if label_encoder is not None:
        return label_encoder.inverse_transform([idx])[0]
    return genres[idx] if idx < len(genres) else str(idx)

# Vocabulaire stoi
with open(args.vocab_path, 'rb') as f:
    stoi = pickle.load(f)
print(f"Vocabulaire chargé : {len(stoi)} tokens")

vocab_size = len(stoi)

# Metadata films (plot + category)
with open(args.metadata, 'r') as f:
    movie_metadata = json.load(f)
print(f"Metadata chargée : {len(movie_metadata)} films")

def format_recommendations(indices, distances):
    results = []
    for idx, dist in zip(indices, distances):
        if idx < len(movie_metadata):
            plot_preview = movie_metadata[idx]['movie_plot']
            genre = movie_metadata[idx]['movie_category']
        else:
            plot_preview = "Synopsis non disponible"
            genre = "inconnu"
        results.append({
            "rank": len(results) + 1,
            "plot_preview": plot_preview,
            "genre": genre,
            "distance": round(float(dist), 4)
        })
        if len(results) >= 5:
            break
    return results

# =============================================================================
# NLP — Modèle Bag-of-Words
# =============================================================================

class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)

    def get_embeddings(self, text):
        return torch.mean(self.embedding(text), 1)

    def forward(self, text):
        return self.fc(self.get_embeddings(text))

bow_model = TextClassificationModel(vocab_size, embed_dim=100, num_class=10).to(device)
bow_model.load_state_dict(torch.load(args.bow_path, map_location=device))
bow_model.eval()
print("Modèle BoW chargé.")

# Construction de l'index Annoy BoW via bow_model.get_embeddings
# (même source que get_bow_vector pour garantir la cohérence à la requête)
print("Construction de l'index Annoy BoW par film...")
bow_annoy_index = AnnoyIndex(100, metric='angular')
for film_idx, film in enumerate(movie_metadata):
    tokens = improved_tokenizer(film['movie_plot'])
    if tokens:
        indices = [stoi.get(t, stoi.get("<unk>", 0)) for t in tokens]
        tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(device)
        with torch.no_grad():
            vector = bow_model.get_embeddings(tensor).squeeze(0).cpu().numpy()
    else:
        vector = np.zeros(100)
    bow_annoy_index.add_item(film_idx, vector)
bow_annoy_index.build(n_trees=10)
print(f"Index Annoy BoW construit : {len(movie_metadata)} films.")

# =============================================================================
# NLP — Modèle LSTM
# =============================================================================

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim=10):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 100, num_layers=2, dropout=0.2, batch_first=True)
        self.fc = nn.Linear(100, output_dim)

    def get_embeddings(self, text, text_lengths):
        embedded = self.embedding(text)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return hidden[-1]

    def forward(self, text, text_lengths):
        return self.fc(self.get_embeddings(text, text_lengths))

lstm_model = TextClassifier(vocab_size, embedding_dim=128, output_dim=10).to(device)
lstm_model.load_state_dict(torch.load(args.lstm_path, map_location=device))
lstm_model.eval()
print("Modèle LSTM chargé.")

# Construction de l'index Annoy LSTM via lstm_model.embedding
# (même source que get_lstm_vector pour garantir la cohérence à la requête)
print("Construction de l'index Annoy LSTM par film...")
lstm_annoy_index = AnnoyIndex(128, metric='angular')
for film_idx, film in enumerate(movie_metadata):
    tokens = improved_tokenizer(film['movie_plot'])
    if tokens:
        indices = [stoi.get(t, stoi.get("<unk>", 0)) for t in tokens]
        tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = lstm_model.embedding(tensor)           # (1, seq_len, 128)
            vector = torch.mean(emb, dim=1).squeeze(0)  # (128,)
        vector = vector.cpu().numpy()
    else:
        vector = np.zeros(128)
    lstm_annoy_index.add_item(film_idx, vector)
lstm_annoy_index.build(n_trees=10)
print(f"Index Annoy LSTM construit : {len(movie_metadata)} films.")

# =============================================================================
# NLP — Modèle BERT (DistilBERT)
# =============================================================================

class BertClf(nn.Module):
    def __init__(self, distilbert, num_labels=10):
        super().__init__()
        self.distilbert = distilbert
        for param in distilbert.parameters():
            param.requires_grad = False
        for name, param in distilbert.named_parameters():
            if any(k in name for k in ['transformer.layer.4', 'transformer.layer.5',
                                        'classifier', 'pre_classifier']):
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        return out.logits, out.hidden_states

bert_tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
distilbert_base = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=10,
    output_hidden_states=True, output_attentions=True)
bert_model = BertClf(distilbert_base).to(device)
bert_model.distilbert.load_state_dict(
    torch.load(args.bert_path, map_location=device), strict=False)
bert_model.eval()
print("Modèle BERT chargé.")

bert_annoy_index = AnnoyIndex(768, metric='angular')
bert_annoy_index.load(args.bert_annoy)
print("Index Annoy BERT chargé.")

# =============================================================================
# Utilitaires vectorisation
# =============================================================================

def text_to_tensor(tokens):
    return torch.tensor([stoi.get(t, stoi.get("<unk>", 0)) for t in tokens], dtype=torch.int64)

def get_bow_vector(tokens):
    indices = [stoi.get(t, stoi.get("<unk>", 0)) for t in tokens]
    if not indices:
        return np.zeros(100)
    text_tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(device)
    with torch.no_grad():
        vector = bow_model.get_embeddings(text_tensor)
    return vector.squeeze(0).cpu().numpy()

def get_lstm_vector(tokens):
    indices = [stoi.get(t, stoi.get("<unk>", 0)) for t in tokens]
    if not indices:
        return np.zeros(128)
    text_tensor = torch.tensor(indices, dtype=torch.int64).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = lstm_model.embedding(text_tensor)      # (1, seq_len, 128)
        vector = torch.mean(emb, dim=1).squeeze(0)  # (128,)
    return vector.cpu().numpy()

def get_bert_cls_vector(plot_text):
    encoding = bert_tokenizer(plot_text, truncation=True, padding=True,
                              max_length=512, return_tensors='pt')
    with torch.no_grad():
        _, hidden_states = bert_model(
            encoding['input_ids'].to(device),
            encoding['attention_mask'].to(device))
    return hidden_states[-1][:, 0, :].squeeze(0).cpu().numpy()

# =============================================================================
# ROUTES VISION
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    img_pil = Image.open(io.BytesIO(request.data)).convert("RGB")
    tensor = transform(img_pil).to(device).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(poster_model(tensor), dim=1)[0].cpu().numpy()
    result = {genres[i]: float(probs[i]) for i in range(len(genres))}
    return jsonify({"prediction_genre": genres[probs.argmax()], "probabilities": result})


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    tensors = [transform(Image.open(f.stream).convert("RGB"))
               for f in request.files.getlist("images[]")]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        _, preds = poster_model(batch).max(1)
    return jsonify({"predictions_index": preds.tolist(),
                    "predictions_genre": [genres[p] for p in preds.tolist()]})

# =============================================================================
# ROUTES NLP — Prédiction de genre
# =============================================================================

@app.route('/predict_plot_bow', methods=['POST'])
def predict_plot_bow():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    tokens = improved_tokenizer(plot)
    if not tokens:
        return jsonify({"error": "Synopsis vide après tokenisation"}), 400
    tensor = text_to_tensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        _, predicted = bow_model(tensor).max(1)
    return jsonify({"model": "bag_of_words", "prediction_genre": decode_label(predicted.item())})


@app.route('/predict_plot_lstm', methods=['POST'])
def predict_plot_lstm():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    tokens = improved_tokenizer(plot)
    if not tokens:
        return jsonify({"error": "Synopsis vide après tokenisation"}), 400
    tensor = text_to_tensor(tokens).unsqueeze(0).to(device)
    length = torch.tensor([len(tokens)], dtype=torch.int64).to(device)
    with torch.no_grad():
        _, predicted = lstm_model(tensor, length).max(1)
    return jsonify({"model": "lstm", "prediction_genre": decode_label(predicted.item())})


@app.route('/predict_plot_bert', methods=['POST'])
def predict_plot_bert():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    encoding = bert_tokenizer(plot, truncation=True, padding=True,
                              max_length=512, return_tensors='pt')
    with torch.no_grad():
        logits, _ = bert_model(encoding['input_ids'].to(device),
                               encoding['attention_mask'].to(device))
        _, predicted = logits.max(1)
    return jsonify({"model": "bert", "prediction_genre": decode_label(predicted.item())})

# =============================================================================
# ROUTES NLP — Recommandation de films
# =============================================================================

@app.route('/recommend_bow', methods=['POST'])
def recommend_bow():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    tokens = improved_tokenizer(plot)
    if not tokens:
        return jsonify({"error": "Synopsis vide après tokenisation"}), 400
    query_vector = get_bow_vector(tokens)
    indices, distances = bow_annoy_index.get_nns_by_vector(query_vector, 6, include_distances=True)
    return jsonify({"model": "bag_of_words",
                    "recommendations": format_recommendations(indices, distances)})


@app.route('/recommend_lstm', methods=['POST'])
def recommend_lstm():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    tokens = improved_tokenizer(plot)
    if not tokens:
        return jsonify({"error": "Synopsis vide après tokenisation"}), 400
    query_vector = get_lstm_vector(tokens)
    indices, distances = lstm_annoy_index.get_nns_by_vector(query_vector, 6, include_distances=True)
    return jsonify({"model": "lstm",
                    "recommendations": format_recommendations(indices, distances)})


@app.route('/recommend_bert', methods=['POST'])
def recommend_bert():
    plot = request.get_json().get("plot", "")
    if not plot:
        return jsonify({"error": "Champ 'plot' manquant"}), 400
    query_vector = get_bert_cls_vector(plot)
    indices, distances = bert_annoy_index.get_nns_by_vector(query_vector, 6, include_distances=True)
    return jsonify({"model": "bert",
                    "recommendations": format_recommendations(indices, distances)})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5075, debug=False)
