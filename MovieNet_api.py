import argparse
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
import io
from MovieNet import MovieNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default='saved_models/movie_poster_model.pth')
args = parser.parse_args()

# Load model
model = MovieNet(num_classes=10).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# Transformations adaptées aux posters
transform = transforms.Compose([
    transforms.Resize((185, 298)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5075, debug=False)