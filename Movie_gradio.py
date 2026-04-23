import gradio as gr
import requests
import io
import os
import matplotlib.pyplot as plt
from PIL import Image

# Si usage de Gradio SANS Docker : 
# API_URL = "http://127.0.0.1:5075/predict"
# RECO_API_URL = "http://127.0.0.1:5075/recommend"

# Si usage de Gradio AVEC Docker : 
API_URL = "http://api:5075/predict"
RECO_API_URL = "http://api:5075/recommend"

genres = ["action", "animation", "documentary", "comedy", "drama", 
          "fantasy", "horror", "romance", "science fiction", "thriller"]


def predict_genre(image):

    # Convert PIL image to binary
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    # Send image to API
    response = requests.post(API_URL, data=img_binary.getvalue())
    result = response.json()

    probs = result["probabilities"]
    
    # Créer un histogramme
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(probs.keys(), probs.values(), color='skyblue')

    ax.set_ylabel("Probabilité")
    ax.set_xlabel("Genres")
    ax.set_title(f"Prédiction : {result['prediction_genre']}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Ajuster la limite supérieure pour laisser de l'espace au texte
    max_prob = max(probs.values())
    ax.set_ylim(0, max_prob * 1.15)  # ajoute 15% d’espace au-dessus

    # Ajouter les valeurs de probabilité au-dessus de chaque barre
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.2f}", ha='center', va='bottom')
    
    return fig


def recommend_movies(image):

    # Convert PIL image to binary
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    # Send image to API
    response = requests.post(RECO_API_URL,files={"file": ("image.png", img_binary.getvalue(), "image/png")})
    result = response.json()

    paths = result["recommendations"]

    imgs = []
    for p in paths:
        if os.path.exists(p):
            imgs.append(Image.open(p))
    
    return imgs


interface = gr.Blocks()
with interface:
    gr.Markdown("IAF Project")

    with gr.Tab("Genre Prediction"):
        inp = gr.Image(type="pil")
        out_plot = gr.Plot()

        gr.Button("Predict").click(
            fn=predict_genre,
            inputs=inp,
            outputs=[out_plot]
        )

    with gr.Tab("Recommendations"):
        inp2 = gr.Image(type="pil")
        out2 = gr.Gallery(label="Similar Movies")

        gr.Button("Recommend").click(
            fn=recommend_movies,
            inputs=inp2,
            outputs=out2
        )

if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    interface.launch(server_name="0.0.0.0", server_port=7860)
