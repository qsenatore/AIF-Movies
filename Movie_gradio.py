import gradio as gr
import requests
import io
import matplotlib.pyplot as plt

API_URL = "http://api:5075/predict"
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
    
    return image, fig

interface = gr.Interface(
    fn=predict_genre,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil"), gr.Plot()],
    title="Classificateur de genres de films",
    description="Téléversez une affiche de film pour prédire son genre. L'histogramme montre les probabilités pour les 10 genres."
)

if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    interface.launch(server_name="0.0.0.0", server_port=7860)
