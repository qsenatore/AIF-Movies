import gradio as gr
import requests
import io
import matplotlib.pyplot as plt

# Si Docker est utilisé
BASE_URL = "http://api:5075"

# Si c'est utilisé en local
# BASE_URL = "http://localhost:5075"

genres = ["action", "animation", "documentary", "comedy", "drama",
          "fantasy", "horror", "romance", "science fiction", "thriller"]

# =============================================================================
# Fonctions — Vision (poster)
# =============================================================================

def predict_genre_from_poster(image):
    if image is None:
        return None, None
    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")
    response = requests.post(f"{BASE_URL}/predict", data=img_binary.getvalue())
    result = response.json()
    probs = result["probabilities"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(probs.keys(), probs.values(), color='skyblue')
    ax.set_ylabel("Probabilité")
    ax.set_xlabel("Genres")
    ax.set_title(f"Prédiction : {result['prediction_genre']}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    ax.set_ylim(0, max(probs.values()) * 1.15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha='center', va='bottom')
    return image, fig

# =============================================================================
# Fonctions — Prédiction de genre (NLP)
# =============================================================================

def predict_genre_nlp(plot_text, model_name, route):
    if not plot_text or not plot_text.strip():
        return "⚠️ Veuillez entrer un synopsis."
    try:
        response = requests.post(f"{BASE_URL}/{route}", json={"plot": plot_text}, timeout=30)
        if response.status_code == 200:
            genre = response.json().get("prediction_genre", "inconnu")
            return f"🎬 Genre prédit ({model_name}) : **{genre}**"
        return f"❌ Erreur API : {response.json().get('error', 'Erreur inconnue')}"
    except Exception as e:
        return f"❌ Connexion impossible à l'API : {str(e)}"

def predict_bow(plot):
    return predict_genre_nlp(plot, "Bag-of-Words", "predict_plot_bow")

def predict_lstm(plot):
    return predict_genre_nlp(plot, "LSTM", "predict_plot_lstm")

def predict_bert(plot):
    return predict_genre_nlp(plot, "BERT", "predict_plot_bert")

# =============================================================================
# Fonctions — Recommandation de films (Annoy)
# =============================================================================

def recommend_films(plot_text, route, model_name):
    if not plot_text or not plot_text.strip():
        return "⚠️ Veuillez entrer un synopsis."
    try:
        response = requests.post(f"{BASE_URL}/{route}", json={"plot": plot_text}, timeout=60)
        if response.status_code != 200:
            return f"❌ Erreur API : {response.json().get('error', 'Erreur inconnue')}"
        recs = response.json().get("recommendations", [])
        if not recs:
            return "Aucune recommandation trouvée."
        lines = [f"### 🎬 Films recommandés par {model_name}\n"]
        for r in recs:
            lines.append(
                f"**#{r['rank']}** — Genre : `{r['genre']}` (distance : {r['distance']})\n"
                f"> {r['plot_preview']}\n"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Connexion impossible à l'API : {str(e)}"

def recommend_bow(plot):
    return recommend_films(plot, "recommend_bow", "Bag-of-Words")

def recommend_lstm(plot):
    return recommend_films(plot, "recommend_lstm", "LSTM")

def recommend_bert(plot):
    return recommend_films(plot, "recommend_bert", "BERT")

# =============================================================================
# Interface Gradio
# =============================================================================

with gr.Blocks(title="Classificateur de genres de films") as demo:

    gr.Markdown("# 🎥 Classificateur & Recommandateur de films")
    gr.Markdown("Prédisez le genre d'un film à partir de son **affiche** ou de son **synopsis**, "
                "et obtenez des **recommandations** de films similaires.")

    # ── Onglet 1 : Affiche ────────────────────────────────────────────────────
    with gr.Tab("🖼️ Prédiction par affiche"):
        gr.Markdown("### Téléversez une affiche de film pour prédire son genre.")
        with gr.Row():
            poster_input = gr.Image(type="pil", label="Affiche du film")
            poster_output_img = gr.Image(type="pil", label="Affiche analysée")
        poster_output_plot = gr.Plot(label="Probabilités par genre")
        poster_btn = gr.Button("🔍 Prédire le genre", variant="primary")
        poster_btn.click(fn=predict_genre_from_poster,
                         inputs=poster_input,
                         outputs=[poster_output_img, poster_output_plot])

    # ── Onglet 2 : Bag-of-Words ───────────────────────────────────────────────
    with gr.Tab("📝 Bag-of-Words"):
        gr.Markdown("### Bag-of-Words — Prédiction de genre et recommandations")
        bow_input = gr.Textbox(lines=5, placeholder="Entrez le synopsis du film ici...",
                               label="Synopsis")
        with gr.Row():
            bow_predict_btn = gr.Button("🔍 Prédire le genre", variant="primary")
            bow_recommend_btn = gr.Button("🎯 Recommander des films similaires", variant="secondary")
        bow_genre_output = gr.Markdown(label="Genre prédit")
        bow_reco_output = gr.Markdown(label="Recommandations")
        bow_predict_btn.click(fn=predict_bow, inputs=bow_input, outputs=bow_genre_output)
        bow_recommend_btn.click(fn=recommend_bow, inputs=bow_input, outputs=bow_reco_output)

    # ── Onglet 3 : LSTM ───────────────────────────────────────────────────────
    with gr.Tab("🔁 LSTM"):
        gr.Markdown("### LSTM — Prédiction de genre et recommandations")
        lstm_input = gr.Textbox(lines=5, placeholder="Entrez le synopsis du film ici...",
                                label="Synopsis")
        with gr.Row():
            lstm_predict_btn = gr.Button("🔍 Prédire le genre", variant="primary")
            lstm_recommend_btn = gr.Button("🎯 Recommander des films similaires", variant="secondary")
        lstm_genre_output = gr.Markdown(label="Genre prédit")
        lstm_reco_output = gr.Markdown(label="Recommandations")
        lstm_predict_btn.click(fn=predict_lstm, inputs=lstm_input, outputs=lstm_genre_output)
        lstm_recommend_btn.click(fn=recommend_lstm, inputs=lstm_input, outputs=lstm_reco_output)

    # ── Onglet 4 : BERT ───────────────────────────────────────────────────────
    with gr.Tab("🤗 BERT"):
        gr.Markdown("### BERT (DistilBERT) — Prédiction de genre et recommandations")
        bert_input = gr.Textbox(lines=5, placeholder="Entrez le synopsis du film ici...",
                                label="Synopsis")
        with gr.Row():
            bert_predict_btn = gr.Button("🔍 Prédire le genre", variant="primary")
            bert_recommend_btn = gr.Button("🎯 Recommander des films similaires", variant="secondary")
        bert_genre_output = gr.Markdown(label="Genre prédit")
        bert_reco_output = gr.Markdown(label="Recommandations")
        bert_predict_btn.click(fn=predict_bert, inputs=bert_input, outputs=bert_genre_output)
        bert_recommend_btn.click(fn=recommend_bert, inputs=bert_input, outputs=bert_reco_output)

if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860)