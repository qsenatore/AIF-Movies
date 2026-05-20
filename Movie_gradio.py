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
# Fonctions Partie 1
# =============================================================================

def predict_genre_from_poster(image):
    if image is None:
        return None

    img_binary = io.BytesIO()
    image.save(img_binary, format="PNG")

    response = requests.post(
        f"{BASE_URL}/predict",
        data=img_binary.getvalue()
    )

    result = response.json()
    probs = result["probabilities"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(list(probs.keys()), list(probs.values()), color='skyblue')

    ax.set_ylabel("Probabilité")
    ax.set_xlabel("Genres")
    ax.set_title(f"Prédiction : {result['prediction_genre']}")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    ax.set_ylim(0, max(probs.values()) * 1.15)

    for i, (k, v) in enumerate(probs.items()):
        ax.text(i, v + 0.01, f"{v:.2f}", ha='center')

    return fig

# =============================================================================
# Fonctions Partie 2
# =============================================================================



# =============================================================================
# Fonctions Partie 3 (Prédiction)
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
# Fonctions — Partie 3 (Recommandation)
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
# Fonctions — Partie 4
# =============================================================================



# =============================================================================
# Couleurs et style de l'interface
# =============================================================================

CSS = """
body {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: Arial, sans-serif;
}

.gradio-container {
    background-color: #ffffff !important;
}

/* Titres */
h1, h2, h3 {
    color: #1f6f4a !important;
    font-weight: 700;
}

/* Texte général */
p, span, label, div {
    color: #000000 !important;
}

/* =========================
   BOUTONS VERTS 3D
   ========================= */
button {
    background: linear-gradient(145deg, #2e8b57, #3cb371) !important;
    color: #ffffff !important;

    border: none !important;
    border-radius: 10px !important;

    padding: 10px 18px !important;
    font-weight: 600 !important;

    /* effet 3D */
    box-shadow:
        4px 4px 10px rgba(0, 0, 0, 0.25),
        -2px -2px 6px rgba(255, 255, 255, 0.6);

    transition: all 0.2s ease-in-out;
}

/* hover */
button:hover {
    transform: translateY(-2px);
    box-shadow:
        6px 6px 14px rgba(0, 0, 0, 0.3),
        -3px -3px 8px rgba(255, 255, 255, 0.7);
}

/* click */
button:active {
    transform: translateY(2px);
    box-shadow:
        2px 2px 6px rgba(0, 0, 0, 0.25);
}

/* blocs */
.block {
    background-color: #f7f7f7 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    border: 1px solid #dcdcdc;
}
"""

# =============================================================================
# Interface Gradio
# =============================================================================

with gr.Blocks(title="Movie AI", css=CSS) as demo:

    state_page = gr.State("home")

    # ─────────────────────────────────────────────────────────────
    # PAGE D'ACCUEIL
    # ─────────────────────────────────────────────────────────────
    with gr.Column(visible=True) as home_page:

        gr.Markdown("# Classificateur et recommandateur de films")

        gr.Markdown(
            "Bienvenue sur notre projet d'Artificial Intelligence Frameworks. "
            "Vous avez à votre disposition plusieurs modèles permettant la prédiction "
            "et la recommandation de films à partir de d'images et de textes."
        )

        gr.Markdown("""
### Navigation
""")

        btn_p1 = gr.Button("Partie 1")
        btn_p2 = gr.Button("Partie 2")
        btn_p3 = gr.Button("Partie 3")
        btn_p4 = gr.Button("Partie 4")
        gr.Markdown("""
---

### Équipe du projet

- SENATORE Quentin 
- HOFMANN Julien 
- PHILIPPE César 
""")

    # ─────────────────────────────────────────────────────────────
    # PARTIE 1
    # ─────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as page_p1:

        gr.Markdown("## Partie 1 - Prédiction du genre à partir du poster")

        poster_input = gr.Image(type="pil", label="Affiche du film")
        poster_output_plot = gr.Plot(label="Probabilités par genre")

        poster_btn = gr.Button("Prédire le genre", variant="primary")

        poster_btn.click(
            fn=predict_genre_from_poster,
            inputs=poster_input,
            outputs=poster_output_plot
        )

        back1 = gr.Button("Retour accueil")

    # ─────────────────────────────────────────────────────────────
    # PARTIE 2
    # ─────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as page_p2:

        gr.Markdown("## Partie 2 - Recommandation de films similaires à partir du poster")

        gr.Markdown("À compléter")

        back2 = gr.Button("Retour accueil")

    # ─────────────────────────────────────────────────────────────
    # PARTIE 3
    # ─────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as page_p3:

        gr.Markdown("## Partie 3 - Prédiction du genre et recommandation de films similaires à partir du synopsis")

        with gr.Tab("Bag of Words"):
            bow_input = gr.Textbox(lines=5, label="Synopsis")
        
            bow_predict_btn = gr.Button("Prédire le genre")
            bow_recommend_btn = gr.Button("Recommander des films similaires")
        
            bow_genre_out = gr.Markdown()
            bow_reco_out = gr.Markdown()
        
            bow_predict_btn.click(
                fn=predict_bow,
                inputs=bow_input,
                outputs=bow_genre_out
            )
        
            bow_recommend_btn.click(
                fn=recommend_bow,
                inputs=bow_input,
                outputs=bow_reco_out
            )

        with gr.Tab("LSTM"):
            lstm_input = gr.Textbox(lines=5, label="Synopsis")
        
            lstm_predict_btn = gr.Button("Prédire le genre")
            lstm_recommend_btn = gr.Button("Recommander des films similaires")
        
            lstm_genre_out = gr.Markdown()
            lstm_reco_out = gr.Markdown()
        
            lstm_predict_btn.click(
                fn=predict_lstm,
                inputs=lstm_input,
                outputs=lstm_genre_out
            )
        
            lstm_recommend_btn.click(
                fn=recommend_lstm,
                inputs=lstm_input,
                outputs=lstm_reco_out
            )
            
        with gr.Tab("BERT"):
            bert_input = gr.Textbox(lines=5, label="Synopsis")
        
            bert_predict_btn = gr.Button("Prédire le genre")
            bert_recommend_btn = gr.Button("Recommander des films similaires")
        
            bert_genre_out = gr.Markdown()
            bert_reco_out = gr.Markdown()
        
            bert_predict_btn.click(
                fn=predict_bert,
                inputs=bert_input,
                outputs=bert_genre_out
            )
        
            bert_recommend_btn.click(
                fn=recommend_bert,
                inputs=bert_input,
                outputs=bert_reco_out
            )
            
        back3 = gr.Button("Retour accueil")
        
    # ─────────────────────────────────────────────────────────────
    # PARTIE 4
    # ─────────────────────────────────────────────────────────────
    with gr.Column(visible=False) as page_p4:

        gr.Markdown("## Partie 4 - A compléter")

        gr.Markdown("À compléter")

        back4 = gr.Button("Retour accueil")

    # ─────────────────────────────────────────────────────────────
    # NAVIGATION LOGIC
    # ─────────────────────────────────────────────────────────────

    def show_home():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def show_p1():
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def show_p2():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def show_p3():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )

    def show_p4():
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )

    # Boutons navigation
    btn_p1.click(show_p1, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    btn_p2.click(show_p2, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    btn_p3.click(show_p3, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    btn_p4.click(show_p4, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    
    back1.click(show_home, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    back2.click(show_home, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    back3.click(show_home, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])
    back4.click(show_home, outputs=[home_page, page_p1, page_p2, page_p3, page_p4])

if __name__ == "__main__":
    print("Démarrage de l'application Gradio...")
    demo.launch(server_name="0.0.0.0", server_port=7860)