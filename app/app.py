import sys
sys.path.insert(0, "/app")

import gradio as gr
import matplotlib.pyplot as plt

from toxicity_detector.model.predictor import predict, predict_with_thresholds
from toxicity_detector.model.loader import load_model

model, tokenizer = load_model()


LABEL_COLORS = {
    "toxic":         "#f13a25",
    "severe_toxic":  "#772219",
    "obscene":       "#e77e23",
    "threat":        "#2541bd",
    "insult":        "#72811b",
    "identity_hate": "#2c3e50",
}

EXAMPLE_TEXTS = [
    "Цей коментар абсолютно доброзичливий.",
    "З самого ранку повна хата людей і дітей!",
    "Я тебе вб'ю, якщо ти ще раз так скажеш."
]


def build_chart(scores: dict[str, float]) -> plt.Figure:
    labels = list(scores.keys())
    probs  = [v * 100 for v in scores.values()]  
    colors = [LABEL_COLORS.get(l, "#4048b9") for l in labels]

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(labels, probs, color=colors, alpha=0.85)

    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_title("Toxicity Category Probabilities", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_ylim(0, 115)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    plt.tight_layout()
    return fig


def build_verdict(binary: dict[str, int]) -> str:
    toxic_labels = [
        label.replace("_", " ") for label, is_toxic in binary.items()
        if is_toxic
    ]
    
    if toxic_labels:
        lines = "\n".join(f"  🔸 {label.upper()}" for label in toxic_labels if label != "toxic")
        return f"🔴 The comment is toxic." + (f" It belongs to the following toxicity categories:\n{lines}" if lines else "")
    return "🟢 The comment is non-toxic."


def classify(text: str, mode: str):
    if not text or not text.strip():
        return gr.update(value=None, visible=False), "", False

    scores = predict(text, model, tokenizer)
    fig = build_chart(scores)

    if mode == "Tuned":
        binary = predict_with_thresholds(text, model, tokenizer) 
    else:
        binary = predict_with_thresholds(text, model, tokenizer, thresholds=[0.5] * len(scores))

    verdict = build_verdict(binary)

    return gr.update(value=fig, visible=False), verdict, False 


def toggle_chart(current_visible):
    return gr.update(visible=not current_visible), not current_visible


with gr.Blocks(title="Ukrainian Toxicity Detector", theme=gr.themes.Soft(primary_hue="sky")) as demo:

    gr.Markdown("""
    # 🛡️ Ukrainian Toxicity Detector
    ### Multi-label toxicity analysis powered by XLM-RoBERTa.
    """)

    with gr.Row():
        with gr.Column(scale=5):
            text_input = gr.Textbox(
                label="Input Comment",
                placeholder="Enter a comment in Ukrainian for analysis...",
                lines=4,
            )
           
            with gr.Row():
                gr.Examples(examples=EXAMPLE_TEXTS, inputs=text_input, label="Examples")
            
            mode_radio = gr.Radio(
                choices=["Default", "Tuned"],
                value="Default",
                label="Threshold Mode",
            )
            
            gr.Markdown("""
            **Default** — uses a fixed 0.5 threshold for all labels.  
            **Tuned** — uses per-label thresholds tuned on the validation set.
            """)
            
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.ClearButton()

        with gr.Column(scale=6):
            verdict_out = gr.Textbox(label="Verdict", interactive=False)
            toggle_btn = gr.Button("Show all probabilities ▼", variant="secondary")
            chart_out = gr.Plot(label="Toxicity Probabilities", visible=False, show_label=False)
            
    chart_visible = gr.State(False)

    clear_btn.click(
        fn=lambda: ("", "", gr.update(visible=False, value=None), False),
        inputs=[],
        outputs=[text_input, verdict_out, chart_out, chart_visible]
    )
    analyze_btn.click(
        fn=classify,
        inputs=[text_input, mode_radio],
        outputs=[chart_out, verdict_out, chart_visible]
    )
    text_input.submit(
        fn=classify,
        inputs=[text_input, mode_radio],
        outputs=[chart_out, verdict_out, chart_visible]
    )
    toggle_btn.click(
        fn=toggle_chart,
        inputs=[chart_visible],
        outputs=[chart_out, chart_visible]
    )


if __name__ == "__main__":
    demo.launch()