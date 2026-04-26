import gradio as gr
import plotly.graph_objects as go

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
    "Цей коментар абсолютно нормальний і доброзичливий.",
    "З самого ранку повна хата людей і дітей!",
    "Я тебе вб'ю, якщо ти ще раз так скажеш."
]


def build_chart(scores: dict[str, float]) -> go.Figure:
    labels = list(scores.keys())
    probs  = list(scores.values())
    colors = [LABEL_COLORS.get(l, "#4048b9") for l in labels]

    fig = go.Figure(go.Bar(
        x=labels,
        y=probs,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=40),
        yaxis=dict(
            range=[0, 1.12],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#e9ecef",
            zeroline=False,
        ),
        xaxis=dict(tickfont=dict(size=13)),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", size=12, color="#343a40"),
        showlegend=False,
    )
    return fig


def build_verdict(binary: dict[str, int]) -> str:
    toxic_labels = [
        label.replace("_", " ") for label, is_toxic in binary.items()
        if is_toxic
    ]
    
    if toxic_labels:
        lines = "\n".join(f"  🔸 {label.upper()}" for label in toxic_labels if label != "toxic")
        return f"🔴 The comment is toxic. It belongs to the following toxicity categories:\n{lines}"
    return "🟢 The comment is non-toxic."


def classify(text: str, mode: str):
    if not text or not text.strip():
        return None, "", gr.update(visible=False)

    scores = predict(text, model, tokenizer)
    fig = build_chart(scores)

    if mode == "Tuned":
        binary = predict_with_thresholds(text, model, tokenizer) 
    else:
        binary = predict_with_thresholds(text, model, tokenizer, thresholds=[0.5] * len(scores))

    verdict = build_verdict(binary)

    return fig, verdict, gr.update(visible=False)   # chart hidden by default


def toggle_chart(current_visible):
    return gr.update(visible=not current_visible), not current_visible


with gr.Blocks(title="Ukrainian Toxicity Detector") as demo:

    gr.Markdown("""
    # 🛡️ Ukrainian Toxicity Detector
    ### Multi-label toxicity analysis powered by XLM-RoBERTa.
    """)

    with gr.Row():
        with gr.Column(scale=5):
            text_input = gr.Textbox(
                label="Input Comment",
                placeholder="Введіть коментар для аналізу…",
                lines=4,
            )
           
            with gr.Row():
                gr.Examples(examples=EXAMPLE_TEXTS, inputs=text_input, label="Examples")
            
            mode_radio = gr.Radio(
                choices=["Default", "Tuned"],
                value="Default",
                label="Threshold Mode",
                info="""
                **Default** — uses a fixed 0.5 threshold for all labels.
                **Tuned** — uses per-label thresholds tuned on the validation set.
                """
            )
            with gr.Row():
                analyze_btn = gr.Button("Analyze", variant="primary")
                clear_btn = gr.ClearButton([text_input], value="Clear")

            

        with gr.Column(scale=6):
            verdict_out = gr.Textbox(label="Verdict", interactive=False)
            toggle_btn = gr.Button("Show all probabilities ▼", variant="secondary")
            chart_out = gr.Plot(label="Toxicity Probabilities", visible=False, show_label=False)
            
    chart_visible = gr.State(False)

    analyze_btn.click(
        fn=classify,
        inputs=[text_input, mode_radio],
        outputs=[chart_out, verdict_out, chart_out]
    )
    text_input.submit(
        fn=classify,
        inputs=[text_input, mode_radio],
        outputs=[chart_out, verdict_out, chart_out]
    )
    toggle_btn.click(
        fn=toggle_chart,
        inputs=[chart_visible],
        outputs=[chart_out, chart_visible]
    )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="sky"))