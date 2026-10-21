import gradio as gr
from safesight import SafeSightPipeline
from safesight.grounding import MockGrounder
def analyze(description, scene_tags, ablate, image):
    if not description or not description.strip():
        return {"error": "Please enter a scene description."}
    tags = [
        tag.strip()
        for tag in (scene_tags or "").split(",")
        if tag.strip()
    ]
    grounder = MockGrounder(tags)
    pipeline = SafeSightPipeline(
        grounder,
        ablate=[
            item.strip()
            for item in (ablate or "").split(",")
            if item.strip()
        ],
    )
    return pipeline.run(image, description)
with gr.Blocks(title="SafeSight AI - Research Interface") as demo:
    gr.Markdown(
        """
        # SafeSight AI
        ### Scene Understanding and Hallucination Analysis
        Enter a scene description and optionally provide scene tags
        or upload an image to analyze the scene.
        **Research prototype:** Results depend on the configured
        grounding component and are not verified safety guidance.
        """
    )
    description = gr.Textbox(
        label="Scene Description",
        placeholder="A person is standing near a road crossing.",
        lines=3,
    )
    scene_tags = gr.Textbox(
        label="Scene Tags (comma-separated)",
        placeholder="person, road, crossing",
    )
    ablate = gr.Textbox(
        label="Ablation Options (optional)",
        placeholder="Leave empty for the default pipeline",
    )
    image = gr.Image(
        label="Optional Scene Image",
        type="pil",
    )
    submit = gr.Button(
        "Analyze Scene",
        variant="primary",
    )
    output = gr.JSON(
        label="Analysis Result",
    )
    submit.click(
        fn=analyze,
        inputs=[
            description,
            scene_tags,
            ablate,
            image,
        ],
        outputs=output,
    )
if __name__ == "__main__":
    demo.launch()