"""
🎨 Beautiful ML Demo with Gradio
Create stunning UIs for your ML models in minutes!
"""

import gradio as gr
import numpy as np
from PIL import Image
import torch
from transformers import pipeline


class MLDemo:
    """
    ML Demo with multiple interfaces
    """

    def __init__(self):
        # Initialize models (example)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def analyze_sentiment(self, text: str):
        """
        Analyze sentiment of input text
        """
        if not text:
            return "Please enter some text!"

        result = self.sentiment_analyzer(text)[0]

        return {
            "label": result['label'],
            "confidence": f"{result['score']:.2%}"
        }

    def classify_image(self, image: Image.Image):
        """
        Classify uploaded image (example)
        """
        if image is None:
            return "Please upload an image!"

        # Placeholder classification
        # Replace with your actual model
        classes = ["cat", "dog", "bird", "fish", "other"]
        probabilities = np.random.random(len(classes))
        probabilities = probabilities / probabilities.sum()

        return {
            class_name: float(prob)
            for class_name, prob in zip(classes, probabilities)
        }

    def predict_numeric(self, feature1: float, feature2: float, feature3: float):
        """
        Make predictions based on numeric inputs
        """
        # Placeholder prediction
        # Replace with your actual model
        prediction = (feature1 * 0.5 + feature2 * 0.3 + feature3 * 0.2)
        confidence = np.random.uniform(0.7, 0.99)

        return f"Prediction: {prediction:.2f} (Confidence: {confidence:.1%})"


def create_demo():
    """
    Create beautiful Gradio demo with multiple tabs
    """
    demo_app = MLDemo()

    # Custom CSS for styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .gr-button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=custom_css,
        title="🤖 MLOps Demo"
    ) as demo:

        gr.Markdown(
            """
            # 🚀 MLOps Ecosystem Demo

            ### Ultra-modern ML interfaces with Gradio 4.0+

            Explore different ML capabilities through interactive demos!
            """
        )

        with gr.Tabs():

            # Tab 1: Text Analysis
            with gr.Tab("📝 Text Analysis"):
                gr.Markdown("### Sentiment Analysis")

                with gr.Row():
                    with gr.Column():
                        text_input = gr.Textbox(
                            label="Enter text to analyze",
                            placeholder="Type something...",
                            lines=5
                        )
                        text_btn = gr.Button("🔍 Analyze Sentiment", variant="primary")

                    with gr.Column():
                        text_output = gr.JSON(label="Results")

                text_btn.click(
                    fn=demo_app.analyze_sentiment,
                    inputs=text_input,
                    outputs=text_output
                )

                gr.Examples(
                    examples=[
                        "This is amazing! I love it!",
                        "This is terrible and disappointing.",
                        "It's okay, nothing special."
                    ],
                    inputs=text_input
                )

            # Tab 2: Image Classification
            with gr.Tab("🖼️ Image Classification"):
                gr.Markdown("### Upload an image to classify")

                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Image"
                        )
                        image_btn = gr.Button("🎯 Classify Image", variant="primary")

                    with gr.Column():
                        image_output = gr.Label(
                            label="Classification Results",
                            num_top_classes=5
                        )

                image_btn.click(
                    fn=demo_app.classify_image,
                    inputs=image_input,
                    outputs=image_output
                )

            # Tab 3: Numeric Prediction
            with gr.Tab("📊 Numeric Prediction"):
                gr.Markdown("### Make predictions based on features")

                with gr.Row():
                    with gr.Column():
                        feature1 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            label="Feature 1"
                        )
                        feature2 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            label="Feature 2"
                        )
                        feature3 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=50,
                            label="Feature 3"
                        )
                        predict_btn = gr.Button("🎲 Predict", variant="primary")

                    with gr.Column():
                        predict_output = gr.Textbox(
                            label="Prediction Result",
                            lines=3
                        )

                predict_btn.click(
                    fn=demo_app.predict_numeric,
                    inputs=[feature1, feature2, feature3],
                    outputs=predict_output
                )

        gr.Markdown(
            """
            ---

            ### 🌟 Features:
            - 🎨 Beautiful, modern UI
            - ⚡ Real-time predictions
            - 📱 Mobile-friendly
            - 🔗 Easy to share

            Built with ❤️ using Gradio 4.0+
            """
        )

    return demo


if __name__ == "__main__":
    # Create and launch demo
    demo = create_demo()

    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
