import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load your trained model
model = tf.keras.models.load_model("keras_model.h5", compile=False)

# 2. Class label map
label_map = {0: "Clean Street", 1: "Dirty Street"}

# 3. Prediction function
def predict_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.reshape((1, 224, 224, 3))
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    return label_map[idx], f"{preds[idx]*100:.2f}%"

# 4. Custom CSS to set your local background.jpg
css = """
.gradio-container {
    background-image: url('background.jpg');
    background-size: cover;
    background-position: center;
    color: white;
}
h1, h2, p, label {
    color: white !important;
}
"""

# 5. Build the Gradio app
with gr.Blocks(css=css, title="Clean Street Detector 🚀") as demo:
    gr.Markdown("## 🧼 Clean Street Detector")
    gr.Markdown("**Upload a street image** and let AI tell you if it's **Clean** or **Dirty**. 🚮🌍")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="📷 Upload Image", type="pil", interactive=True)
            analyze_btn = gr.Button("🚀 Analyze")

        with gr.Column():
            out_label = gr.Label(label="📄 Prediction")
            out_conf = gr.Label(label="📊 Confidence")

    analyze_btn.click(predict_image, inputs=img_input, outputs=[out_label, out_conf])

# 6. Launch the app
demo.launch()
