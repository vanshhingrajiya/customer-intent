import os
import json
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

MODEL_WEIGHTS = "./banking77_distilbert.pth"
LABELS_FILE = "./labels.json"
METRICS_FILE = "./metrics.json"

if not os.path.exists(MODEL_WEIGHTS) or not os.path.exists(LABELS_FILE):
    raise FileNotFoundError(f"Model weights or labels file not found. Please train in Colab, download '{os.path.basename(MODEL_WEIGHTS)}' and '{os.path.basename(LABELS_FILE)}' and place them here.")

print("Loading labels...")
with open(LABELS_FILE, "r") as f:
    labels = json.load(f)
num_labels = len(labels)

print("Loading model and tokenizer...")
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=torch.device('cpu')))
model.eval()

# Display metrics if available
metrics_text = "Metrics not loaded."
if os.path.exists(METRICS_FILE):
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
        metrics_text = f"Accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f} | F1: {metrics.get('eval_f1', 'N/A'):.4f} | Precision: {metrics.get('eval_precision', 'N/A'):.4f}"

def predict_intent(text):
    if not text.strip():
        return "Please enter a valid customer support query."
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    predicted_label = labels[predicted_class_id]
    
    return predicted_label

print("Starting Gradio app...")

# Modern UI setup using Gradio
with gr.Blocks() as demo:
    gr.Markdown(
        f"""
        # 🏦 Customer Support Intent Pattern Recognition
        ### Bank Query Routing System
        Type in a short customer service inquiry, and this AI model (DistilBERT) will identify the intent category to route it appropriately.
        
        **Model Evaluation Metrics (from Training):**
        {metrics_text}
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Customer Inquiry", 
                placeholder="e.g. My card was stolen, please block it.",
                lines=4
            )
            submit_btn = gr.Button("Recognize Intent", variant="primary")
            
        with gr.Column(scale=1):
            text_output = gr.Textbox(
                label="Recognized Intent Route", 
                interactive=False,
                lines=2
            )
            
    submit_btn.click(fn=predict_intent, inputs=text_input, outputs=text_output)
    text_input.submit(fn=predict_intent, inputs=text_input, outputs=text_output)

    gr.Examples(
        examples=[
            "I have forgotten my pin code.",
            "Can I use my card abroad?",
            "My payment was declined, why?",
            "How long does a transfer to Europe take?",
            "I need to cancel my lost card."
        ],
        inputs=text_input,
        label="Sample Inquiries"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", share=True, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
