import torch
import time
import gradio as gr
from model_loader import load_model, load_quantized, load_tokenizer

tokenizer = load_tokenizer()
model = load_model()
quantized_model = load_quantized()
results = {}

def gen_response(prompt='who is your favorite poker player', use_quantized=True):
    if use_quantized:
        model = quantized_model
    inputs = tokenizer(prompt, return_tensors='pt')
    
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 150
        )
    end = time.time()
    
    resp = tokenizer.decode(outputs[0], skip_special_tokens = True)
    time_elapsed = f"{end - start:.2f} seconds"
    return resp, time_elapsed

interface = gr.Interface(
    fn=gen_response,
    inputs = [
        gr.Textbox(lines=2, label="Prompt"),
        gr.Checkbox(label="Use Quantized Models", value=True)
        ],
    outputs=[
    gr.Textbox(label="Generated Output"),
    gr.Textbox(label="Inference Time")
],
title="Qwen Inference Demo",
description="Run Qwen1.5-1.8B in FP32 or INT8 (quantized) mode."
)

if __name__ == "__main__":
    interface.launch(share=True)