import time
import psutil
import os
import torch
from model_loader import load_model, load_tokenizer, load_quantized

def benchmark(prompt='who is the best poker player in the world?', max_tokens = 150):
    tokenizer = load_tokenizer()
    model = load_model()
    quantized_model = load_quantized()
    results = {}
    
    for model_name, model in [("FP32", model), ("INT8", quantized_model)]:
        inputs = tokenizer(prompt, return_tensors="pt")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        end = time.time()
        memory = psutil.Process(os.getpid()).memory_info().rss / 1e6
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results[model_name] = {
            "output": decoded,
            "latency": round(end - start, 2),
            "memory_mb": round(memory, 2)
        }
    return results


if __name__ == "__main__":
    results = benchmark()
    for version, info in results.items():
        print(f"=== {version} ===")
        print(f"Latency: {info['latency']} sec")
        print(f"Memory: {info['memory_mb']} MB")
        print(f"Output: {info['output']}\n")

