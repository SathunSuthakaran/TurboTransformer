from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def run_model():
    quant_path = "quant-path"
    id = "Qwen/Qwen1.5-1.8B" #relatively small in terms of paramters
    tokenizer = AutoTokenizer.from_pretrained(id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(id, trust_remote_code=True).to('cpu')
    prompt = 'who is the best poker player in the world'
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
        )
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    
    with torch.no_grad():
        output = quantized_model.generate(**inputs, max_new_tokens = 150)
        print(tokenizer.decode(output[0], skip_special_tokens=True))
    
if __name__ == '__main__':
    run_model()
    
    