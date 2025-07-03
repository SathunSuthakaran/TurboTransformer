from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

id = "Qwen/Qwen1.5-1.8B" #relatively small in terms of paramters

def load_tokenizer():
    return AutoTokenizer.from_pretrained(id, trust_remote_code=True)

def load_model():
    return AutoModelForCausalLM.from_pretrained(id, trust_remote_code=True).to('cpu')

def load_quantized(model=None):
    if not model:
        model = load_model()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
        )
    return quantized_model    