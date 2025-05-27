"""Test GPT-2 model loading"""
import time
from transformers import GPT2Model, GPT2Tokenizer

print("Loading GPT-2 model...")
start = time.time()
model = GPT2Model.from_pretrained('gpt2')
print(f"Model loaded in {time.time() - start:.2f} seconds")

print("Loading tokenizer...")
start = time.time()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
print(f"Tokenizer loaded in {time.time() - start:.2f} seconds")

print("Model ready!")