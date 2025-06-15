"""
Extract activations for ALL GPT-2 tokens (50,257)
Process in chunks to manage memory
"""

import torch
import numpy as np
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer
import json
from datetime import datetime
import gc
import os

class AllTokenActivationExtractor:
    """Extract activations for the entire GPT-2 vocabulary"""
    
    def __init__(self, model_name='gpt2', chunk_size=5000):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading {model_name} model...")
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.vocab_size = len(self.tokenizer)
        print(f"Model loaded. Vocabulary size: {self.vocab_size}")
        print(f"Using device: {self.device}")
        print(f"Processing in chunks of {chunk_size}")
        
    def extract_single_token_activation(self, token_id: int) -> np.ndarray:
        """Extract activation for a single token"""
        with torch.no_grad():
            # Create input (single token)
            input_ids = torch.tensor([[token_id]], device=self.device)
            
            # Get hidden states
            outputs = self.model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract activations from each layer (skip embedding layer)
            activations = []
            for layer_idx in range(1, 13):  # Layers 0-11
                layer_act = hidden_states[layer_idx][0, 0, :].cpu().numpy()
                activations.append(layer_act)
            
            return np.array(activations)  # Shape: (12, 768)
    
    def extract_chunk(self, start_idx: int, end_idx: int) -> np.ndarray:
        """Extract activations for a chunk of tokens"""
        chunk_size = end_idx - start_idx
        chunk_activations = []
        
        for token_id in range(start_idx, end_idx):
            if (token_id - start_idx) % 100 == 0:
                print(f"  Token {token_id} ({token_id - start_idx + 1}/{chunk_size})", end='\r')
            
            activation = self.extract_single_token_activation(token_id)
            chunk_activations.append(activation)
        
        return np.array(chunk_activations)
    
    def extract_all_activations(self, output_dir: Path, resume_from: int = 0):
        """Extract activations for all tokens, saving in chunks"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we're resuming
        if resume_from > 0:
            print(f"Resuming from token {resume_from}")
        
        # Process in chunks
        for chunk_start in range(resume_from, self.vocab_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, self.vocab_size)
            chunk_id = chunk_start // self.chunk_size
            
            # Check if chunk already exists
            chunk_file = output_dir / f"activations_chunk_{chunk_id:04d}.npy"
            if chunk_file.exists():
                print(f"\nChunk {chunk_id} already exists, skipping...")
                continue
            
            print(f"\nProcessing chunk {chunk_id}: tokens {chunk_start}-{chunk_end}")
            start_time = datetime.now()
            
            # Extract chunk
            chunk_activations = self.extract_chunk(chunk_start, chunk_end)
            
            # Save chunk
            np.save(chunk_file, chunk_activations)
            print(f"\nSaved {chunk_file}")
            
            # Log progress
            elapsed = (datetime.now() - start_time).total_seconds()
            tokens_per_sec = (chunk_end - chunk_start) / elapsed
            remaining_tokens = self.vocab_size - chunk_end
            eta_seconds = remaining_tokens / tokens_per_sec if tokens_per_sec > 0 else 0
            
            print(f"Chunk time: {elapsed:.1f}s ({tokens_per_sec:.1f} tokens/sec)")
            print(f"Progress: {chunk_end}/{self.vocab_size} ({chunk_end/self.vocab_size*100:.1f}%)")
            if eta_seconds > 0:
                print(f"ETA: {eta_seconds/60:.1f} minutes")
            
            # Clear GPU cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
        
        print("\nAll chunks extracted!")
        
        # Create metadata
        metadata = {
            'model_name': self.model_name,
            'vocab_size': self.vocab_size,
            'n_layers': 12,
            'hidden_dim': 768,
            'chunk_size': self.chunk_size,
            'n_chunks': (self.vocab_size + self.chunk_size - 1) // self.chunk_size,
            'extraction_date': datetime.now().isoformat()
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {output_dir}/metadata.json")
        
        return metadata
    
    def load_all_activations(self, output_dir: Path) -> np.ndarray:
        """Load all activation chunks into a single array"""
        
        print("Loading activation chunks...")
        
        # Get metadata
        with open(output_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        n_chunks = metadata['n_chunks']
        all_activations = []
        
        for chunk_id in range(n_chunks):
            chunk_file = output_dir / f"activations_chunk_{chunk_id:04d}.npy"
            if not chunk_file.exists():
                raise FileNotFoundError(f"Missing chunk: {chunk_file}")
            
            chunk_data = np.load(chunk_file)
            all_activations.append(chunk_data)
            print(f"Loaded chunk {chunk_id}/{n_chunks-1}", end='\r')
        
        print("\nConcatenating chunks...")
        full_activations = np.concatenate(all_activations, axis=0)
        print(f"Full activation shape: {full_activations.shape}")
        
        return full_activations
    
    def extract_sample_activations(self, token_ids: list, output_file: Path):
        """Extract activations for a specific set of tokens"""
        
        print(f"Extracting activations for {len(token_ids)} specific tokens...")
        activations = []
        
        for i, token_id in enumerate(token_ids):
            if i % 100 == 0:
                print(f"Processing token {i}/{len(token_ids)}", end='\r')
            
            activation = self.extract_single_token_activation(token_id)
            activations.append(activation)
        
        activations = np.array(activations)
        np.save(output_file, activations)
        print(f"\nSaved to {output_file}")
        
        return activations


def main():
    """Extract all GPT-2 token activations"""
    
    output_dir = Path("experiments/gpt2/all_tokens/activations")
    
    # Check if we need to resume
    resume_from = 0
    if output_dir.exists():
        # Find the last completed chunk
        chunk_files = list(output_dir.glob("activations_chunk_*.npy"))
        if chunk_files:
            last_chunk = max(int(f.stem.split('_')[-1]) for f in chunk_files)
            resume_from = (last_chunk + 1) * 5000
            print(f"Found {len(chunk_files)} existing chunks, will resume from token {resume_from}")
    
    # Initialize extractor
    extractor = AllTokenActivationExtractor(chunk_size=5000)
    
    # Extract all activations
    metadata = extractor.extract_all_activations(output_dir, resume_from=resume_from)
    
    print("\n=== Extraction Complete ===")
    print(f"Total tokens: {metadata['vocab_size']}")
    print(f"Total chunks: {metadata['n_chunks']}")
    print(f"Output directory: {output_dir}")
    
    # Optional: Extract a sample for quick testing
    print("\nExtracting sample activations for testing...")
    
    # Sample: first 100 tokens, some subwords, some punctuation
    sample_ids = list(range(100)) + list(range(1000, 1100)) + list(range(10000, 10100))
    sample_file = output_dir.parent / "sample_activations.npy"
    extractor.extract_sample_activations(sample_ids, sample_file)


if __name__ == "__main__":
    main()