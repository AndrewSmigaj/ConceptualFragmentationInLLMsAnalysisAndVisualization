"""
Extract activations for ALL GPT-2 tokens with progress bar and time estimates
"""

import torch
import numpy as np
from pathlib import Path
from transformers import GPT2Model, GPT2Tokenizer
import json
from datetime import datetime, timedelta
import gc
import os
try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    import subprocess
    subprocess.check_call(["pip", "install", "tqdm"])
    from tqdm import tqdm
import time

class AllTokenActivationExtractor:
    """Extract activations for the entire GPT-2 vocabulary with progress tracking"""
    
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
    
    def extract_chunk(self, start_idx: int, end_idx: int, chunk_pbar: tqdm) -> np.ndarray:
        """Extract activations for a chunk of tokens"""
        chunk_size = end_idx - start_idx
        chunk_activations = []
        
        # Create progress bar for tokens within chunk
        token_pbar = tqdm(range(start_idx, end_idx), 
                         desc="  Tokens", 
                         leave=False,
                         unit=" tokens")
        
        for token_id in token_pbar:
            activation = self.extract_single_token_activation(token_id)
            chunk_activations.append(activation)
            
            # Update chunk progress bar
            chunk_pbar.update(1)
        
        return np.array(chunk_activations)
    
    def extract_all_activations(self, output_dir: Path, resume_from: int = 0):
        """Extract activations for all tokens with comprehensive progress tracking"""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate total work
        total_tokens = self.vocab_size - resume_from
        n_chunks = (self.vocab_size - resume_from + self.chunk_size - 1) // self.chunk_size
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION PLAN")
        print(f"{'='*60}")
        print(f"Total tokens to process: {total_tokens:,}")
        print(f"Chunks to process: {n_chunks}")
        print(f"Estimated time: {self._estimate_total_time(total_tokens)}")
        print(f"{'='*60}\n")
        
        # Overall progress bar
        overall_pbar = tqdm(total=total_tokens, 
                           desc="Overall Progress", 
                           unit=" tokens",
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        if resume_from > 0:
            overall_pbar.update(resume_from)
            print(f"Resuming from token {resume_from}")
        
        # Track timing for better estimates
        chunk_times = []
        start_time = time.time()
        
        # Process in chunks
        for chunk_idx, chunk_start in enumerate(range(resume_from, self.vocab_size, self.chunk_size)):
            chunk_end = min(chunk_start + self.chunk_size, self.vocab_size)
            chunk_id = chunk_start // self.chunk_size
            
            # Check if chunk already exists
            chunk_file = output_dir / f"activations_chunk_{chunk_id:04d}.npy"
            if chunk_file.exists():
                print(f"\nChunk {chunk_id} already exists, skipping...")
                overall_pbar.update(chunk_end - chunk_start)
                continue
            
            print(f"\n{'='*60}")
            print(f"CHUNK {chunk_id}/{n_chunks-1}: Tokens {chunk_start:,}-{chunk_end:,}")
            print(f"{'='*60}")
            
            chunk_start_time = time.time()
            
            # Create progress bar for this chunk
            chunk_pbar = tqdm(total=chunk_end - chunk_start,
                            desc=f"Chunk {chunk_id}",
                            unit=" tokens",
                            leave=False)
            
            # Extract chunk
            chunk_activations = self.extract_chunk(chunk_start, chunk_end, chunk_pbar)
            
            # Save chunk
            np.save(chunk_file, chunk_activations)
            
            # Calculate timing
            chunk_time = time.time() - chunk_start_time
            chunk_times.append(chunk_time)
            tokens_per_sec = (chunk_end - chunk_start) / chunk_time
            
            # Update overall progress
            overall_pbar.update(chunk_end - chunk_start)
            
            # Print chunk statistics
            print(f"\n[OK] Chunk {chunk_id} complete!")
            print(f"  Time: {timedelta(seconds=int(chunk_time))}")
            print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")
            print(f"  File: {chunk_file.name}")
            
            # Update time estimate
            if len(chunk_times) > 0:
                avg_chunk_time = np.mean(chunk_times[-5:])  # Use last 5 chunks
                remaining_chunks = n_chunks - chunk_idx - 1
                eta_seconds = remaining_chunks * avg_chunk_time
                print(f"  ETA: {timedelta(seconds=int(eta_seconds))}")
            
            # Memory management
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()
            
            # Close chunk progress bar
            chunk_pbar.close()
        
        overall_pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE!")
        print(f"{'='*60}")
        print(f"Total time: {timedelta(seconds=int(total_time))}")
        print(f"Average speed: {total_tokens/total_time:.1f} tokens/sec")
        print(f"Output directory: {output_dir}")
        
        # Create metadata
        metadata = {
            'model_name': self.model_name,
            'vocab_size': self.vocab_size,
            'n_layers': 12,
            'hidden_dim': 768,
            'chunk_size': self.chunk_size,
            'n_chunks': (self.vocab_size + self.chunk_size - 1) // self.chunk_size,
            'extraction_date': datetime.now().isoformat(),
            'total_extraction_time': total_time,
            'average_tokens_per_second': total_tokens / total_time
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def _estimate_total_time(self, n_tokens: int) -> str:
        """Estimate total extraction time"""
        # Rough estimate based on hardware
        if self.device.type == 'cuda':
            tokens_per_sec = 100  # GPU estimate
        else:
            tokens_per_sec = 20   # CPU estimate
        
        total_seconds = n_tokens / tokens_per_sec
        return str(timedelta(seconds=int(total_seconds)))
    
    def verify_extraction(self, output_dir: Path):
        """Verify all chunks were extracted"""
        print("\nVerifying extraction...")
        
        with open(output_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        n_chunks = metadata['n_chunks']
        missing_chunks = []
        
        pbar = tqdm(range(n_chunks), desc="Checking chunks")
        for chunk_id in pbar:
            chunk_file = output_dir / f"activations_chunk_{chunk_id:04d}.npy"
            if not chunk_file.exists():
                missing_chunks.append(chunk_id)
        
        if missing_chunks:
            print(f"\n[WARNING] Missing chunks: {missing_chunks}")
            print(f"Run again to complete extraction")
            return False
        else:
            print(f"\n[OK] All {n_chunks} chunks verified!")
            return True


def main():
    """Extract all GPT-2 token activations with progress tracking"""
    
    output_dir = Path("experiments/gpt2/all_tokens/activations")
    
    print("\n" + "="*60)
    print("GPT-2 ALL TOKEN ACTIVATION EXTRACTION")
    print("="*60)
    
    # Check if we need to resume
    resume_from = 0
    if output_dir.exists():
        chunk_files = list(output_dir.glob("activations_chunk_*.npy"))
        if chunk_files:
            last_chunk = max(int(f.stem.split('_')[-1]) for f in chunk_files)
            resume_from = (last_chunk + 1) * 5000
            print(f"\n[Found] {len(chunk_files)} existing chunks")
            print(f"[Resume] Will resume from token {resume_from:,}")
    
    # Initialize extractor
    print("\n[Starting] Initializing extractor...")
    extractor = AllTokenActivationExtractor(chunk_size=5000)
    
    # Extract all activations
    metadata = extractor.extract_all_activations(output_dir, resume_from=resume_from)
    
    # Verify extraction
    extractor.verify_extraction(output_dir)
    
    print("\n[COMPLETE] Extraction pipeline complete!")


if __name__ == "__main__":
    main()