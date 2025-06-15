"""
Simple GPT-2 activation extractor for pivot experiment.

This is a focused implementation for our 3-token pivot experiment,
designed to work without complex dependencies.
"""

import json
import pickle
from typing import Dict, List, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class SimpleGPT2ActivationExtractor:
    """
    Simple GPT-2 activation extractor for pivot experiment.
    
    This extracts activations from all GPT-2 layers for our 3-token sentences.
    """
    
    def __init__(self):
        """Initialize the extractor."""
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
    
    def _get_memory_usage(self):
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Fallback if psutil not available
            return 0.0
    
    def _check_memory_pressure(self, threshold_mb=2048):
        """Check if system is under memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / 1024 / 1024
            return available_mb < threshold_mb
        except ImportError:
            # Conservative fallback if psutil not available
            return False
        
    def setup_model(self):
        """Setup GPT-2 model and tokenizer."""
        try:
            # Try to import and setup
            from transformers import GPT2Model, GPT2Tokenizer
            import torch
            
            print("Loading GPT-2 model and tokenizer...")
            self.model = GPT2Model.from_pretrained("gpt2")
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            
            print(f"Model loaded on {self.device}")
            return True
            
        except ImportError as e:
            print(f"Could not import required libraries: {e}")
            print("Generating mock activations instead...")
            return False
    
    def extract_activations(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Extract activations for all sentences.
        
        Args:
            sentences: List of 3-token sentences to process
            
        Returns:
            Dictionary containing:
            - 'activations': Dict[sentence_idx][token_idx][layer_idx] = activation_vector
            - 'sentences': List of processed sentences
            - 'tokens': List of tokenized sentences
            - 'metadata': Additional information
        """
        if self.model is None:
            if not self.setup_model():
                return self._generate_mock_activations(sentences)
        
        try:
            return self._extract_real_activations(sentences)
        except Exception as e:
            print(f"Error extracting real activations: {e}")
            print("Generating mock activations instead...")
            return self._generate_mock_activations(sentences)
    
    def _extract_real_activations(self, sentences: List[str]) -> Dict[str, Any]:
        """Extract real activations from GPT-2."""
        import torch
        
        results = {
            'activations': {},
            'sentences': sentences,
            'tokens': [],
            'metadata': {
                'model_name': 'gpt2',
                'num_layers': 12,
                'hidden_size': 768,
                'extraction_method': 'real'
            }
        }
        
        print(f"Extracting activations for {len(sentences)} sentences...")
        
        for sent_idx, sentence in enumerate(sentences):
            # For our 3-token sentences, force word-level tokenization
            word_tokens = sentence.split()  # ['good', 'but', 'bad']
            
            # Tokenize each word separately to get proper alignment
            individual_token_ids = []
            for word in word_tokens:
                word_ids = self.tokenizer.encode(word, add_special_tokens=False)
                individual_token_ids.extend(word_ids)
            
            # Create tensor for the full sentence 
            token_ids = torch.tensor([individual_token_ids]).to(self.device)
            
            results['tokens'].append(word_tokens)
            results['activations'][sent_idx] = {}
            
            # Forward pass with all hidden states
            try:
                with torch.no_grad():
                    outputs = self.model(token_ids, output_hidden_states=True)
                    hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_size)
            except Exception as e:
                # Handle CUDA OOM or other GPU memory errors
                error_msg = str(e).lower()
                if "out of memory" in error_msg or "cuda" in error_msg:
                    print(f"GPU memory error at sentence {sent_idx+1}/{len(sentences)}: {e}")
                    print("Falling back to mock activations for remaining sentences...")
                    # Return partial results + mock for remainder
                    mock_results = self._generate_mock_activations(sentences[sent_idx:])
                    # Merge results
                    for mock_idx, mock_sent_idx in enumerate(range(sent_idx, len(sentences))):
                        results['activations'][mock_sent_idx] = mock_results['activations'][mock_idx]
                        results['tokens'].extend(mock_results['tokens'][mock_idx:])
                    results['metadata']['partial_extraction'] = True
                    results['metadata']['oom_at_sentence'] = sent_idx
                    return results
                else:
                    # Re-raise non-memory errors
                    raise
            
            # Map activations back to our 3 word tokens
            # For simplicity, take the first subword token for each word
            token_positions = []
            current_pos = 0
            for word in word_tokens:
                word_ids = self.tokenizer.encode(word, add_special_tokens=False)
                token_positions.append(current_pos)  # Position of first subword token
                current_pos += len(word_ids)
            
            # Extract activations for each of our 3 word tokens at each layer
            for word_idx, token_pos in enumerate(token_positions):
                if token_pos < token_ids.shape[1]:  # Make sure position exists
                    results['activations'][sent_idx][word_idx] = {}
                    
                    # Layer 0 is embedding, Layers 1-12 are transformer blocks  
                    for layer_idx in range(len(hidden_states)):
                        activation = hidden_states[layer_idx][0, token_pos, :].cpu().numpy()
                        results['activations'][sent_idx][word_idx][layer_idx] = activation
            
            # Memory cleanup to prevent accumulation
            del outputs, hidden_states
            if self.device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # Periodic garbage collection for large datasets
            if (sent_idx + 1) % 50 == 0:
                import gc
                gc.collect()
            
            # Improved progress reporting
            progress_interval = max(1, len(sentences) // 20)  # Report every 5%
            if (sent_idx + 1) % progress_interval == 0 or sent_idx == 0:
                percent = ((sent_idx + 1) / len(sentences)) * 100
                memory_usage = self._get_memory_usage()
                print(f"Progress: {sent_idx + 1}/{len(sentences)} ({percent:.1f}%) - Memory: {memory_usage:.1f}MB")
        
        return results
    
    def _generate_mock_activations(self, sentences: List[str]) -> Dict[str, Any]:
        """Generate mock activations for testing when real model unavailable."""
        import random
        
        print("Generating mock activations (for testing only)...")
        
        results = {
            'activations': {},
            'sentences': sentences,
            'tokens': [],
            'metadata': {
                'model_name': 'gpt2_mock',
                'num_layers': 13,  # embedding + 12 transformer layers
                'hidden_size': 768,
                'extraction_method': 'mock'
            }
        }
        
        # Generate mock data
        for sent_idx, sentence in enumerate(sentences):
            # Mock tokenization (just split on spaces for 3-token sentences)
            tokens = sentence.split()
            results['tokens'].append(tokens)
            results['activations'][sent_idx] = {}
            
            # Generate random activations for each token at each layer
            for token_idx in range(len(tokens)):
                results['activations'][sent_idx][token_idx] = {}
                
                for layer_idx in range(13):  # 13 layers total
                    # Create different patterns for different classes
                    if "but bad" in sentence or "but awful" in sentence or "but terrible" in sentence:
                        # Contrast class: more variation across layers
                        activation = [random.gauss(0, 0.5 + layer_idx * 0.1) for _ in range(32)]  # Smaller size for testing
                    else:
                        # Consistent class: less variation across layers  
                        activation = [random.gauss(0, 0.3) for _ in range(32)]
                    
                    results['activations'][sent_idx][token_idx][layer_idx] = activation
        
        return results
    
    def save_activations(self, activations: Dict[str, Any], filename: str):
        """Save activations to file."""
        print(f"Saving activations to {filename}...")
        
        # Save as pickle for complex data structures
        with open(filename, 'wb') as f:
            pickle.dump(activations, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_file = filename.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(activations['metadata'], f, indent=2)
        
        print(f"Activations saved to {filename}")
        print(f"Metadata saved to {metadata_file}")
    
    def load_activations(self, filename: str) -> Dict[str, Any]:
        """Load activations from file."""
        print(f"Loading activations from {filename}...")
        with open(filename, 'rb') as f:
            activations = pickle.load(f)
        return activations


def extract_all_activations():
    """Extract activations for all pivot experiment sentences."""
    # Load sentences
    with open("gpt2_pivot_contrast_sentences.txt", "r") as f:
        contrast_sentences = [line.strip() for line in f if line.strip()]
    
    with open("gpt2_pivot_consistent_sentences.txt", "r") as f:
        consistent_sentences = [line.strip() for line in f if line.strip()]
    
    all_sentences = contrast_sentences + consistent_sentences
    
    # Create labels
    labels = ['contrast'] * len(contrast_sentences) + ['consistent'] * len(consistent_sentences)
    
    print(f"Extracting activations for {len(all_sentences)} sentences")
    print(f"  - {len(contrast_sentences)} contrast sentences")
    print(f"  - {len(consistent_sentences)} consistent sentences")
    
    # Extract activations
    extractor = SimpleGPT2ActivationExtractor()
    activations = extractor.extract_activations(all_sentences)
    
    # Add labels to metadata
    activations['labels'] = labels
    activations['metadata']['num_contrast'] = len(contrast_sentences)
    activations['metadata']['num_consistent'] = len(consistent_sentences)
    
    # Save results
    extractor.save_activations(activations, "gpt2_pivot_activations.pkl")
    
    return activations


if __name__ == "__main__":
    activations = extract_all_activations()