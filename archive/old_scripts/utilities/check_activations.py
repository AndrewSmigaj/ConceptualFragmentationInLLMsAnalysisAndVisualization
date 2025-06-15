import pickle
import sys

# Path to activations file
path = "D:/concept_fragmentation_results/baselines/heart/heart_baseline_seed0_20250520/layer_activations.pkl"

try:
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    print("All keys:", list(data.keys()))
    print("\nLayer keys:", [k for k in data.keys() if k.startswith('layer')])
    print("\nInput key present:", 'input' in data.keys())
    
    if 'input' in data:
        print("\nInput data structure:")
        print("  Type:", type(data['input']))
        if isinstance(data['input'], dict):
            print("  Keys:", list(data['input'].keys()))
            for split in ['train', 'test']:
                if split in data['input']:
                    print(f"  {split} shape:", data['input'][split][-1].shape if isinstance(data['input'][split], list) else data['input'][split].shape)

except Exception as e:
    print("Error:", str(e))
    raise 