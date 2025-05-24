#!/usr/bin/env python3
"""Check the structure of word metadata."""

import pickle
from pathlib import Path

metadata_file = Path("word_metadata.pkl")
with open(metadata_file, 'rb') as f:
    metadata = pickle.load(f)

print("Metadata keys:", metadata.keys())
print("\nTokens structure:")
tokens = metadata.get('tokens', [])
print(f"  Type: {type(tokens)}")
print(f"  Length: {len(tokens)}")
if tokens:
    print(f"  First item type: {type(tokens[0])}")
    print(f"  First few items: {tokens[:3]}")

print("\nSemantic subtypes structure:")
subtypes = metadata.get('semantic_subtypes', {})
print(f"  Type: {type(subtypes)}")
print(f"  Keys: {list(subtypes.keys())}")
if subtypes:
    first_key = list(subtypes.keys())[0]
    print(f"  First subtype '{first_key}' has {len(subtypes[first_key])} words")
    print(f"  Sample words: {subtypes[first_key][:5]}")