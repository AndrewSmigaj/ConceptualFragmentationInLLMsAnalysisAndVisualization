"""Quick test of the 5k common words experiment"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.gpt2.shared.gpt2_token_validator import GPT2TokenValidator

# Test token validation
validator = GPT2TokenValidator()

# Test a few words
test_words = ["cat", "the", "running", "beautiful", "quickly"]
print("Testing token validation:")
for word in test_words:
    result = validator.validate_word(word)
    print(f"{word}: single_token={result.is_single_token}, token_count={result.token_count}")

# Test getting common words
print("\nTesting common word extraction (limited):")
from collections import Counter
from nltk.corpus import brown
import nltk

# Make sure Brown corpus is loaded
nltk.download('brown', quiet=True)

# Get a small sample
word_freq = Counter()
for i, word in enumerate(brown.words()):
    if i > 10000:  # Very small sample
        break
    word_lower = word.lower()
    if word_lower.isalpha() and len(word_lower) > 1:
        word_freq[word_lower] += 1

print(f"Found {len(word_freq)} unique words in sample")
print("Top 10 most common:")
for word, count in word_freq.most_common(10):
    print(f"  {word}: {count}")