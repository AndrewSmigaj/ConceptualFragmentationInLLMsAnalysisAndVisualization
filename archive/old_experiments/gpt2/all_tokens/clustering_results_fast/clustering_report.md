# GPT-2 All Token Fast Clustering Analysis

Analyzed 50,257 tokens
Layer analyzed: 11 (final layer)
K values tested: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

## Best Configuration
- Best k: 10
- Best silhouette score: 0.1426

## Clustering Results Summary

| k | Silhouette | Specialized | Punctuation | Numeric | Special |
|---|------------|-------------|-------------|---------|---------|
| 10 | 0.1426 | 2 | 0 | 1 | 0 |
| 20 | 0.1062 | 8 | 0 | 1 | 0 |
| 30 | 0.0875 | 11 | 1 | 1 | 0 |
| 40 | 0.0818 | 13 | 1 | 1 | 0 |
| 50 | 0.0773 | 17 | 1 | 1 | 0 |
| 60 | 0.0705 | 25 | 1 | 2 | 0 |
| 70 | 0.0672 | 30 | 2 | 2 | 0 |
| 80 | 0.0683 | 33 | 4 | 2 | 0 |
| 90 | 0.0640 | 41 | 4 | 3 | 0 |
| 100 | 0.0574 | 42 | 4 | 3 | 0 |

## Interpretation

The clustering analysis reveals how GPT-2 organizes its full vocabulary:

1. **Token Type Organization**: The presence of specialized clusters suggests GPT-2
   groups tokens by their linguistic function (punctuation, numbers, words).

2. **Subword Patterns**: With higher k values, we should see whether subwords
   cluster by morphological patterns (prefixes, suffixes, stems).

3. **Optimal Clustering**: The best k value balances between too few clusters
   (mixing different token types) and too many (overfitting to individual tokens).