# GPT-2 Semantic Subtypes Experiment Results (Enhanced)

## Experiment Overview
- 774 single-token words across 8 semantic subtypes
- 13 GPT-2 layers (embedding + 12 transformer)
- Two clustering methods: K-means and ETS

## K-means Clustering Results
- Total unique paths: 24
- Layer-wise silhouette scores:
  - layer_0: 0.012 (2 clusters)
  - layer_1: 0.071 (2 clusters)
  - layer_10: 0.325 (2 clusters)
  - layer_11: 0.324 (2 clusters)
  - layer_2: 0.102 (2 clusters)
  - layer_3: 0.174 (2 clusters)
  - layer_4: 0.258 (2 clusters)
  - layer_5: 0.289 (2 clusters)
  - layer_6: 0.306 (2 clusters)
  - layer_7: 0.316 (2 clusters)
  - layer_8: 0.320 (2 clusters)
  - layer_9: 0.324 (2 clusters)

### Cluster Contents (Key Layers)

#### Layer 0

**Cluster L0C0** (342 sentences):
orange, success, right, wrong, good, evil, meaning, reason, big, small, large, tiny, huge, mini, thin, tall, short, long, deep, high, low, broad, lean, heavy, solid, great, grand, mega, super, micro ... and 312 more

**Cluster L0C1** (432 sentences):
cat, dog, bird, fish, horse, cow, mouse, rat, bear, wolf, fox, monkey, frog, bee, ant, fly, worm, chair, table, bed, book, pen, paper, cup, plate, bowl, fork, knife, glass, box ... and 402 more

#### Layer 6

**Cluster L6C0** (431 sentences):
dog, bird, fish, bear, wolf, fox, frog, bee, ant, fly, worm, table, bed, paper, cup, plate, bowl, fork, knife, glass, box, bag, key, lock, door, wall, floor, stairs, clock, phone ... and 401 more

**Cluster L6C1** (343 sentences):
cat, horse, cow, mouse, rat, monkey, chair, book, pen, window, car, train, grass, sand, sun, star, cloud, egg, orange, cookie, hate, shock, truth, success, image, concept, fact, news, future, science ... and 313 more

#### Layer 11

**Cluster L11C0** (344 sentences):
cat, horse, cow, mouse, rat, monkey, chair, book, pen, window, car, train, grass, sand, sun, star, cloud, egg, orange, cookie, hate, shock, truth, success, image, concept, fact, news, story, future ... and 314 more

**Cluster L11C1** (430 sentences):
dog, bird, fish, bear, wolf, fox, frog, bee, ant, fly, worm, table, bed, paper, cup, plate, bowl, fork, knife, glass, box, bag, key, lock, door, wall, floor, stairs, clock, phone ... and 400 more

## Semantic Organization Analysis
### Within-Subtype Coherence
- abstract_nouns: 0.859 coherence (13 paths for 85 words)
- action_verbs: 0.900 coherence (12 paths for 110 words)
- concrete_nouns: 0.910 coherence (9 paths for 89 words)
- degree_adverbs: 0.921 coherence (14 paths for 164 words)
- emotive_adjectives: 0.810 coherence (9 paths for 42 words)
- manner_adverbs: 0.867 coherence (15 paths for 105 words)
- physical_adjectives: 0.792 coherence (12 paths for 53 words)
- stative_verbs: 0.881 coherence (16 paths for 126 words)

### Example Archetypal Paths by Subtype (with words)

**abstract_nouns** (top 3 paths):

- Path: L0C1 -> L1C0 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 29 words
  Words: dog, bird, fish, fox, frog, fly, worm, table, paper, plate ... and 142 more

- Path: L0C1 -> L1C1 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 22 words
  Words: bear, wolf, bee, ant, bed, bowl, knife, bag, door, wall ... and 99 more

- Path: L0C1 -> L1C1 -> L2C0 -> L3C1 -> L4C0 -> L5C1 -> L6C1 -> L7C1 -> L8C0 -> L9C0 -> L10C0 -> L11C0
  Count: 14 words
  Words: cat, monkey, book, window, train, sand, sun, egg, shock, truth ... and 65 more

**action_verbs** (top 3 paths):

- Path: L0C1 -> L1C0 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 28 words
  Words: dog, bird, fish, fox, frog, fly, worm, table, paper, plate ... and 142 more

- Path: L0C1 -> L1C1 -> L2C0 -> L3C1 -> L4C0 -> L5C1 -> L6C1 -> L7C1 -> L8C0 -> L9C0 -> L10C0 -> L11C0
  Count: 25 words
  Words: cat, monkey, book, window, train, sand, sun, egg, shock, truth ... and 65 more

- Path: L0C1 -> L1C1 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 24 words
  Words: bear, wolf, bee, ant, bed, bowl, knife, bag, door, wall ... and 99 more

**concrete_nouns** (top 3 paths):

- Path: L0C1 -> L1C0 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 37 words
  Words: dog, bird, fish, fox, frog, fly, worm, table, paper, plate ... and 142 more

- Path: L0C1 -> L1C1 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 28 words
  Words: bear, wolf, bee, ant, bed, bowl, knife, bag, door, wall ... and 99 more

- Path: L0C1 -> L1C0 -> L2C0 -> L3C1 -> L4C0 -> L5C1 -> L6C1 -> L7C1 -> L8C0 -> L9C0 -> L10C0 -> L11C0
  Count: 7 words
  Words: mouse, rat, chair, pen, grass, cloud, cookie, custom, fashion, class ... and 24 more

**degree_adverbs** (top 3 paths):

- Path: L0C0 -> L1C1 -> L2C0 -> L3C1 -> L4C0 -> L5C1 -> L6C1 -> L7C1 -> L8C0 -> L9C0 -> L10C0 -> L11C0
  Count: 59 words
  Words: right, wrong, meaning, mini, long, deep, great, straight, packed, blue ... and 91 more

- Path: L0C0 -> L1C0 -> L2C0 -> L3C1 -> L4C0 -> L5C1 -> L6C1 -> L7C1 -> L8C0 -> L9C0 -> L10C0 -> L11C0
  Count: 27 words
  Words: success, big, tall, short, broad, lean, heavy, grand, super, micro ... and 52 more

- Path: L0C0 -> L1C0 -> L2C1 -> L3C0 -> L4C1 -> L5C0 -> L6C0 -> L7C0 -> L8C1 -> L9C1 -> L10C1 -> L11C1
  Count: 19 words
  Words: small, large, tiny, high, low, solid, mega, flat, bent, sharp ... and 63 more

## ETS Clustering Results
- Total unique paths: 24

### ETS Cluster Contents (Layer 6 example)

**Cluster L6C0** (431 sentences):
dog, bird, fish, bear, wolf, fox, frog, bee, ant, fly, worm, table, bed, paper, cup, plate, bowl, fork, knife, glass ... and 411 more

**Cluster L6C1** (343 sentences):
cat, horse, cow, mouse, rat, monkey, chair, book, pen, window, car, train, grass, sand, sun, star, cloud, egg, orange, cookie ... and 323 more

## Key Questions for Analysis

### Cluster Interpretation
1. Looking at the sentences in each cluster, what semantic or grammatical themes emerge?
2. Can you suggest descriptive labels for the clusters in layers 0, 6, and 12?

### Archetypal Path Analysis
3. What do the archetypal paths represent semantically?
4. Why might certain words follow the same path through the layers?
5. Do words within the same subtype follow similar paths?

### Layer Evolution
6. How does the clustering evolve from early to late layers?
7. Do early layers capture more syntactic features while later layers capture semantics?

### Semantic Organization Insights
8. How does GPT-2 organize semantic knowledge across layers?
9. Do grammatical categories (nouns, verbs, etc.) cluster together?
10. What do the coherence scores reveal about GPT-2's internal organization?
