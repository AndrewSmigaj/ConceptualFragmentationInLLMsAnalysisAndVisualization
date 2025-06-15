# GPT-2 Semantic Subtypes - Window Analysis Report

## Key Findings

### Most Interesting Windows

1. **layers_0-2** (score: 5.849)
   - Unique paths: 8
   - Entropy: 2.92
   - Fragmentation: 2.00

2. **layers_1-3** (score: 5.072)
   - Unique paths: 8
   - Entropy: 2.54
   - Fragmentation: 2.00

3. **layers_2-4** (score: 3.317)
   - Unique paths: 7
   - Entropy: 1.66
   - Fragmentation: 2.00

### Representative Path Analysis

Words from each semantic subtype follow these paths:

**concrete_nouns**:
- ant: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- cow: `L0CL0C1 -> L1CL1C0 -> L2CL2C0 -> L3CL3C0 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- ice: `L0CL0C1 -> L1CL1C0 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`

**abstract_nouns**:
- age: `L0CL0C1 -> L1CL1C0 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- end: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- end: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`

**physical_adjectives**:
- aged: `L0CL0C1 -> L1CL1C0 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- fixed: `L0CL0C0 -> L1CL1C1 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- long: `L0CL0C0 -> L1CL1C1 -> L2CL2C0 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`

**manner_adverbs**:
- young: `L0CL0C0 -> L1CL1C0 -> L2CL2C0 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- actively: `L0CL0C0 -> L1CL1C1 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- fast: `L0CL0C0 -> L1CL1C0 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`

**emotive_adjectives**:
- adult: `L0CL0C0 -> L1CL1C1 -> L2CL2C0 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- free: `L0CL0C1 -> L1CL1C0 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- odd: `L0CL0C0 -> L1CL1C0 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`

**degree_adverbs**:
- about: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- fully: `L0CL0C0 -> L1CL1C0 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- more: `L0CL0C0 -> L1CL1C1 -> L2CL2C0 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`

**action_verbs**:
- answer: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- enter: `L0CL0C1 -> L1CL1C0 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- lift: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`

**stative_verbs**:
- able: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`
- express: `L0CL0C1 -> L1CL1C1 -> L2CL2C0 -> L3CL3C1 -> L4CL4C0 -> L5CL5C1 -> L6CL6C1 -> L7CL7C1 -> L8CL8C0 -> L9CL9C0 -> L10CL10C0 -> L11CL11C0`
- able: `L0CL0C1 -> L1CL1C1 -> L2CL2C1 -> L3CL3C0 -> L4CL4C1 -> L5CL5C0 -> L6CL6C0 -> L7CL7C0 -> L8CL8C1 -> L9CL9C1 -> L10CL10C1 -> L11CL11C1`

## Questions for Analysis

1. Why do layers 6-8 show the highest fragmentation?
2. What semantic transformations occur in the most interesting windows?
3. Do words from the same semantic subtype converge or diverge in their paths?
4. How do the cluster counts relate to the semantic organization at each layer?