# gzip-mean

Classify text by comparing average compression score per class, rather than pre-calculating normalized compression distances and using KNN.

Assumption: on average, a compression score of positive target text combined with positive example from the train set will be higher than score when it's combined with negative example. Just comparing average scores of positive and negative examples when they are combined with new text should be enough to classify new text.

For example:

- `good film + good movie` will have higher compression score than `good film + bad movie`

Score formula:
```python
def score(x1: str, x2: str) -> float:
    return 1 - size(x1 + ' ' + x2) / (size(x1) + size(x2))
```

Size formula:
```python
def size(x: str) -> int:
    return len(gzip.compress(x.encode('utf-8')))
```

## Results
Accuracies fluctuate between **73-78%** on the same dataset based on random selection of samples in positive and negative classes (`500` per class):

```text
Accuracy per run: 0.7696, 0.7576, 0.7376, 0.7381, 0.7766
Accuracy (mean): 0.7559
```

It makes sense to experiment with larger number of samples per class.

## Reproduce

```bash
git clone https://github.com/zemlyansky/gzip-mean.git
cd gzip-mean
wget https://github.com/Sentdex/Simple-kNN-Gzip/raw/main/sentiment-dataset-10000.pickle
for i in {1..5}; do python gzip-mean.py; done
```

## References

- Original Gzip-KNN method: [paper](https://aclanthology.org/2023.findings-acl.426.pdf), [code](https://github.com/bazingagin/npc_gzip)
- Simple-KNN-Gzip [code](https://github.com/Sentdex/Simple-kNN-Gzip), [video](https://www.youtube.com/watch?v=jkdWzvMOPuo)