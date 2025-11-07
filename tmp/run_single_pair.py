import os
import sys

# Ensure repository root is on sys.path so we can import nlp_research
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from nlp_research import logical_score, combined_preference_score

A_PATH = os.path.join('tmp', 'test_a.txt')
B_PATH = os.path.join('tmp', 'test_b.txt')

with open(A_PATH, 'r', encoding='utf-8') as fa:
    text_a = fa.read()
with open(B_PATH, 'r', encoding='utf-8') as fb:
    text_b = fb.read()

score_a = logical_score(text_a)
score_b = logical_score(text_b)
print(f"Logical Score A: {score_a:.4f}")
print(f"Logical Score B: {score_b:.4f}")
print("-")
hybrid_a = combined_preference_score(text_a, human_weight=0.5)
hybrid_b = combined_preference_score(text_b, human_weight=0.5)
print(f"Hybrid Score A (human_weight=0.5): {hybrid_a:.4f}")
print(f"Hybrid Score B (human_weight=0.5): {hybrid_b:.4f}")
if hybrid_a > hybrid_b:
    print("Winner (hybrid): A")
elif hybrid_b > hybrid_a:
    print("Winner (hybrid): B")
else:
    print("Winner (hybrid): tie")
