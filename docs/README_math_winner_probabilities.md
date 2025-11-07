# Winner Probability Formulation for Pairwise Model Comparison

This README explains the mathematical reasoning behind the probability mapping used to convert two task scores (Model A and Model B) into a normalized triplet of probabilities:

- `winner_model_a` (pA): probability Model A is better
- `winner_model_b` (pB): probability Model B is better
- `winner_tie` (pT): probability the models are effectively tied

The mapping intentionally **amplifies small score differences**, **boosts the leading model**, **suppresses ties when a leader emerges**, and **normalizes** the probability mass.

---
## 1. Inputs
Let
- sA = score_a_task
- sB = score_b_task
- d  = sB - sA (score difference)

Interpretation:
- d > 0 → B leads
- d < 0 → A leads
- d = 0 → tie baseline

---
## 2. Raw Strengths
We first transform the raw scores into *strengths* RA and RB:

\[
R_A = s_A^{\alpha} \cdot (1 - \beta \cdot \max(0, d))
\]
\[
R_B = s_B^{\alpha} \cdot (1 + \gamma \cdot \max(0, d))
\]

Where parameters control the shaping:
- **α (alpha)**: sharpness exponent (e.g. ≈ 19.83) magnifies small absolute differences in sA vs sB.
- **γ (gamma)**: directional boost applied *only* when B is ahead (d > 0); scales RB upward.
- **β (beta)**: optional suppression factor applied to RA when B leads; often set to 0 for neutrality.

### Why exponentiation?
Raising scores to a high power (α ≫ 1) stretches the scale: a tiny lead (e.g. 0.01) becomes a much larger ratio difference in RA vs RB. This provides a decisive preference even for modest raw score gaps.

### Directional boost (γ)
The multiplicative term `(1 + γ·max(0, d))` compounds RB when B leads. If d is small but positive, RB experiences an *extra* amplification, helping avoid ambiguous near-ties.

### Optional suppression (β)
Symmetric to γ, β can reduce RA when B leads to further emphasize the winner. In tuned runs β = 0, leaving RA untouched.

---
## 3. Tie Mass Construction
A raw tie strength RT is allocated as a fraction of total model strength:

\[
\varepsilon(d) = \varepsilon_0 \cdot e^{-\eta \cdot \max(0, d)}
\]
\[
R_T = \varepsilon(d) \cdot (R_A + R_B)
\]

Parameters:
- **ε₀ (eps0)**: baseline fraction reserved for tie when scores are equal.
- **η (eta)**: decay rate; as the leader’s advantage grows (d increases), the tie fraction shrinks exponentially.

### Intuition
- At d = 0: \( \varepsilon(d) = \varepsilon_0 \) → tie receives up to ε₀ of the total mass.
- As d grows: \( e^{-\eta d} \) rapidly → 0, stripping probability away from the tie and reallocating it to the winner.

This preserves a realistic interpretation: strong leads produce low tie probability; near-equal scores maintain non-trivial tie probability.

---
## 4. Normalization
Aggregate raw strengths:
\[
S = R_A + R_B + R_T
\]
Convert to probabilities:
\[
p_A = \frac{R_A}{S}, \quad p_B = \frac{R_B}{S}, \quad p_T = \frac{R_T}{S}
\]
This guarantees:
\[
p_A + p_B + p_T = 1
\]

Normalization preserves relative magnitudes while ensuring valid probability outputs.

---
## 5. Parameter Roles
| Parameter | Role |
|-----------|------|
| α (alpha) | Sharpness exponent; increases sensitivity to small score differences |
| γ (gamma) | Boost factor for the *leading* model (only when d > 0 or, symmetrically, could adapt for d < 0) |
| β (beta)  | Optional suppression for the trailing model (0 keeps neutrality) |
| ε₀ (eps0) | Base tie fraction at exact or near ties |
| η (eta)   | Tie decay aggressiveness with growing lead |

---
## 6. Conceptual Flow
```
Task Scores: sA, sB
        |
        v
Raw strengths: RA, RB (power + directional adjustments)
        |
        v
Tie fraction: RT = eps(d) * (RA + RB)
        |
        v
Normalize: pA, pB, pT
        |
        v
Final Probabilities: [winner_model_a, winner_model_b, winner_tie]
```

---
## 7. Example
Given tuned parameters:
```
alpha = 19.831675854828404
gamma = 49.50181595059537
eta   = 13.438806629499513
eps0  = 0.2512371308916018
beta  = 0.0
```
Case example:
```
sA = 0.640567
sB = 0.860696
d  = 0.220129
```
Result (illustrative):
```
pA ≈ 0.0024
pB ≈ 0.9870
pT ≈ 0.0106
```
Interpretation:
- B’s small-to-moderate lead becomes decisive.
- Tie is strongly suppressed due to exponential decay with d.
- A’s probability collapses under the high-α magnification and γ boost for B.

---
## 8. Behavior Characteristics
1. **Winner Amplification:** High α + γ ensures the leading model’s probability dominates quickly.
2. **Graceful Tie Handling:** Near ties (d ≈ 0) still allocate meaningful mass to pT via ε₀.
3. **Fast Tie Decay:** As d grows, `exp(-η·d)` removes tie mass rapidly.
4. **Parameter Tunability:** Adjust α for sensitivity, γ for aggressiveness, ε₀ for tie baseline, η for decay speed.
5. **Symmetry Extension:** A symmetric variant could apply the γ boost when A leads (d < 0) and suppression to B; current form implicitly favors forward direction (B-leading scenario). Add:
   ```python
   RA = (sA ** alpha) * (1 + gamma * max(0.0, -d))
   RB = (sB ** alpha) * (1 + gamma * max(0.0, d))
   ```
   for full symmetry.

---
## 9. Implementation Reference
Minimal Python implementation (see `math_process.py`):
```python
import math

def probs_from_scores(sA, sB, alpha, gamma, eta, eps0, beta=0.0):
    d = sB - sA
    RA = (sA ** alpha) * (1 - beta * max(0.0, d))
    RB = (sB ** alpha) * (1 + gamma * max(0.0, d))
    eps = eps0 * math.exp(-eta * max(0.0, d))
    RT = eps * (RA + RB)
    S = RA + RB + RT
    return (RA / S, RB / S, RT / S)
```

---
## 10. Extension Ideas
- **Adaptive γ/η:** Fit γ and η per domain or per score range for better calibration.
- **Tie Floor:** Impose a minimum tie probability pT_min to avoid absolute certainty declarations.
- **Confidence Score:** Add `confidence = 1 - pT` as a separate metric.
- **Regularization:** Use log-scores or temperature scaling to temper extreme ratios.

---
## 11. Summary
This probability formulation:
- Converts raw pairwise scores into interpretable winner/tie probabilities.
- Is simple yet expressive—five parameters guide sharpness, directional bias, and tie handling.
- Enables highly decisive outputs when desirable while remaining tunable for softer decisions.

It provides a robust backbone for ranking, ensembling, or automated arbitration systems where nuanced score differences must map to actionable probabilistic judgments.
