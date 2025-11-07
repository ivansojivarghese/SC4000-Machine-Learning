import math

# ---- parameters (the tuned values found above) ----
alpha = 19.831675854828404
gamma = 49.50181595059537
eta   = 13.438806629499513
eps0  = 0.2512371308916018
beta  = 0.0   # optional suppression of A when B leads

def probs_from_scores(sA, sB, alpha, gamma, eta, eps0, beta=0.0):
    d = sB - sA
    RA = (sA ** alpha) * (1 - beta * max(0.0, d))
    RB = (sB ** alpha) * (1 + gamma * max(0.0, d))
    eps = eps0 * math.exp(-eta * max(0.0, d))
    RT = eps * (RA + RB)
    S = RA + RB + RT
    return (RA/S, RB/S, RT/S)

# your rows:
rows = {
    0: (0.640567, 0.860696),
    1: (0.7023, 0.6961),
    # 1: (0.6026, 0.5994),
    # 1: (0.5911, 0.5819),
    2: (0.748224, 0.760298),
}

for cid, (sA, sB) in rows.items():
    pA,pB,pT = probs_from_scores(sA, sB, alpha, gamma, eta, eps0, beta)
    print(f"case {cid}: pA={pA:.12f}, pB={pB:.12f}, pT={pT:.12f}")
