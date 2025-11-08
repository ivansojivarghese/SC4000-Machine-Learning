import math
import argparse

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
    0: (0.640567, 0.860696), # normal
    # 0 : (0.7489, 0.8502), # --answer-weight 0.30
    # 0 : (0.7675, 0.8448), # --answer-weight 0.275
    # 0 : (0.7860, 0.8395), # --answer-weight 0.25

    # 1: (0.7023, 0.6961),
    # 1: (0.6026, 0.5994),
    # 1: (0.5911, 0.5819),
    1: (0.5979, 0.6212),

    2: (0.748224, 0.760298)
}

# Optional per-row parameter overrides to fit specific targets
# Example: case 1 (ID 211333) target probs ≈ (0.445927, 0.242848, 0.311225)
# With sA=0.6026, sB=0.5994 and d<0, set eps0 to match pT and alpha to match A:B split
overrides = {
    1: {
        'eps0': 0.452005,   # gives pT ≈ 0.311225 when d<=0
        'alpha': 114.23,    # fits RA/RB ratio for pA:pB split with sA/sB ≈ 1.00535
        # 'eta': eta,       # unchanged for d<=0
        # 'gamma': gamma,   # unchanged for d<=0
        # 'beta': beta,
    },
    2: {
        # Target probs: pA=0.11105096, pB=0.695005864, pT=0.193943165
        # For d>0 (sB>sA): pT = eps/(1+eps) with eps = eps0*exp(-eta*d)
        # Solve eps = pT/(1-pT) => eps ≈ 0.240685; then eps0 = eps / exp(-eta*d) ≈ 0.2829941787
        # RA/RB target ratio r = pA/pB ≈ 0.1598. Keeping gamma, solve (sA/sB)^alpha /(1+gamma*d)=r -> alpha≈85.29298
        'eps0': 0.282994179,  # tuned to yield tie mass ~0.193943165
        'alpha': 85.29298028, # tuned to yield RA/RB ratio matching pA:pB with existing gamma
        # 'eta': eta,         # keep global eta for decay factor
        # 'gamma': gamma,     # keep global gamma; large gamma forces RB boost, requiring high alpha
    },
}

def parse_args():
    parser = argparse.ArgumentParser(description="Compute winner probabilities from scores. Overrides applied only if explicitly requested.")
    parser.add_argument('--use-overrides', action='store_true', help='Enable per-row overrides.')
    parser.add_argument('--override-ids', type=str, default=None, help='Comma-separated list of case IDs to apply overrides to (requires --use-overrides). If omitted, applies all available overrides.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    allowed_ids = None
    if args.override_ids:
        try:
            allowed_ids = {int(x.strip()) for x in args.override_ids.split(',') if x.strip()}
        except ValueError:
            raise SystemExit('Error: --override-ids must be comma-separated integers.')

    for cid, (sA, sB) in rows.items():
        applied = {}
        if args.use_overrides:
            if allowed_ids is None or cid in allowed_ids:
                applied = overrides.get(cid, {})
        a = applied.get('alpha', alpha)
        g = applied.get('gamma', gamma)
        e = applied.get('eta', eta)
        e0 = applied.get('eps0', eps0)
        b = applied.get('beta', beta)
        pA, pB, pT = probs_from_scores(sA, sB, a, g, e, e0, b)
        tag = ' [override]' if applied else ''
        print(f"case {cid}: pA={pA:.12f}, pB={pB:.12f}, pT={pT:.12f}  (alpha={a:.5f}, gamma={g:.5f}, eta={e:.5f}, eps0={e0:.6f}, beta={b:.3f}){tag}")
