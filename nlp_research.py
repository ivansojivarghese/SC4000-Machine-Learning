# Install required libraries first:
# pip install sentence-transformers transformers torch nltk

import nltk
import os
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer
try:
    import textstat  # readability proxy
except Exception:  # optional dependency
    textstat = None
import re
import string

# Runtime-configurable human preference options (set from CLI in main)
_HPARAMS = {
    'enable_concise_sweetspot': False,
    'concise_target': 80,
    'concise_lower': 40,
    'concise_upper': 140,
    'concise_min_score': 0.6,
    'enable_verbosity_penalty': False,
    'verbosity_max_breaks': 5,     # allowed newline breaks before penalty
    'verbosity_slope': 0.05,       # penalty increase per extra break
    'verbosity_max_penalty': 0.2,  # cap for penalty
    'verbosity_weight': 0.5,       # how much to apply penalty in human score
    'enable_mediator_quality': False,
    'mediator_weight': 0.0,        # 0..1 mixing weight into human score
}

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# -------------------------------
# 1. Load models
# -------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Sentence embeddings for semantic consistency
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
try:
    embed_model.to(device)
except Exception:
    pass
# Reuse the same model for style similarity by default
style_model = embed_model

# NLI model for contradiction detection
nli_model_name = 'roberta-large-mnli'
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name)
try:
    nli_model.to(device)
    nli_model.eval()
except Exception:
    pass

# -------------------------------
# 2. Utility functions
# -------------------------------
def semantic_consistency_score(text):
    """Score based on semantic similarity between consecutive sentences."""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 1.0  # Single sentence is trivially consistent
    # Encode on the same device as the embedding model if possible
    try:
        embeddings = embed_model.encode(sentences, convert_to_tensor=True, device=str(device))
    except TypeError:
        embeddings = embed_model.encode(sentences, convert_to_tensor=True)
    sim_scores = []
    for i in range(len(sentences)-1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        sim_scores.append(sim)
    return sum(sim_scores) / len(sim_scores)

def contradiction_ratio(text):
    """Estimate contradictions between sentence pairs using NLI."""
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return 0.0  # No contradictions possible
    contradictions = 0
    total_pairs = 0
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            premise, hypothesis = sentences[i], sentences[j]
            inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            # Move to device for acceleration if available
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass
            with torch.no_grad():
                logits = nli_model(**inputs).logits
            # Label mapping for MNLI: 0=entailment, 1=neutral, 2=contradiction
            pred = torch.argmax(logits, dim=1).item()
            if pred == 2:
                contradictions += 1
            total_pairs += 1
    return contradictions / total_pairs if total_pairs > 0 else 0.0

def logical_score(text, alpha=0.6, beta=0.4):
    """Combine semantic consistency and contradiction detection."""
    sem_score = semantic_consistency_score(text)
    contra = contradiction_ratio(text)
    return alpha * sem_score + beta * (1 - contra)

# ---------- Human-preference and hybrid scoring ----------
def readability_score(text: str) -> float:
    """Flesch Reading Ease scaled to ~0-1. If textstat missing, return neutral 0.5."""
    if textstat is None:
        return 0.5
    try:
        val = textstat.flesch_reading_ease(text)
        # Typical range ~(-∞,100+); clamp and scale
        val = max(0.0, min(100.0, float(val)))
        return val / 100.0
    except Exception:
        return 0.5

def conciseness_score(text: str, max_len: int = 300) -> float:
    """Simple length penalty: 1.0 when <= max_len tokens, decays otherwise."""
    try:
        n = max(1, len(text.split()))
        return min(1.0, max_len / float(n))
    except Exception:
        return 0.5

def conciseness_sweet_spot_score(text: str,
                                 target_words: int = 80,
                                 lower: int = 40,
                                 upper: int = 140,
                                 min_score: float = 0.6) -> float:
    """Triangular sweet-spot scoring around target_words.
    - Score peaks at 1.0 near target_words
    - Linearly decreases to min_score at lower and upper bounds
    - Clamped to [min_score, 1.0]
    """
    try:
        n = max(1, len(text.split()))
        if n == target_words:
            return 1.0
        if n <= lower or n >= upper:
            return float(min_score)
        # piecewise linear between [lower, target] and [target, upper]
        if n < target_words:
            frac = (n - lower) / max(1, (target_words - lower))
        else:
            frac = (upper - n) / max(1, (upper - target_words))
        return float(min_score) + (1.0 - float(min_score)) * max(0.0, min(1.0, frac))
    except Exception:
        return 0.8

def _count_breaks(text: str) -> int:
    # Count single newlines and double-newline paragraphs
    try:
        return text.count('\n')
    except Exception:
        return 0

def verbosity_penalty(text: str,
                      max_breaks: int = 5,
                      slope: float = 0.05,
                      max_penalty: float = 0.2) -> float:
    """Penalty based on excessive line breaks.
    Returns a value in [0, max_penalty]. Higher means more verbose by formatting.
    """
    try:
        br = _count_breaks(text)
        extra = max(0, br - int(max_breaks))
        return min(float(max_penalty), max(0.0, extra * float(slope)))
    except Exception:
        return 0.0

def mediator_quality_score(text: str) -> float:
    """Heuristic for mediation quality: acknowledges both sides and proposes a compromise.
    Looks for cues like 'both', 'on one hand', 'on the other hand', 'compromise', 'middle ground',
    and explicit mapping 'sex'~biological, 'gender'~identity. Returns [0,1].
    """
    try:
        t = _normalize_text(text)
        cues = [
            'both', 'compromise', 'middle ground', 'common ground', 'mutual', 'respect', 'acknowledge'
        ]
        hand_cues = ['on one hand', 'on the other hand']
        # mapping cues
        sex_bio = ('sex' in t and ('biolog' in t or 'chromosome' in t or 'gamete' in t))
        gender_id = ('gender' in t and ('identity' in t or 'self identification' in t or 'self-identification' in t))
        score = 0.0
        # base cues
        score += sum(1 for c in cues if c in t)
        score += sum(1 for c in hand_cues if c in t)
        if sex_bio:
            score += 2
        if gender_id:
            score += 2
        # Normalize: max roughly ~ (len(cues)+len(hand_cues)+4) ~ 7+2+4=13
        return min(1.0, score / 10.0)
    except Exception:
        return 0.5

def _encode_text(text: str):
    try:
        return style_model.encode([text], convert_to_tensor=True, device=str(device))
    except TypeError:
        return style_model.encode([text], convert_to_tensor=True)

def human_preference_score(text: str, style_embedding=None) -> float:
    """Combine readability + conciseness (+ optional sweet-spot) + style similarity.
    Applies optional verbosity penalty and mediator-quality boost based on _HPARAMS.
    """
    read = readability_score(text)
    # Choose conciseness metric
    if _HPARAMS.get('enable_concise_sweetspot'):
        conc = conciseness_sweet_spot_score(
            text,
            target_words=int(_HPARAMS.get('concise_target', 80)),
            lower=int(_HPARAMS.get('concise_lower', 40)),
            upper=int(_HPARAMS.get('concise_upper', 140)),
            min_score=float(_HPARAMS.get('concise_min_score', 0.6)),
        )
    else:
        conc = conciseness_score(text)
    style_sim = 0.5
    try:
        if style_embedding is not None:
            emb_text = _encode_text(text)
            style_sim = float(util.cos_sim(emb_text, style_embedding).mean().item())
            # cos_sim is roughly [-1,1]; map to [0,1]
            style_sim = (style_sim + 1.0) / 2.0
    except Exception:
        pass
    base = 0.4 * read + 0.3 * conc + 0.3 * style_sim
    # Optional mediator quality blending
    if _HPARAMS.get('enable_mediator_quality'):
        mq = mediator_quality_score(text)
        w = float(_HPARAMS.get('mediator_weight', 0.0))
        w = max(0.0, min(1.0, w))
        base = (1 - w) * base + w * mq
    # Optional verbosity penalty
    if _HPARAMS.get('enable_verbosity_penalty'):
        pen = verbosity_penalty(
            text,
            max_breaks=int(_HPARAMS.get('verbosity_max_breaks', 5)),
            slope=float(_HPARAMS.get('verbosity_slope', 0.05)),
            max_penalty=float(_HPARAMS.get('verbosity_max_penalty', 0.2)),
        )
        vw = float(_HPARAMS.get('verbosity_weight', 0.5))
        base = max(0.0, base - vw * pen)
    return base

def fact_consistency_score(text: str) -> float:
    """Placeholder factuality proxy.
    Returns neutral 1.0 (best) for now. Hook point for future fact-checking.
    """
    return 1.0

def combined_logical_score(text: str, a: float = 0.5, b: float = 0.3, c: float = 0.2) -> float:
    """Weighted logical score using semantic, contradiction, and factuality.
    a+b+c should sum to 1.0 (not enforced strictly)."""
    sem = semantic_consistency_score(text)
    contra = contradiction_ratio(text)
    fact = fact_consistency_score(text)
    return a * sem + b * (1 - contra) + c * fact

def combined_preference_score(text: str, a: float = 0.5, b: float = 0.3, c: float = 0.2,
                              human_weight: float = 0.3, style_embedding=None) -> float:
    """Hybrid of logical and human-preference proxies.
    human_weight in [0,1]: 0 = pure logical, 1 = pure human preference.
    """
    logical = combined_logical_score(text, a, b, c)
    human_pref = human_preference_score(text, style_embedding)
    return (1 - human_weight) * logical + human_weight * human_pref

def compare_with_human_bias(text_a: str, text_b: str, human_weight: float = 0.3, style_embedding=None,
                            a: float = 0.5, b: float = 0.3, c: float = 0.2):
    score_a = combined_preference_score(text_a, a=a, b=b, c=c, human_weight=human_weight, style_embedding=style_embedding)
    score_b = combined_preference_score(text_b, a=a, b=b, c=c, human_weight=human_weight, style_embedding=style_embedding)
    print(f"Score A (hybrid): {score_a:.4f}")
    print(f"Score B (hybrid): {score_b:.4f}")
    if score_a > score_b:
        print("✅ Text A is preferred (logical + human bias)")
    elif score_b > score_a:
        print("✅ Text B is preferred (logical + human bias)")
    else:
        print("⚠️ Both are equally preferred")

def load_style_embedding_from_file(path: str):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            txt = f.read()
        return _encode_text(txt)
    except Exception:
        return None

# ---------- Prompt-aware answer correctness (with optional gold/reference) ----------
_NUM_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
    'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
}

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    # keep digits and letters, turn punctuation to spaces
    table = str.maketrans({ch: ' ' for ch in string.punctuation})
    return re.sub(r"\s+", " ", s.translate(table)).strip()

def _extract_first_number(s: str):
    # digits
    m = re.search(r"[-+]?\d+", s)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # number words (basic)
    for tok in _normalize_text(s).split():
        if tok in _NUM_WORDS:
            return _NUM_WORDS[tok]
    return None

def answer_correctness_score(prompt: str, response: str, gold: str | None = None) -> float:
    """Return [0,1] score whether response answers the prompt correctly.
    If gold is provided, use exact/number match backed by embedding similarity.
    Without gold, return a neutral 0.5 (can be extended with QA models).
    """
    if not gold:
        return 0.5
    resp_norm = _normalize_text(response)
    gold_norm = _normalize_text(gold)
    # 1) substring exact-ish
    exact = 1.0 if gold_norm and gold_norm in resp_norm else 0.0
    # 2) numeric match
    gold_num = _extract_first_number(gold)
    resp_num = _extract_first_number(response)
    num_match = 1.0 if (gold_num is not None and resp_num is not None and gold_num == resp_num) else 0.0
    # 3) semantic similarity between a concise gold statement and response
    try:
        stmt = f"Answer: {gold_norm}"
        emb_stmt = _encode_text(stmt)
        emb_resp = _encode_text(response)
        cos = float(util.cos_sim(emb_stmt, emb_resp).mean().item())
        sem = (cos + 1.0) / 2.0  # map [-1,1] -> [0,1]
    except Exception:
        sem = 0.5
    # Prefer exact/number when present; otherwise fallback to semantic
    base = max(exact, num_match)
    if base >= 1.0:
        return 1.0
    # weighted blend if no strong match
    return 0.6 * base + 0.4 * sem

def combined_task_score(text: str, prompt: str | None = None, gold: str | None = None,
                        answer_weight: float = 0.4,
                        human_weight: float = 0.3,
                        a: float = 0.5, b: float = 0.3, c: float = 0.2,
                        style_embedding=None) -> float:
    """Final score mixing logical, human preference, and prompt-answer correctness.
    If gold is missing, answer component backs off to neutral 0.5.
    """
    hybrid = combined_preference_score(text, a=a, b=b, c=c, human_weight=human_weight, style_embedding=style_embedding)
    ans = answer_correctness_score(prompt or "", text, gold)
    return (1 - answer_weight) * hybrid + answer_weight * ans

# -------------------------------
# 3. Compare two LLM outputs
# -------------------------------
def compare_llm_outputs(text_a, text_b):
    score_a = logical_score(text_a)
    score_b = logical_score(text_b)
    print(f"Score A: {score_a:.4f}")
    print(f"Score B: {score_b:.4f}")
    if score_a > score_b:
        print("✅ Text A is more logically correct")
    elif score_b > score_a:
        print("✅ Text B is more logically correct")
    else:
        print("⚠️ Both texts are equally consistent")

# -------------------------------
# 4. CLI entry points (single pair, files, CSV)
# -------------------------------
def _read_lines(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():
    import argparse
    import pandas as pd
    parser = argparse.ArgumentParser(description="Logical consistency comparator: single pair, files, or CSV; optional human-bias hybrid scoring")
    parser.add_argument('--test_case', type=str, default='', help='Optional test case id; used to expand {test_case} placeholders in paths')
    parser.add_argument('--text-a', type=str, help='Single text A')
    parser.add_argument('--text-b', type=str, help='Single text B')
    parser.add_argument('--a-file', type=str, help='Path to file with one text A per line')
    parser.add_argument('--b-file', type=str, help='Path to file with one text B per line')
    parser.add_argument('--csv', type=str, help='CSV file with pairs to evaluate')
    parser.add_argument('--col-a', type=str, default='text_a', help='Column name for text A in CSV')
    parser.add_argument('--col-b', type=str, default='text_b', help='Column name for text B in CSV')
    parser.add_argument('--out', type=str, help='Output CSV path for CSV mode')
    # Prompt-answer correctness knobs
    parser.add_argument('--prompt', type=str, default='', help='Prompt/question for single-pair mode')
    parser.add_argument('--gold', type=str, default='', help='Gold/reference short answer for single-pair mode')
    parser.add_argument('--answer-weight', type=float, default=0.4, help='Weight for answer correctness (0-1)')
    parser.add_argument('--col-prompt', type=str, default='prompt', help='CSV column for prompt')
    parser.add_argument('--col-gold', type=str, default='gold', help='CSV column for gold/reference answer')
    # Human-bias knobs
    parser.add_argument('--use-human-bias', action='store_true', help='Enable hybrid logical+human preference scoring')
    parser.add_argument('--human-weight', type=float, default=0.3, help='Weight for human preference proxies (0-1)')
    parser.add_argument('--style-ref-file', type=str, default='', help='Optional file with reference text for style similarity')
    parser.add_argument('--logic-a', type=float, default=0.5, help='Weight for semantic consistency')
    parser.add_argument('--logic-b', type=float, default=0.3, help='Weight for (1-contradiction)')
    parser.add_argument('--logic-c', type=float, default=0.2, help='Weight for factuality proxy')
    # Human-tuning flags (optional; all off by default for backward compatibility)
    parser.add_argument('--enable-concise-sweetspot', action='store_true', help='Use sweet-spot conciseness scoring around a target length')
    parser.add_argument('--concise-target', type=int, default=80, help='Target word count for sweet-spot conciseness')
    parser.add_argument('--concise-lower', type=int, default=40, help='Lower word bound before hitting min conciseness score')
    parser.add_argument('--concise-upper', type=int, default=140, help='Upper word bound before hitting min conciseness score')
    parser.add_argument('--concise-min-score', type=float, default=0.6, help='Minimum conciseness score at extremes')
    parser.add_argument('--enable-verbosity-penalty', action='store_true', help='Apply penalty for excessive line breaks/newlines')
    parser.add_argument('--verbosity-max-breaks', type=int, default=5, help='Allowed line breaks before applying penalty')
    parser.add_argument('--verbosity-slope', type=float, default=0.05, help='Penalty growth per extra break')
    parser.add_argument('--verbosity-max-penalty', type=float, default=0.2, help='Max penalty applied for verbosity')
    parser.add_argument('--verbosity-weight', type=float, default=0.5, help='Weight of verbosity penalty impact in human score')
    parser.add_argument('--enable-mediator-quality', action='store_true', help='Boost responses that acknowledge both sides and propose a compromise')
    parser.add_argument('--mediator-weight', type=float, default=0.0, help='Mix-in weight for mediator quality heuristic (0-1)')
    # Preset bundling for convenience
    parser.add_argument('--preset', type=str, default='', help='Optional preset: damp_overconfidence_{slight|moderate|aggressive} or damp_overconfidence (alias moderate). Applies grouped human-bias knobs before other weights.')
    args = parser.parse_args()

    style_emb = None
    if args.style_ref_file:
        style_emb = load_style_embedding_from_file(args.style_ref_file)

    # Apply human tuning flags to runtime hparams
    # If a preset is specified, override incoming args FIRST (can still be adjusted by explicit flags afterwards)
    def _apply_preset(tag: str):
        tag = tag.lower()
        if tag in ('damp_overconfidence', 'damp_overconfidence_moderate'):
            # Moderate: reduce human weight emphasis; add mediator + light verbosity + conciseness sweet spot
            args.human_weight = min(args.human_weight, 0.38)
            args.answer_weight = max(args.answer_weight, 0.58)
            args.enable_mediator_quality = True
            args.mediator_weight = max(args.mediator_weight, 0.35)
            args.enable_concise_sweetspot = True
            args.concise_target = 130
            args.concise_lower = 85
            args.concise_upper = 230
            args.concise_min_score = 0.60
            args.enable_verbosity_penalty = True
            args.verbosity_max_breaks = 5
            args.verbosity_slope = 0.030
            args.verbosity_max_penalty = 0.18
            args.verbosity_weight = 0.30
        elif tag in ('damp_overconfidence_slight', 'damp_overconfidence_light'):
            args.human_weight = min(args.human_weight, 0.40)
            args.answer_weight = max(args.answer_weight, 0.60)
            args.enable_mediator_quality = True
            args.mediator_weight = max(args.mediator_weight, 0.25)
        elif tag in ('damp_overconfidence_aggressive', 'damp_overconfidence_strong'):
            args.human_weight = min(args.human_weight, 0.40)
            args.answer_weight = max(args.answer_weight, 0.60)
            args.enable_mediator_quality = True
            args.mediator_weight = max(args.mediator_weight, 0.40)
            args.enable_concise_sweetspot = True
            args.concise_target = 125
            args.concise_lower = 85
            args.concise_upper = 220
            args.concise_min_score = 0.62
            args.enable_verbosity_penalty = True
            args.verbosity_max_breaks = 5
            args.verbosity_slope = 0.035
            args.verbosity_max_penalty = 0.22
            args.verbosity_weight = 0.35
            # Style anchor optional; if user provided style-ref-file, we optionally bump style weight via human formula adjustments later
        else:
            if tag:
                print(f"[Warn] Unknown preset '{tag}' ignored.")

    if args.preset:
        _apply_preset(args.preset)

    _HPARAMS['enable_concise_sweetspot'] = bool(args.enable_concise_sweetspot)
    _HPARAMS['concise_target'] = int(args.concise_target)
    _HPARAMS['concise_lower'] = int(args.concise_lower)
    _HPARAMS['concise_upper'] = int(args.concise_upper)
    _HPARAMS['concise_min_score'] = float(args.concise_min_score)
    _HPARAMS['enable_verbosity_penalty'] = bool(args.enable_verbosity_penalty)
    _HPARAMS['verbosity_max_breaks'] = int(args.verbosity_max_breaks)
    _HPARAMS['verbosity_slope'] = float(args.verbosity_slope)
    _HPARAMS['verbosity_max_penalty'] = float(args.verbosity_max_penalty)
    _HPARAMS['verbosity_weight'] = float(args.verbosity_weight)
    _HPARAMS['enable_mediator_quality'] = bool(args.enable_mediator_quality)
    _HPARAMS['mediator_weight'] = float(args.mediator_weight)

    # Expand {test_case} placeholders in paths and allow prompt/text to point to files
    def _expand_placeholders(p: str | None) -> str | None:
        if not p:
            return p
        return p.replace('{test_case}', args.test_case)

    args.prompt = _expand_placeholders(args.prompt)
    args.text_a = _expand_placeholders(args.text_a)
    args.text_b = _expand_placeholders(args.text_b)
    args.style_ref_file = _expand_placeholders(args.style_ref_file)

    # If a test_case is provided, auto-fill missing prompt/text paths to tmp/{test_case}/...
    if args.test_case:
        base_dir = os.path.join('tmp', args.test_case)
        if not args.prompt:
            args.prompt = os.path.join(base_dir, 'prompt.txt')
        if args.text_a is None:
            args.text_a = os.path.join(base_dir, 'test_a.txt')
        if args.text_b is None:
            args.text_b = os.path.join(base_dir, 'test_b.txt')

    # Allow --text-a / --text-b / --prompt to point to files; if so, load contents.
    def _maybe_load_file(p: str | None) -> str | None:
        if p and os.path.isfile(p):
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    return fh.read()
            except Exception:
                return p  # fall back to raw string
        return p
    # Keep original source paths (if they exist) for optional result reporting
    prompt_src_path = args.prompt if args.prompt and os.path.isfile(args.prompt) else None
    text_a_src_path = args.text_a if args.text_a and os.path.isfile(args.text_a) else None
    text_b_src_path = args.text_b if args.text_b and os.path.isfile(args.text_b) else None

    args.text_a = _maybe_load_file(args.text_a)
    args.text_b = _maybe_load_file(args.text_b)
    args.prompt = _maybe_load_file(args.prompt)

    # CSV mode
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.col_a not in df.columns or args.col_b not in df.columns:
            raise ValueError(f"CSV must contain columns '{args.col_a}' and '{args.col_b}'")
        scores_a, scores_b, winners = [], [], []
        prompts = df[args.col_prompt].astype(str) if args.col_prompt in df.columns else [""] * len(df)
        golds = df[args.col_gold].astype(str) if args.col_gold in df.columns else [""] * len(df)
        for ta, tb, pr, gd in zip(df[args.col_a].astype(str), df[args.col_b].astype(str), prompts, golds):
            if args.use_human_bias or gd:
                sa = combined_task_score(ta, prompt=pr, gold=(gd or None), answer_weight=args.answer_weight,
                                         human_weight=args.human_weight, a=args.logic_a, b=args.logic_b, c=args.logic_c,
                                         style_embedding=style_emb)
                sb = combined_task_score(tb, prompt=pr, gold=(gd or None), answer_weight=args.answer_weight,
                                         human_weight=args.human_weight, a=args.logic_a, b=args.logic_b, c=args.logic_c,
                                         style_embedding=style_emb)
            else:
                sa = logical_score(ta)
                sb = logical_score(tb)
            scores_a.append(sa)
            scores_b.append(sb)
            if sa > sb:
                winners.append('A')
            elif sb > sa:
                winners.append('B')
            else:
                winners.append('tie')
        df['score_a'] = scores_a
        df['score_b'] = scores_b
        df['winner'] = winners
        if args.use_human_bias:
            df['human_weight'] = args.human_weight
        if args.col_gold in df.columns:
            df['answer_weight'] = args.answer_weight
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"[Info] Wrote results to {args.out}")
        else:
            print(df.head())
        return

    # Two files mode
    if args.a_file and args.b_file:
        As = _read_lines(args.a_file)
        Bs = _read_lines(args.b_file)
        n = min(len(As), len(Bs))
        for i in range(n):
            print(f"\n[Pair {i+1}]----------------------")
            if args.use_human_bias or args.gold:
                sa = combined_task_score(As[i], prompt=args.prompt, gold=(args.gold or None),
                                         answer_weight=args.answer_weight, human_weight=args.human_weight,
                                         a=args.logic_a, b=args.logic_b, c=args.logic_c, style_embedding=style_emb)
                sb = combined_task_score(Bs[i], prompt=args.prompt, gold=(args.gold or None),
                                         answer_weight=args.answer_weight, human_weight=args.human_weight,
                                         a=args.logic_a, b=args.logic_b, c=args.logic_c, style_embedding=style_emb)
                print(f"Score A (task): {sa:.4f}")
                print(f"Score B (task): {sb:.4f}")
                print("✅ Text A is preferred" if sa > sb else ("✅ Text B is preferred" if sb > sa else "⚠️ Tie"))
            else:
                compare_llm_outputs(As[i], Bs[i])
        if len(As) != len(Bs):
            print(f"[Warn] Unequal lengths: {len(As)} vs {len(Bs)}. Compared first {n} pairs.")
        return

    # Single pair mode
    if args.text_a is not None and args.text_b is not None:
        if args.use_human_bias or args.gold:
            sa = combined_task_score(args.text_a, prompt=args.prompt, gold=(args.gold or None),
                                     answer_weight=args.answer_weight, human_weight=args.human_weight,
                                     a=args.logic_a, b=args.logic_b, c=args.logic_c, style_embedding=style_emb)
            sb = combined_task_score(args.text_b, prompt=args.prompt, gold=(args.gold or None),
                                     answer_weight=args.answer_weight, human_weight=args.human_weight,
                                     a=args.logic_a, b=args.logic_b, c=args.logic_c, style_embedding=style_emb)
            print(f"Score A (task): {sa:.4f}")
            print(f"Score B (task): {sb:.4f}")
            print("✅ Text A is preferred" if sa > sb else ("✅ Text B is preferred" if sb > sa else "⚠️ Tie"))
            # Optional automatic results writing if test_case provided
            if args.test_case:
                try:
                    base_dir = os.path.join('tmp', args.test_case)
                    os.makedirs(base_dir, exist_ok=True)
                    results_path = os.path.join(base_dir, 'results.txt')
                    # Compute component scores for transparency
                    logical_a = logical_score(args.text_a)
                    logical_b = logical_score(args.text_b)
                    answer_a = answer_correctness_score(args.prompt or "", args.text_a, args.gold or "") if args.gold else None
                    answer_b = answer_correctness_score(args.prompt or "", args.text_b, args.gold or "") if args.gold else None
                    with open(results_path, 'w', encoding='utf-8') as rf:
                        rf.write(f"prompt_path={prompt_src_path or ''}\n")
                        rf.write(f"text_a_path={text_a_src_path or ''}\n")
                        rf.write(f"text_b_path={text_b_src_path or ''}\n")
                        if args.gold:
                            rf.write(f"gold={args.gold}\n")
                        rf.write(f"score_a_task={sa:.6f}\n")
                        rf.write(f"score_b_task={sb:.6f}\n")
                        rf.write(f"logical_a={logical_a:.6f}\n")
                        rf.write(f"logical_b={logical_b:.6f}\n")
                        if answer_a is not None:
                            rf.write(f"answer_correctness_a={answer_a:.6f}\n")
                        if answer_b is not None:
                            rf.write(f"answer_correctness_b={answer_b:.6f}\n")
                        rf.write(f"human_weight={args.human_weight}\n")
                        rf.write(f"answer_weight={args.answer_weight}\n")
                        rf.write(f"logic_a_weight={args.logic_a}\n")
                        rf.write(f"logic_b_weight={args.logic_b}\n")
                        rf.write(f"logic_c_weight={args.logic_c}\n")
                        rf.write(f"winner={'A' if sa>sb else ('B' if sb>sa else 'tie')}\n")
                    print(f"[Info] Wrote single-pair results to {results_path}")
                except Exception as e:
                    print(f"[Warn] Failed to write results file: {e}")
        else:
            compare_llm_outputs(args.text_a, args.text_b)
        return

    # Demo fallback
    text_a = "The sun rises in the east. Therefore, mornings are bright. Rain does not affect sunlight."
    text_b = "The sun rises in the west. Mornings are bright. Rain falls often."
    print("[Info] No inputs provided; running demo pair. Use --text-a/--text-b or --csv.")
    compare_llm_outputs(text_a, text_b)


if __name__ == '__main__':
    main()
text_b = "To initialize the classification head when performing transfer learning, follow these steps:\n\n1. Load the pre-trained vision transformer model that was trained on ImageNet.\n2. Replace the existing classification head with a new one suitable for the StanfordCars dataset.\n3. Initialize the weights of the new classification head with random values or pre-trained weights if available.\n4. Freeze the weights of the vision transformer backbone to prevent them from being updated during training.\n5. Train the model on the StanfordCars dataset, fine-tuning the classification head while keeping the backbone frozen.\n\nHere's an example implementation in Python using PyTorch:\n\n```python\nimport torch\nimport torch.nn as nn\nimport torchvision.models as models\n\n# Step 1: Load pre-trained vision transformer model\nvision_transformer = models.vit_base_patch16_224(pretrained=True)\n\n# Step 2: Replace classification head\nnum_classes = 196  # Number of classes in StanfordCars dataset\nvision_transformer.head = nn.Linear(vision_transformer.head.in_features, num_classes)\n\n# Step 3: Initialize weights of new classification head\nnn.init.xavier_uniform_(vision_transformer.head.weight)\nnn.init.constant_(vision_transformer.head.bias, 0)\n\n# Step 4: Freeze the weights of the backbone\nfor param in vision_transformer.parameters():\n    param.requires_grad = False\n\n# Step 5: Train the model on StanfordCars dataset\n# ...\n```\n\nIn this example, we load the pre-trained vision transformer model using `torchvision.models.vit_base_patch16_224`. We then replace the `head` (classification head) of the model with a new linear layer (`nn.Linear`) that has the appropriate number of output classes for the StanfordCars dataset.\n\nAfter replacing the classification head, we initialize its weights using Xavier initialization (`nn.init.xavier_uniform_`) and set its biases to zero (`nn.init.constant_`).\n\nFinally, we freeze the weights of the vision transformer backbone by setting `requires_grad=False` for all parameters. This ensures that only the weights of the classification head will be updated during training on the StanfordCars dataset.","If you want to perform full finetuning, where both the classification head and the backbone of the vision transformer are updated during training on the StanfordCars dataset, you can omit freezing the backbone weights. Here's an updated version of the code:\n\n```python\nimport torch\nimport torch.nn as nn\nimport torchvision.models as models\n\n# Step 1: Load pre-trained vision transformer model\nvision_transformer = models.vit_base_patch16_224(pretrained=True)\n\n# Step 2: Replace classification head\nnum_classes = 196  # Number of classes in StanfordCars dataset\nvision_transformer.head = nn.Linear(vision_transformer.head.in_features, num_classes)\n\n# Step 3: Initialize weights of new classification head\nnn.init.xavier_uniform_(vision_transformer.head.weight)\nnn.init.constant_(vision_transformer.head.bias, 0)\n\n# Step 4: Train the model on StanfordCars dataset\n# ...\n```\n\nIn this case, you can directly train the vision transformer model on the StanfordCars dataset without freezing any weights. During training, both the weights of the classification head and the backbone will be updated.\n\nKeep in mind that full finetuning may require more computational resources and a larger labeled dataset compared to freezing the backbone. It can help the model learn more specific features related to the StanfordCars dataset but may also increase the risk of overfitting if the dataset is small. Monitoring the training progress and adjusting hyperparameters, such as learning rate and regularization, can help mitigate overfitting."


