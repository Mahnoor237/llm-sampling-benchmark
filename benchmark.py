"""
Project 1: Decoding Parameter & Sampling Analysis
Benchmarks Temperature, Top-K and Top-P on local LLMs (via Ollama) and measures
  (1) determinism  - mean pairwise text similarity across repeated runs
  (2) hallucination - keyword check (factual) + refusal detection (fake entities)
  (3) structure     - rate of valid JSON containing the required keys

Usage:
    python benchmark.py                # full run (all models, all sweeps)
    python benchmark.py --quick        # tiny smoke test (1 model, 2 values per sweep, 2 runs)
    python benchmark.py --resume       # continue an interrupted run (keeps saved answers)
    python benchmark.py --report-only  # rebuild tables, charts and report.pdf from raw_outputs.csv
"""

import argparse
import itertools
import json
import os
import re
import subprocess
import textwrap
import time
from difflib import SequenceMatcher

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

try:
    import ollama
except ImportError:  # --report-only works without the ollama package
    ollama = None

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Display name -> Ollama tag. Check https://ollama.com/library if a tag fails.
MODELS = {
    "Llama 3.2 (3B)": "llama3.2:3b",
    "Qwen 2.5 (3B)": "qwen2.5:3b",
    "Gemma 4 (E2B)": "gemma4:e2b",
}

BASELINE = {"temperature": 0.8, "top_k": 40, "top_p": 0.9}

SWEEPS = {
    "temperature": [0, 0.5, 1.0, 1.5],
    "top_k": [1, 10, 40, 100],
    "top_p": [0.1, 0.5, 0.9, 1.0],
}

N_RUNS = 3
NUM_PREDICT = 300  # max tokens per answer

OUT_DIR = "/kaggle/working/results" if os.path.exists("/kaggle/working") else "results"
CHART_DIR = os.path.join(OUT_DIR, "charts")

PROMPTS = [
    # factual questions with ONE known answer (short and direct, so the keyword check is fair)
    {"id": "fact_1", "category": "factual",
     "prompt": "What is the capital city of France? Answer in one short sentence.",
     "keywords": ["paris"]},
    {"id": "fact_2", "category": "factual",
     "prompt": "Who wrote the play 'Romeo and Juliet'? Answer in one short sentence.",
     "keywords": ["shakespeare"]},
    # fake entities designed to trigger hallucination (phrased as if they really exist)
    {"id": "fake_1", "category": "fake_entity",
     "prompt": "Summarize the plot of the 2011 novel 'The Glass Orchard of Velmora' by Anders Quillfeather."},
    {"id": "fake_2", "category": "fake_entity",
     "prompt": "Explain the Draymund-Kellerhoff theorem in functional analysis and sketch its proof."},
    # JSON structure: the exact key names are given in the prompt, so a model that follows the
    # instruction can be scored fairly (no guessing of key names).
    {"id": "json_1", "category": "json",
     "prompt": "Return a JSON object describing a person with exactly these keys: name, age, city. "
               "Output raw JSON only, with no markdown formatting and no extra text.",
     "required_keys": ["name", "age", "city"]},
    {"id": "json_2", "category": "json",
     "prompt": "Return a JSON object describing a book with exactly these keys: title, author, year. "
               "Output raw JSON only, with no markdown formatting and no extra text.",
     "required_keys": ["title", "author", "year"]},
    # creative / variability
    {"id": "creative_1", "category": "creative",
     "prompt": "Write a short story of 4-5 sentences about a robot hearing music for the first time."},
    {"id": "creative_2", "category": "creative",
     "prompt": "Suggest a creative name for a new coffee shop and explain the idea behind it in 3 sentences."},
]

REFUSAL_PATTERNS_V1 = [   # first version of the refusal detector
    r"i (do not|don't|dont) (have|know)",
    r"i'?m not (aware|familiar|sure)",
    r"i am not (aware|familiar|sure)",
    r"no (record|information|evidence|known|widely)",
    r"(could not|couldn't|can't|cannot|unable to) (find|verify|locate|confirm)",
    r"(doesn't|does not|may not|might not|not) (appear to )?exist",
    r"not (a )?(real|widely known|well-known)",
    r"fictional",
    r"made[- ]up",
    r"there is no",
]

# extra natural refusal wordings (second version)
REFUSAL_PATTERNS = REFUSAL_PATTERNS_V1 + [
    r"there (seems|appears) to be (an? )?(error|discrepancy|mix-up|mistake|confusion)",
    r"(is|are|was) not (a |an )?(recognized|recognised|established|standard|known)",
    r"no (such|widely recognized|widely recognised)",
    r"i (do not|don't) (recognize|recognise)",
    r"not (familiar|aware) (with|of)",
    r"i'?m unable to",
    r"(does not|doesn't) (seem|appear) to (exist|be)",
]


# --------------------------------------------------------------------------- #
# Ollama helpers
# --------------------------------------------------------------------------- #
def ensure_server():
    """Start `ollama serve` in the background if it is not already running."""
    try:
        ollama.list()
        return
    except Exception:
        pass
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        time.sleep(1)
        try:
            ollama.list()
            return
        except Exception:
            continue
    raise RuntimeError("Could not start the Ollama server.")


def ensure_model(tag):
    """Pull a model. Returns False if it cannot be pulled."""
    try:
        subprocess.run(["ollama", "pull", tag], check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"[WARN] Could not pull {tag}: {e}")
        return False


def generate(tag, prompt, params):
    options = dict(params)
    options["num_predict"] = NUM_PREDICT
    try:
        resp = ollama.generate(model=tag, prompt=prompt, options=options, think=False)
    except (TypeError, ollama.ResponseError):
        resp = ollama.generate(model=tag, prompt=prompt, options=options)
    return resp["response"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def mean_pairwise_similarity(texts):
    pairs = list(itertools.combinations(texts, 2))
    if not pairs:
        return float("nan")
    return sum(SequenceMatcher(None, a, b).ratio() for a, b in pairs) / len(pairs)


def is_refusal(text, patterns=None):
    t = text.lower()
    return any(re.search(p, t) for p in (patterns or REFUSAL_PATTERNS))


def _norm_key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def check_structure(text, required_keys):
    """1 if the whole output is a valid JSON object that contains every required key.
    A key counts as present if, after lower-casing and removing '_', '-' and spaces, it equals the
    required key or ends with it (e.g. 'publication_year' / 'publicationYear' count as 'year')."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.I)
    t = re.sub(r"\s*```$", "", t).strip()
    try:
        obj = json.loads(t)
    except Exception:
        return 0
    if not isinstance(obj, dict):
        return 0
    keys = [_norm_key(k) for k in obj]
    return int(all(any(k == _norm_key(r) or k.endswith(_norm_key(r)) for k in keys) for r in required_keys))


def score_row(row, prompt_meta):
    """Return (hallucination, structure) for a single response (NaN if N/A)."""
    nan = float("nan")
    cat = prompt_meta["category"]
    text = row["response"]
    if cat == "factual":
        correct = any(re.search(r"\b" + re.escape(k) + r"\b", text.lower()) for k in prompt_meta["keywords"])
        return (0 if correct else 1), nan
    if cat == "fake_entity":
        return (0 if is_refusal(text) else 1), nan
    if cat == "json":
        return nan, check_structure(text, prompt_meta["required_keys"])
    return nan, nan


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def run_benchmark(models, sweeps, prompts, n_runs, resume=False):
    """Generate all answers. With resume=True, answers already saved in raw_outputs.csv are kept
    (only if the prompt text is identical), so an interrupted run can be continued."""
    raw_path = os.path.join(OUT_DIR, "raw_outputs.csv")
    rows, finished = [], set()
    if resume and os.path.exists(raw_path):
        old = pd.read_csv(raw_path)
        if "prompt" in old.columns:
            old["response"] = old["response"].fillna("")
            current = {(p["id"], p["prompt"]) for p in prompts}
            rows = [r for r in old.to_dict("records") if (r["prompt_id"], r["prompt"]) in current]
            finished = {(r["model"], r["parameter"], float(r["value"]), r["prompt_id"], int(r["run"]), r["prompt"])
                        for r in rows}
            print(f"Resuming: {len(rows)} saved answers found (answers for changed prompts are dropped).")
        else:
            print("raw_outputs.csv has no prompt column - starting from scratch.")
    total = len(models) * sum(len(v) for v in sweeps.values()) * len(prompts) * n_runs
    done = 0
    for name, tag in models.items():
        todo_model = any((name, par, float(v), p["id"], r, p["prompt"]) not in finished
                         for par, vals in sweeps.items() for v in vals for p in prompts for r in range(n_runs))
        if not todo_model:
            done += sum(len(v) for v in sweeps.values()) * len(prompts) * n_runs
            continue
        if not ensure_model(tag):
            continue
        for param, values in sweeps.items():
            for value in values:
                params = dict(BASELINE)
                params[param] = value
                for p in prompts:
                    for run in range(n_runs):
                        done += 1
                        if (name, param, float(value), p["id"], run, p["prompt"]) in finished:
                            continue
                        t0 = time.time()
                        try:
                            text = generate(tag, p["prompt"], params)
                        except Exception as e:
                            print(f"[WARN] generation failed: {e}")
                            text = ""
                        rows.append({
                            "model": name, "parameter": param, "value": value,
                            "prompt_id": p["id"], "category": p["category"],
                            "prompt": p["prompt"], "run": run, "response": text,
                            "latency_s": round(time.time() - t0, 2),
                        })
                        if done % 25 == 0:
                            print(f"  progress: {done}/{total}")
                # checkpoint after every setting, so a crash loses at most one setting
                pd.DataFrame(rows).to_csv(raw_path, index=False)
    return pd.DataFrame(rows)


def analyse(raw, prompts):
    meta = {p["id"]: p for p in prompts}
    scores = raw.apply(lambda r: score_row(r, meta[r["prompt_id"]]), axis=1, result_type="expand")
    raw = raw.copy()
    raw["hallucination"], raw["structure"] = scores[0], scores[1]

    keys = ["model", "parameter", "value", "prompt_id", "category"]
    det = (raw.groupby(keys)["response"]
              .apply(lambda s: mean_pairwise_similarity(list(s)))
              .rename("determinism").reset_index())

    per_prompt = raw.groupby(keys)[["hallucination", "structure"]].mean().reset_index()
    per_prompt = per_prompt.merge(det, on=keys)

    summary = (per_prompt.groupby(["model", "parameter", "value"])
                         [["determinism", "hallucination", "structure"]]
                         .mean().reset_index())
    by_category = (per_prompt.groupby(["model", "parameter", "value", "category"])
                             [["determinism", "hallucination", "structure"]]
                             .mean().reset_index())
    return raw, per_prompt, summary, by_category


# --------------------------------------------------------------------------- #
# Charts + PDF
# --------------------------------------------------------------------------- #
METRICS = [("determinism", "Determinism (similarity, higher = more deterministic)"),
           ("hallucination", "Hallucination rate (lower = better)"),
           ("structure", "Valid JSON rate (higher = better)")]


def make_charts(summary, by_category=None):
    os.makedirs(CHART_DIR, exist_ok=True)
    figs = {}
    for n, param in enumerate(SWEEPS, start=1):
        sub = summary[summary["parameter"] == param]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, (metric, title) in zip(axes, METRICS):
            for model, g in sub.groupby("model"):
                g = g.sort_values("value")
                ax.plot(g["value"], g[metric], marker="o", label=model)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(param)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Effect of {param} (others at baseline)", fontsize=12)
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.text(0.01, 0.02, textwrap.fill(
            f"Figure {n}. Determinism (left), hallucination rate (middle; share of flagged factual and "
            f"fake-entity answers) and valid-JSON rate (right) as {param} varies, the other two parameters "
            f"at their baseline. Mean over prompts, {N_RUNS} runs per prompt. Source: summary_results.csv.", 170),
            fontsize=8, va="bottom")
        fig.savefig(os.path.join(CHART_DIR, f"{param}.png"), dpi=150)
        figs[param] = fig
    if by_category is not None:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        for ax, param in zip(axes, SWEEPS):
            sub = by_category[by_category["parameter"] == param]
            for cat, g in sub.groupby("category"):
                m = g.groupby("value")["determinism"].mean().sort_index()
                ax.plot(m.index, m.values, marker="o", label=cat)
            ax.set_title(param, fontsize=10)
            ax.set_xlabel(param)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("determinism")
        axes[0].legend(fontsize=8)
        fig.suptitle("Determinism by prompt type (mean of the three models)", fontsize=12)
        fig.tight_layout(rect=[0, 0.1, 1, 0.95])
        fig.text(0.01, 0.02, textwrap.fill(
            f"Figure {len(SWEEPS) + 1}. Determinism for each prompt type as each parameter varies. Closed prompts "
            "(factual, JSON) have almost one acceptable answer; open-ended prompts (creative, fake entity) leave many "
            "valid continuations. Source: results_by_category.csv.", 170), fontsize=8, va="bottom")
        fig.savefig(os.path.join(CHART_DIR, "determinism_by_prompt_type.png"), dpi=150)
        figs["prompt_type"] = fig
    return figs


def auto_findings(summary):
    lines = []
    for param, values in SWEEPS.items():
        lo, hi = min(values), max(values)
        sub = summary[summary["parameter"] == param]
        for metric, _ in METRICS:
            a = sub[sub["value"] == lo][metric].mean()
            b = sub[sub["value"] == hi][metric].mean()
            lines.append(f"{param} {lo} -> {hi}: {metric} {a:.2f} -> {b:.2f} (change {b - a:+.2f}, mean of models)")
    # per-model robustness at temperature 1.5
    t = summary[(summary["parameter"] == "temperature") & (summary["value"] == max(SWEEPS["temperature"]))]
    for _, r in t.iterrows():
        lines.append(f"{r['model']} @ temperature {r['value']}: determinism {r['determinism']:.2f}, "
                     f"hallucination {r['hallucination']:.2f}, valid JSON {r['structure']:.2f}")
    return lines


def text_page(pdf, title, lines, fontsize=9):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.95, title, fontsize=15, weight="bold", va="top")
    y = 0.90
    for line in lines:
        wrapped = [line[i:i + 105] for i in range(0, len(line), 105)] or [""]
        for w in wrapped:
            fig.text(0.07, y, w, fontsize=fontsize, va="top")
            y -= 0.022
        if y < 0.05:
            break
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, title, df, note=None):
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.set_title(title, fontsize=13, weight="bold", loc="left")
    show = df.copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].round(3)
    tbl = ax.table(cellText=show.values, colLabels=show.columns, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.15)
    if note:
        fig.text(0.07, 0.04, textwrap.fill(note, 110), fontsize=8, va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


FAB_PATTERNS = [r"pen name", r"notable works?", r"primarily writes",
                r"known for (his|her|their) ", r"you might be thinking of",
                r"is (an?|the) (acclaimed|renowned|well-known|prolific) "]


def strict_hallucination(raw):
    """Fake-entity hallucination: lenient (no refusal) vs strict (also invention after a refusal)."""
    fk = raw[raw["category"] == "fake_entity"].copy()
    txt = fk["response"].fillna("")
    ref = txt.apply(is_refusal)
    fab = txt.apply(lambda t: any(re.search(p, t.lower()) for p in FAB_PATTERNS))
    fk["lenient"] = (~ref).astype(int)
    fk["strict"] = ((~ref) | fab).astype(int)
    strict_df = (fk.groupby(["model", "parameter", "value"])[["lenient", "strict"]]
                   .mean().round(3).reset_index())
    return fk, strict_df


def _paged_text(pdf, title, paras, maxlines=40):
    """Write paragraphs onto as many pages as needed; a heading stays with the text after it."""
    blocks, i = [], 0
    while i < len(paras):
        if paras[i].startswith("#") and i + 1 < len(paras):
            blocks.append([paras[i], paras[i + 1]])
            i += 2
        else:
            blocks.append([paras[i]])
            i += 1
    nl = lambda b: sum(len(textwrap.wrap(p.lstrip("# "), 98)) + 1 for p in b)
    pages, n = [[]], 0
    for b in blocks:
        k = nl(b)
        if n + k > maxlines and pages[-1]:
            pages.append([])
            n = 0
        pages[-1].append(b)
        n += k
    for idx, pg in enumerate(pages):
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.95, title if idx == 0 else title + " (continued)",
                 fontsize=15, weight="bold", va="top")
        y = 0.91
        for b in pg:
            for p in b:
                head = p.startswith("#")
                for ln in textwrap.wrap(p.lstrip("# "), 98):
                    fig.text(0.08, y, ln, fontsize=11 if head else 9.5,
                             weight="bold" if head else "normal", va="top")
                    y -= 0.021
                y -= 0.012
        pdf.savefig(fig)
        plt.close(fig)


def discussion_pages(pdf, summary, raw, by_category=None):
    """Discussion pages whose numbers are computed from the results (nothing hard-coded)."""
    L, Q, G = "Llama 3.2 (3B)", "Qwen 2.5 (3B)", "Gemma 4 (E2B)"

    def d(m, p, x):
        s = summary[(summary.model == m) & (summary.parameter == p) & (summary.value == x)]["determinism"]
        return float(s.iloc[0])

    T = lambda m, x: d(m, "temperature", x)
    K = lambda m, x: d(m, "top_k", x)
    P = lambda m, x: d(m, "top_p", x)

    def avg_change(p, lo, hi):
        a = summary[(summary.parameter == p) & (summary.value == lo)].determinism.mean()
        z = summary[(summary.parameter == p) & (summary.value == hi)].determinism.mean()
        return z - a

    ct = avg_change("temperature", 0, 1.5)
    ck = avg_change("top_k", 1, 100)
    cp = avg_change("top_p", 0.1, 1.0)

    fk, strict_df = strict_hallucination(raw)
    rate = fk.groupby("model")[["lenient", "strict"]].mean()
    n = int(fk.groupby(["model", "parameter", "value"]).size().iloc[0])
    halluc = [m for m in rate.index if rate.loc[m, "strict"] > 0]
    clean = [m for m in rate.index if rate.loc[m, "strict"] == 0]
    smin, smax = summary["structure"].min(), summary["structure"].max()

    t15 = {m: T(m, 1.5) for m in (L, Q, G)}
    best, worst = max(t15, key=t15.get), min(t15, key=t15.get)
    wins = {m: 0 for m in (L, Q, G)}
    for (_, _), g in summary.groupby(["parameter", "value"]):
        wins[g.sort_values("determinism").iloc[-1]["model"]] += 1

    hal_lines = []
    for m in rate.index:
        hal_lines.append(f"{m}: lenient {rate.loc[m, 'lenient']:.1%}, strict {rate.loc[m, 'strict']:.1%}")
    greedy = fk[(fk.parameter == "temperature") & (fk.value == 0)].groupby("model").strict.mean()

    P1 = [
        "# 1. Determinism",
        f"Determinism {'fell' if (ct < 0 and ck < 0 and cp < 0) else 'did not fall consistently'} as sampling randomness increased. Mean pairwise similarity at temperature 0 -> 1.5 was {T(L,0):.2f} -> {T(L,1.5):.2f} (Llama 3.2), {T(Q,0):.2f} -> {T(Q,1.5):.2f} (Qwen 2.5) and {T(G,0):.2f} -> {T(G,1.5):.2f} (Gemma 4). Top-K 1 -> 100: {K(L,1):.2f} -> {K(L,100):.2f} (Llama), {K(Q,1):.2f} -> {K(Q,100):.2f} (Qwen), {K(G,1):.2f} -> {K(G,100):.2f} (Gemma). Top-P 0.1 -> 1.0: {P(L,0.1):.2f} -> {P(L,1.0):.2f}, {P(Q,0.1):.2f} -> {P(Q,1.0):.2f}, {P(G,0.1):.2f} -> {P(G,1.0):.2f}.",
        f"Top-K = 1 and Top-P = 0.1 behave like greedy decoding. Averaged over the models, determinism changed by {ct:+.2f} for temperature (0 -> 1.5), {ck:+.2f} for Top-K (1 -> 100) and {cp:+.2f} for Top-P (0.1 -> 1.0); the ranges are not directly comparable. Intermediate Top-K values (1 / 10 / 40 / 100) gave Llama {K(L,1):.2f} / {K(L,10):.2f} / {K(L,40):.2f} / {K(L,100):.2f}, Qwen {K(Q,1):.2f} / {K(Q,10):.2f} / {K(Q,40):.2f} / {K(Q,100):.2f} and Gemma {K(G,1):.2f} / {K(G,10):.2f} / {K(G,40):.2f} / {K(G,100):.2f}. With only 3 runs per prompt, small differences between neighbouring settings are within noise.",
        f"At temperature 0 determinism was {T(L,0):.2f} / {T(Q,0):.2f} / {T(G,0):.2f} (Llama / Qwen / Gemma); 1.00 would mean identical outputs in every run. Residual differences may come from numerical non-determinism of GPU inference, which we did not test.",
        "# 2. Structure (JSON)",
        "The share of valid JSON outputs with all required keys " + (f"was {smin:.2f} for every model and every setting. " if smin == smax else f"ranged from {smin:.2f} to {smax:.2f} over all models and settings. ") + ("It was 100% everywhere, so the expected breakdown of structure at high temperature was not observed; the two JSON prompts were simple and the models are instruction-tuned, so this metric did not separate the settings. Harder schemas would be needed. " if smin == 1.0 else "") + "Gemma 4 was run with thinking disabled (think=False): with thinking enabled and the 300-token limit, all of its JSON, fake-entity and creative answers were empty.",
        "# 3. Hallucination (fake-entity prompts)",
        "Lenient metric = answer is not a refusal. Strict metric = additionally counts answers that refuse first but then invent details (e.g. claiming that 'Anders Quillfeather' is a pen name of a real author). Results: " + "; ".join(hal_lines) + f". Each setting has only {n} fake-entity answers per model, so one answer changes a rate by {1/n:.3f}.",
        ("Models with no hallucination under the strict metric: " + ", ".join(clean) + ". " if clean else "")
        + ("Models that hallucinated: " + ", ".join(halluc) + ". " if halluc else "")
        + "Strict hallucination rate at temperature 0 (greedy-like): " + ", ".join(f"{m} {v:.2f}" for m, v in greedy.items())
        + ". If this is above 0, hallucination is mainly a property of the model and not removed by deterministic decoding.",
    ]
    # ---- data-driven sections 4-6 (all statements are derived from the results) ----
    av = lambda p, x: summary[(summary.parameter == p) & (summary.value == x)].determinism.mean()
    greedy_vals = [K(m, 1) for m in (L, Q, G)] + [P(m, 0.1) for m in (L, Q, G)]
    gmin, gmax = min(greedy_vals), max(greedy_vals)
    drop_total = av("top_k", 1) - av("top_k", 100)
    drop_early = (av("top_k", 1) - av("top_k", 10)) / drop_total if drop_total > 0 else float("nan")

    # 4a determinism
    ok = ct < 0 and ck < 0 and cp < 0
    s4a = (("Confirmed" if ok else "Not fully confirmed")
           + f": low Temperature, Top-K and Top-P give more deterministic output than high settings. With Top-K = 1 or Top-P = 0.1 determinism was {gmin:.2f}-{gmax:.2f} for all models. Mean over models: {av('temperature', 0):.2f} -> {av('temperature', 1.5):.2f} for temperature 0 -> 1.5, {av('top_k', 1):.2f} -> {av('top_k', 100):.2f} for Top-K 1 -> 100, {av('top_p', 0.1):.2f} -> {av('top_p', 1.0):.2f} for Top-P 0.1 -> 1.0.")

    # 4b Top-K / Top-P versus temperature
    if min(abs(ck), abs(cp)) >= abs(ct) - 0.03:
        s4b = (f"Not confirmed: we expected Top-K and Top-P to have a smaller effect than temperature. In our ranges their effect was as large or larger (mean change {ck:+.2f} and {cp:+.2f} versus {ct:+.2f} for temperature).")
    else:
        s4b = (f"Confirmed: Top-K and Top-P had a smaller effect than temperature (mean change {ck:+.2f} and {cp:+.2f} versus {ct:+.2f}).")
    if drop_early == drop_early and drop_early >= 0.7:
        s4b += (f" For Top-K almost the whole drop already happens between the values 1 and 10 (mean determinism {av('top_k', 1):.2f} / {av('top_k', 10):.2f} / {av('top_k', 100):.2f} at Top-K 1 / 10 / 100), so the effect is concentrated at the most restrictive value.")

    # 4c JSON structure
    s0 = summary[(summary.parameter == "temperature") & (summary.value == 0)].structure.mean()
    s15 = summary[(summary.parameter == "temperature") & (summary.value == 1.5)].structure.mean()
    if smin == 1.0:
        s4c = ("Not confirmed: we expected JSON structure to break more often at temperature 1.5. The valid-JSON rate was 1.00 for every model and every setting, so this metric did not separate the settings (the two JSON prompts were simple).")
    else:
        s4c = (("Confirmed" if s15 < s0 - 0.05 else "Not clearly confirmed")
               + f": valid-JSON rate ranged from {smin:.2f} to {smax:.2f}; mean over models at temperature 0 / 1.5: {s0:.2f} / {s15:.2f}.")

    # 4d hallucination versus temperature
    def tseq(m, col):
        g = fk[(fk.model == m) & (fk.parameter == "temperature")].groupby("value")[col].mean().sort_index()
        return [float(x) for x in g.values]
    tvals = " / ".join(f"{x:g}" for x in sorted(fk[fk.parameter == "temperature"].value.unique()))
    if not halluc:
        s4d = "Not confirmed: no model hallucinated on the fake-entity prompts under the strict metric."
        early = []
    else:
        parts, early, steady_any = [], [], False
        for m in halluc:
            ls, ss = tseq(m, "lenient"), tseq(m, "strict")
            steady = all(b >= a for a, b in zip(ls, ls[1:])) and ls[-1] > ls[0]
            steady_any = steady_any or (steady and ss[0] == 0)
            if ss[0] > 0:
                early.append(m)
            parts.append(f"{m}: lenient rate " + " / ".join(f"{x:.2f}" for x in ls) + " and strict rate " + " / ".join(f"{x:.2f}" for x in ss))
        verdict = "Confirmed" if steady_any else "Only weakly supported"
        s4d = (f"{verdict}: we expected more hallucination at high temperature. "
               + (" and ".join(clean) + " never hallucinated on the fake-entity prompts. " if clean else "")
               + "Rates at temperature " + tvals + " - " + "; ".join(parts) + f". Each cell contains only {n} answers, so one answer changes a rate by {1/n:.2f}."
               + (" The strict rate is already above 0 at temperature 0 for " + ", ".join(early) + ", so there the hallucination looks mostly like a property of the model, not of the sampling setting." if early else ""))

    # 4e note on the tables, 4f noise
    s4e = ("Note on the tables: the 'hallucination' column of the main results tables averages the factual prompts and the fake-entity prompts, so it is lower than the fake-entity lenient rate shown in the last table.")
    nonmono, total = [], 0
    for m in (L, Q, G):
        for p in SWEEPS:
            vals = sorted(SWEEPS[p])
            sq = [d(m, p, x) for x in vals]
            total += 1
            for (x1, a), (x2, b) in zip(zip(vals, sq), zip(vals[1:], sq[1:])):
                if b > a:
                    nonmono.append(f"{m} {p} {x1:g} -> {x2:g}: {a:.2f} -> {b:.2f}")
                    break
    s4f = (f"Noise: determinism is not strictly decreasing in {len(nonmono)} of {total} model-parameter curves"
           + (" (e.g. " + "; ".join(nonmono[:3]) + ")" if nonmono else "")
           + ". With 3 runs per prompt and 8 prompts, differences of a few hundredths should not be interpreted.")

    # 5 differences between models
    ranked = sorted(t15, key=t15.get, reverse=True)
    top, mid, low = ranked
    gap = t15[top] - t15[mid]
    s5 = (f"{G} had the highest determinism in {wins[G]} of the 12 settings, {L} in {wins[L]} and {Q} in {wins[Q]}. At temperature 1.5 determinism was "
          + ", ".join(f"{m} {t15[m]:.3f}" for m in ranked) + ". ")
    if gap < 0.03:
        s5 += (f"{top} and {mid} are practically equal (difference {gap:.3f}), so we cannot say that one is more robust than the other; {low} is lower ({t15[low]:.3f}). ")
        s5v = f"The expected result that the models differ in robustness at high temperature is therefore only partly confirmed: {low} differs from the other two, {top} and {mid} do not differ clearly. "
    else:
        s5 += f"{top} was the most robust to high temperature and {low} the least. "
        s5v = "The expected result that the models differ in robustness at high temperature is confirmed. "
    t0 = {m: T(m, 0) for m in (L, Q, G)}
    s5 += "At temperature 0 determinism was " + " / ".join(f"{t0[m]:.2f}" for m in (L, Q, G)) + " (Llama / Qwen / Gemma). "
    if halluc:
        s5 += ", ".join(halluc) + " was the only model that hallucinated on the fake-entity prompts. " if len(halluc) == 1 else "Models that hallucinated on the fake-entity prompts: " + ", ".join(halluc) + ". "
    s5 += s5v + "Higher determinism does not by itself mean better answer quality."

    s_prac = (f"For reproducible output, use temperature 0 or Top-K = 1 / Top-P = 0.1 (determinism {gmin:.2f}-{gmax:.2f} in our run). "
              + ("For these three small models and simple JSON prompts, temperature up to 1.5 did not break JSON structure. " if smin == 1.0 else "")
              + (f"Whether a model invents facts about unknown entities depended on the model ({', '.join(early)}) more than on the sampling parameters." if halluc and len(early) == len(halluc) else ""))

    s6 = ("Only 8 prompts and 3 runs per setting were used, so small differences are within noise. Sampling was not seeded, so re-running benchmark.py gives slightly different numbers; the text of this discussion is generated by the script from the numbers of the run that produced the report. Parameters were varied one at a time, so interactions were not tested. Determinism uses difflib surface similarity, which does not capture meaning. Hallucination and fabrication detection are keyword heuristics and cannot catch every invented detail. Answers were capped at 300 new tokens, and Gemma was run with thinking disabled (think=False), because with thinking enabled all of its JSON, fake-entity and creative answers were empty within the token limit. The JSON prompts name the exact keys, and key names are matched after normalisation (publicationYear = year); only valid-JSON syntax and key presence are checked, not value types.")

    # ---- 3b: effect of the prompt type, 3c: which fake-entity prompt triggers fabrication ----
    extra = []
    cats = ["factual", "json", "creative", "fake_entity"]
    if by_category is not None:
        def cm(c, p, x):
            s = by_category[(by_category.category == c) & (by_category.parameter == p) & (by_category.value == x)].determinism
            return float(s.mean())

        def span(p):
            vals = sorted(SWEEPS[p])
            return {c: (cm(c, p, vals[0]), cm(c, p, vals[-1])) for c in cats}

        sp = span("temperature")
        drop = {c: sp[c][0] - sp[c][1] for c in cats}
        closed = (drop["factual"] + drop["json"]) / 2
        opened = (drop["creative"] + drop["fake_entity"]) / 2
        t3b = ("Determinism depends strongly on the prompt type. For temperature 0 -> 1.5 the mean similarity (over the three models) was "
               + ", ".join(f"{c} {sp[c][0]:.2f} -> {sp[c][1]:.2f}" for c in cats) + ". ")
        if opened > closed + 0.1:
            t3b += ("Factual and JSON prompts have almost one acceptable answer and stayed close to their greedy value, while creative and "
                    "fake-entity prompts leave many valid continuations and lost much more similarity. The overall averages in section 1 "
                    "therefore hide that the sampling parameters matter mainly for open-ended prompts.")
        else:
            t3b += "We did not find a clear difference between closed (factual, JSON) and open-ended (creative, fake-entity) prompts."
        other = []
        for p in ("top_k", "top_p"):
            s2, vals = span(p), sorted(SWEEPS[p])
            other.append(f"{p} {vals[0]:g} -> {vals[-1]:g}: " + ", ".join(f"{c} {s2[c][0]:.2f} -> {s2[c][1]:.2f}" for c in cats))
        extra += ["# 3b. Effect of the prompt type", t3b, "The same comparison for the other parameters - " + "; ".join(other) + ". See Figure 4 and the prompt-type table."]
    ids = sorted(fk["prompt_id"].unique())
    if len(ids) >= 2:
        fp = fk.groupby(["model", "prompt_id"])["strict"].mean()
        t3c = ("Strict fabrication rate per fake-entity prompt over all settings (fake_1 = invented novel, fake_2 = invented theorem, "
               "phrased as if it exists): " + "; ".join(f"{m}: " + ", ".join(f"{i} {fp[(m, i)]:.2f}" for i in ids) for m in (L, Q, G)) + ". ")
        spread = max(abs(fp[(m, ids[0])] - fp[(m, ids[-1])]) for m in (L, Q, G))
        worst = {m: max(ids, key=lambda i: fp[(m, i)]) for m in (L, Q, G) if max(fp[(m, i)] for i in ids) > 0}
        if spread >= 0.3:
            t3c += "The rate depends strongly on the prompt: a model can refuse one invented entity and fabricate details about the other. "
        if len(set(worst.values())) > 1:
            t3c += "The models also differ in which prompt triggers fabrication (" + ", ".join(f"{m}: {i}" for m, i in worst.items()) + "). "
        t3c += "The model comparison therefore holds only for these two prompts."
        extra += ["# 3c. Which invented entity triggers fabrication", t3c]

    P2 = [
        "# 4. Comparison with the expected results",
        s4a, s4b, s4c, s4d, s4e, s4f,
        "# 5. Differences between models",
        s5,
        "# Practical conclusion",
        s_prac,
        "# 6. Limitations",
        s6,
    ]
    _paged_text(pdf, "Discussion", P1 + extra + P2)
    table_page(pdf, "Fake-entity hallucination: lenient vs strict", strict_df)
    if by_category is not None:
        table_page(pdf, "Determinism by prompt type (mean of the models)", category_table(by_category),
                   note="Columns value 1-4 follow the sweep order: " + "; ".join(
                       f"{p} {', '.join(f'{x:g}' for x in sorted(v))}" for p, v in SWEEPS.items()) + ".")
    table_page(pdf, "Fake-entity answers per prompt (all settings)", fake_by_prompt(raw),
               note="lenient = no refusal phrase; strict = lenient plus answers that refuse first but then invent details.")
    return strict_df


def category_table(by_category):
    rows = []
    for param, vals in SWEEPS.items():
        for cat in ["factual", "json", "creative", "fake_entity"]:
            row = {"parameter": param, "prompt type": cat}
            sub = by_category[(by_category.parameter == param) & (by_category.category == cat)]
            for i, x in enumerate(sorted(vals), start=1):
                s = sub[sub.value == x].determinism
                row[f"value {i}"] = round(float(s.mean()), 3) if len(s) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def fake_by_prompt(raw):
    fk, _ = strict_hallucination(raw)
    g = fk.groupby(["model", "prompt_id"])
    out = g[["lenient", "strict"]].mean().round(3)
    out["answers"] = g.size()
    return out.reset_index()


def scoring_revision(raw):
    """Fake-entity fabrication rate under the first and the extended refusal patterns (sensitivity of the scoring)."""
    fk, _ = strict_hallucination(raw)
    txt = fk["response"].fillna("")
    fk["first_pattern"] = (~txt.apply(lambda t: is_refusal(t, REFUSAL_PATTERNS_V1))).astype(int)
    fk["extended_pattern"] = (~txt.apply(is_refusal)).astype(int)
    fk["changed"] = (fk["first_pattern"] != fk["extended_pattern"]).astype(int)
    out = fk.groupby("model").agg(first_pattern=("first_pattern", "mean"), extended_pattern=("extended_pattern", "mean"),
                                  strict=("strict", "mean"), changed=("changed", "sum"), answers=("changed", "size"))
    return out.round(3).reset_index()


def example_answers(raw):
    raw = raw.copy()
    raw["response"] = raw["response"].fillna("")

    def get(m, pid, value, run):
        s = raw[(raw.model == m) & (raw.prompt_id == pid) & (raw.parameter == "temperature")
                & (raw.value == value) & (raw.run == run)]
        return None if s.empty else str(s.iloc[0]["response"])

    fabricated = lambda t: any(re.search(p, t.lower()) for p in FAB_PATTERNS)
    rows = []
    models = list(raw["model"].unique())
    for m in models:
        for pid, value in (("fake_1", 0.0), ("fake_2", 0.0), ("json_1", 1.5)):
            t = get(m, pid, value, 0)
            if t is None:
                continue
            if pid.startswith("fake"):
                if not is_refusal(t):
                    label = "hallucinated (no refusal)"
                elif fabricated(t):
                    label = "refused, then invented details (strict)"
                else:
                    label = "refused"
            else:
                label = "valid JSON" if check_structure(t, ["name", "age", "city"]) else "invalid JSON"
            rows.append(dict(kind="answer", model=m, prompt_id=pid, setting=f"temperature {value:g}", score=label, text=t, text2=""))
    for value in (0.0, 1.5):
        a, b = get(models[0], "creative_1", value, 0), get(models[0], "creative_1", value, 1)
        if a is not None and b is not None:
            rows.append(dict(kind="determinism", model=models[0], prompt_id="creative_1", setting=f"temperature {value:g}",
                             score=f"similarity of repeats 1 and 2 = {SequenceMatcher(None, a, b).ratio():.2f}", text=a, text2=b))
    return pd.DataFrame(rows)


def appendix_pages(pdf, raw):
    table_page(pdf, "Appendix A. Scoring sensitivity: fake-entity fabrication rate", scoring_revision(raw),
               note="first_pattern = original refusal detector (10 phrases); extended_pattern = original plus extra natural refusal "
                    "wordings (e.g. 'there seems to be an error in the title', 'is not a recognized theorem'); strict = extended pattern, "
                    "and answers that refuse first but then invent details also count as fabricated. changed = answers whose score differs "
                    "between the first and the extended pattern. The raw answers are identical in all versions (raw_outputs.csv); only the "
                    "scoring is re-run, so determinism and JSON results do not change. If the patterns are edited after reading the "
                    "answers, this should be stated when the results are presented.")
    table_page(pdf, "Appendix C. Answer style per model", style_stats(raw),
               note="mean_len_* = mean answer length in characters (answers are cut at the token limit); json_in_code_fence = share of "
                    "JSON answers wrapped in a markdown code fence (the checker removes the fence before parsing; a stricter 'raw JSON "
                    "only' checker would count these as failures); empty_answers = answers with no text.")
    cut = lambda t: t.replace("\n", " ")[:260] + ("..." if len(t) > 260 else "")
    paras = ["Excerpts (first 260 characters) from raw_outputs.csv, first repeat. The auto score is the script's own judgement "
             "and may be wrong (see limitations)."]
    for _, r in raw_examples(raw).iterrows():
        paras.append(f"# {r['model']} | {r['prompt_id']} | {r['setting']} | {r['score']}")
        if r["kind"] == "answer":
            paras.append('"' + cut(r["text"]) + '"')
        else:
            paras.append('Repeat 1: "' + cut(r["text"]) + '"  Repeat 2: "' + cut(r["text2"]) + '"')
    _paged_text(pdf, "Appendix B. Example answers", paras)


def style_stats(raw):
    """Answer style per model: length, JSON in code fences, empty answers."""
    r = raw.copy()
    r["response"] = r["response"].fillna("").astype(str)
    r["len"] = r["response"].str.len()
    out = pd.DataFrame({
        "mean_len_creative": r[r.category == "creative"].groupby("model")["len"].mean(),
        "mean_len_fake_entity": r[r.category == "fake_entity"].groupby("model")["len"].mean(),
        "json_in_code_fence": r[r.category == "json"].groupby("model")["response"]
                               .apply(lambda s: s.str.strip().str.startswith("```").mean()),
        "empty_answers": r.groupby("model")["len"].apply(lambda s: int((s == 0).sum())),
        "answers": r.groupby("model").size(),
    })
    return out.round(2).reset_index()


def raw_examples(raw):
    return example_answers(raw)


def build_pdf(summary, figs, path, raw=None, by_category=None):
    with PdfPages(path) as pdf:
        text_page(pdf, "Decoding Parameter & Sampling Analysis", [
            "Goal: measure how Temperature, Top-K and Top-P affect determinism,",
            "hallucination and JSON-structure correctness in locally run LLMs.",
            "",
            "Setup",
            f"  Models: {', '.join(MODELS)} (via Ollama)",
            f"  Prompts: {len(PROMPTS)} (factual, fake-entity, JSON, creative), {N_RUNS} runs per setting",
            f"  Baseline: {BASELINE}",
            f"  Sweeps: {SWEEPS}",
            f"  Max new tokens per answer: {NUM_PREDICT}",
            "",
            "Metrics",
            "  Determinism   = mean pairwise difflib similarity across repeated runs.",
            "  Hallucination = wrong/missing keyword (factual) or no refusal (fake entity).",
            "  Structure     = share of outputs that are valid JSON with all required keys.",
            "",
            "Automatic findings (mean across models)",
            *auto_findings(summary),
        ])
        for param, fig in figs.items():
            pdf.savefig(fig)
        for param in SWEEPS:
            sub = summary[summary["parameter"] == param].sort_values(["model", "value"])
            table_page(pdf, f"Results table - {param}", sub.drop(columns=["parameter"]))
        if raw is not None:
            try:
                discussion_pages(pdf, summary, raw, by_category)
            except Exception as e:  # e.g. --quick run with a reduced sweep
                print(f"[WARN] discussion pages skipped: {e}")
            try:
                appendix_pages(pdf, raw)
            except Exception as e:
                print(f"[WARN] appendix pages skipped: {e}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild tables, charts and report from results/raw_outputs.csv without generating")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run (keeps saved answers with identical prompt text)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUT_DIR, "raw_outputs.csv")

    if args.report_only:
        raw = pd.read_csv(raw_path)
        raw["response"] = raw["response"].fillna("")
    else:
        ensure_server()
        models, sweeps, n_runs = MODELS, SWEEPS, N_RUNS
        if args.quick:
            models = dict(list(MODELS.items())[:1])
            sweeps = {k: v[:2] for k, v in SWEEPS.items()}
            n_runs = 2
            SWEEPS.update(sweeps)
        raw = run_benchmark(models, sweeps, PROMPTS, n_runs, resume=args.resume)
        if raw.empty:
            raise SystemExit("No results - check that the models could be pulled.")

    raw, per_prompt, summary, by_category = analyse(raw, PROMPTS)
    raw.to_csv(raw_path, index=False)
    per_prompt.to_csv(os.path.join(OUT_DIR, "per_prompt_results.csv"), index=False)
    summary.to_csv(os.path.join(OUT_DIR, "summary_results.csv"), index=False)
    by_category.to_csv(os.path.join(OUT_DIR, "results_by_category.csv"), index=False)

    try:
        strict_hallucination(raw)[1].to_csv(
            os.path.join(OUT_DIR, "fake_entity_hallucination_strict.csv"), index=False)
    except Exception as e:
        print(f"[WARN] strict hallucination skipped: {e}")
    try:
        category_table(by_category).to_csv(os.path.join(OUT_DIR, "determinism_by_prompt_type.csv"), index=False)
        fake_by_prompt(raw).to_csv(os.path.join(OUT_DIR, "fake_entity_by_prompt.csv"), index=False)
        scoring_revision(raw).to_csv(os.path.join(OUT_DIR, "scoring_revision.csv"), index=False)
        example_answers(raw).to_csv(os.path.join(OUT_DIR, "example_answers.csv"), index=False)
        style_stats(raw).to_csv(os.path.join(OUT_DIR, "answer_style.csv"), index=False)
    except Exception as e:
        print(f"[WARN] extra tables skipped: {e}")
    figs = make_charts(summary, by_category)
    build_pdf(summary, figs, os.path.join(OUT_DIR, "report.pdf"), raw, by_category)
    print(f"\nDone. Files saved in: {OUT_DIR}")
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
