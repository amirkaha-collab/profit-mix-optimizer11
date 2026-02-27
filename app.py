
import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Page + styling (RTL + dark)
# -----------------------------
st.set_page_config(page_title="מנוע הקצאה – שילובי מסלולים", layout="wide")

DARK_RTL_CSS = """
<style>
/* RTL + dark theme polish */
html, body, [class*="css"]  {
  direction: rtl;
  text-align: right;
}

div[data-testid="stAppViewContainer"]{
  background: #0b0f17;
}

div[data-testid="stHeader"]{
  background: rgba(11,15,23,0.0);
}

div[data-testid="stSidebar"]{
  background: #0a0d14;
  border-left: 1px solid rgba(255,255,255,0.08);
}

h1, h2, h3, h4, h5, h6, p, label, span, div {
  color: rgba(255,255,255,0.92) !important;
}

.stButton button, .stDownloadButton button {
  border-radius: 14px;
}

div[data-testid="stMetric"]{
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  padding: 10px 12px;
  border-radius: 14px;
}

/* Dark tables */
[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.02);
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.08);
  padding: 6px;
}

/* Keep selectboxes dark-ish */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.04) !important;
}

/* Slider color neutrality; keep tooltip inside-ish */
div[data-testid="stSlider"] {
  padding-top: 6px;
}
</style>
"""
st.markdown(DARK_RTL_CSS, unsafe_allow_html=True)


# -----------------------------
# Data loading
# -----------------------------
REQUIRED_COLS = [
    "provider", "fund",
    "stocks", "foreign", "fx", "illiquid",
    "sharpe", "service"
]

def load_data(uploaded) -> pd.DataFrame:
    if uploaded is None:
        # fallback to bundled template in the repo
        return pd.read_excel("template_data.xlsx")

    name = (uploaded.name or "").lower()
    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded)
    elif name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        raise ValueError("פורמט לא נתמך. העלה CSV או Excel.")

    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"חסרות עמודות חובה: {missing}")

    # Keep only required columns, but don't mutate values.
    return df[REQUIRED_COLS].copy()


def load_service_csv(uploaded, providers: List[str]) -> Optional[Dict[str, float]]:
    if uploaded is None:
        return None
    df = pd.read_csv(uploaded)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "provider" not in df.columns or "score" not in df.columns:
        raise ValueError("CSV שירות חייב להכיל עמודות provider,score")
    m = {}
    for _, r in df.iterrows():
        p = str(r["provider"])
        try:
            s = float(r["score"])
        except Exception:
            continue
        m[p] = s
    # Only keep known providers
    m2 = {p: m[p] for p in m.keys() if p in providers}
    return m2 if m2 else None


# -----------------------------
# Portfolio math
# -----------------------------
@dataclass(frozen=True)
class Candidate:
    rows: Tuple[int, ...]           # indices in df
    weights: Tuple[float, ...]      # sum to 1
    provider_set: Tuple[str, ...]   # providers in candidate (unique, sorted)

def portfolio_exposures(df: pd.DataFrame, idxs: Tuple[int, ...], w: Tuple[float, ...]) -> Dict[str, float]:
    cols = ["stocks", "foreign", "fx", "illiquid", "sharpe", "service"]
    vals = {}
    for c in cols:
        vals[c] = float(np.dot(df.loc[list(idxs), c].astype(float).values, np.array(w)))
    # Israel exposure computed as 100 - foreign (per requirement)
    vals["israel"] = 100.0 - vals["foreign"]
    return vals


def score_candidate(expo: Dict[str, float], targets: Dict[str, float], weights_cfg: Dict[str, float],
                    illiquid_max: float, hard_illiquid: bool) -> Tuple[float, Dict[str, float], bool]:
    """
    Returns (score, parts, violated)
    score: lower is better
    parts: contributions for transparency
    violated: true if hard constraint violated
    """
    violated = False
    if hard_illiquid and expo["illiquid"] > illiquid_max:
        violated = True

    parts = {}
    # absolute deviation parts
    for k in ["stocks", "foreign", "fx", "illiquid"]:
        parts[k] = abs(expo[k] - targets[k]) * weights_cfg.get(k, 1.0)

    # Sharpe: we want higher, convert to penalty
    sharpe_pen = max(0.0, targets.get("sharpe_target", 0.0) - expo["sharpe"])
    parts["sharpe"] = sharpe_pen * weights_cfg.get("sharpe", 1.0)

    # Service: want higher, penalty on shortfall vs target
    service_pen = max(0.0, targets.get("service_target", 0.0) - expo["service"])
    parts["service"] = service_pen * weights_cfg.get("service", 1.0)

    total = float(sum(parts.values()))
    # Soft penalty if illiquid exceeds max but not hard filter
    if (not hard_illiquid) and expo["illiquid"] > illiquid_max:
        total += (expo["illiquid"] - illiquid_max) * 10.0  # strong penalty

    return total, parts, violated


def weight_grid_for_k(k: int, step: int) -> Iterable[Tuple[float, ...]]:
    if k == 1:
        yield (1.0,)
        return
    # Generate weights with given step size, sum to 1.0
    steps = int(100 / step)
    # w in {0,step,...,100} but require positive for each leg to avoid duplicates
    for ints in itertools.product(range(1, steps), repeat=k-1):
        # Convert to percentages for first k-1, last is residual
        s = sum(ints)
        last = steps - s
        if last <= 0:
            continue
        w = [i/steps for i in ints] + [last/steps]
        # reorder-invariant duplicates will be filtered later by row combinations; keep as is
        yield tuple(w)


def generate_candidates(df: pd.DataFrame,
                        combo_sizes: List[int],
                        step: int,
                        single_provider_only: bool) -> Iterable[Candidate]:
    n = len(df)
    for k in combo_sizes:
        for idxs in itertools.combinations(range(n), k):
            providers = tuple(sorted(set(df.loc[list(idxs), "provider"].astype(str).tolist())))
            if single_provider_only and len(providers) != 1:
                continue
            for w in weight_grid_for_k(k, step):
                yield Candidate(rows=idxs, weights=w, provider_set=providers)


def describe_candidate(df: pd.DataFrame, cand: Candidate) -> pd.DataFrame:
    rows = []
    for i, wi in zip(cand.rows, cand.weights):
        r = df.loc[i]
        rows.append({
            "גוף": str(r["provider"]),
            "מסלול": str(r["fund"]),
            "משקל": round(wi*100, 2),
            "מניות": r["stocks"],
            "חו״ל": r["foreign"],
            "מט״ח": r["fx"],
            "לא-סחיר": r["illiquid"],
            "שארפ": r["sharpe"],
            "שירות": r["service"],
        })
    return pd.DataFrame(rows)


def pick_top3_distinct(scored: List[Tuple[float, Candidate, Dict[str,float], Dict[str,float]]]) -> List[Tuple[float, Candidate, Dict[str,float], Dict[str,float]]]:
    """
    Enforce: 3 alternatives with different provider sets (no identical set).
    """
    picked = []
    used_sets = set()
    for s, c, expo, parts in scored:
        key = tuple(c.provider_set)
        if key in used_sets:
            continue
        used_sets.add(key)
        picked.append((s, c, expo, parts))
        if len(picked) == 3:
            break
    return picked


def advantage_text(rank: int, score: float, expo: Dict[str,float], targets: Dict[str,float]) -> str:
    if rank == 1:
        return f"הכי קרוב ליעד – סטייה כוללת {score:.2f}"
    if rank == 2:
        return f"שארפ גבוה יותר ({expo['sharpe']:.2f}) תוך סטייה כוללת {score:.2f}"
    return f"ציון שירות משוקלל גבוה ({expo['service']:.2f}) עם סטייה כוללת {score:.2f}"


# -----------------------------
# UI
# -----------------------------
st.title("מנוע הקצאה – שילובי מסלולים (1–3)")

st.caption("העלה אקסל/CSV עם המסלולים, בחר יעד חשיפה, וקבל 3 חלופות. ישראל מחושב כ־100 − חו״ל.")

with st.sidebar:
    st.header("קלט נתונים")
    uploaded = st.file_uploader("קובץ מסלולים (Excel/CSV)", type=["xlsx", "xls", "csv"])
    st.write("אם לא תעלה קובץ – תיטען תבנית הדגמה שמצורפת לפרויקט.")

    st.divider()
    st.header("שירות (אופציונלי)")
    service_csv = st.file_uploader("CSV ציוני שירות (provider,score)", type=["csv"])
    st.caption("אם העלית CSV – הוא גובר על ערכי service בקובץ המסלולים.")

    st.divider()
    st.header("חיפוש חלופות")
    combo_sizes = st.multiselect("מספר מסלולים בחלופה", options=[1,2,3], default=[2,3])
    if not combo_sizes:
        combo_sizes = [2,3]
    step = st.select_slider("גרנולריות משקלים (ל-2/3 מסלולים)", options=[1,2,5,10], value=5)
    single_provider_only = st.checkbox("רק חלופות מאותו גוף מנהל (כל המסלולים באותה חלופה מאותו גוף)", value=False)

    st.divider()
    if st.button("איפוס הגדרות", use_container_width=True):
        st.session_state.clear()
        st.rerun()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

# Apply service overrides
providers = sorted(df["provider"].astype(str).unique().tolist())
try:
    svc_map = load_service_csv(service_csv, providers)
except Exception as e:
    st.error(str(e))
    st.stop()

if svc_map is not None:
    df = df.copy()
    df["service"] = df["provider"].astype(str).map(lambda p: float(svc_map.get(p, df.loc[df["provider"].astype(str)==p, "service"].iloc[0])))

tabs = st.tabs(["הגדרות יעד", "תוצאות (3 חלופות)", "שקיפות חישוב"])

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        tgt_stocks = st.slider("יעד מניות (%)", 0, 100, 60, 1)
        tgt_foreign = st.slider("יעד חו״ל (%)", 0, 100, 50, 1)
    with c2:
        tgt_fx = st.slider("יעד מט״ח (%)", 0, 100, 40, 1)
        tgt_illiquid = st.slider("יעד לא־סחיר (%)", 0, 50, 10, 1)
    with c3:
        illiquid_max = st.slider("מגבלת לא־סחיר מקסימלית (%)", 0, 60, 20, 1)
        hard_illiquid = st.checkbox("מגבלה קשיחה (פסילה מעל המקסימום)", value=True)

    st.subheader("משקולות דירוג")
    wcol1, wcol2, wcol3 = st.columns(3)
    with wcol1:
        w_stocks = st.slider("משקל דיוק מניות", 0.0, 5.0, 2.0, 0.1)
        w_foreign = st.slider("משקל דיוק חו״ל", 0.0, 5.0, 2.0, 0.1)
    with wcol2:
        w_fx = st.slider("משקל דיוק מט״ח", 0.0, 5.0, 1.5, 0.1)
        w_illiquid = st.slider("משקל דיוק לא־סחיר", 0.0, 5.0, 2.5, 0.1)
    with wcol3:
        w_sharpe = st.slider("משקל שארפ", 0.0, 5.0, 1.5, 0.1)
        w_service = st.slider("משקל שירות", 0.0, 5.0, 2.5, 0.1)

    st.caption("שירות משפיע גם על חלופות 1–2 בהתאם למשקל שהגדרת כאן.")

targets = {
    "stocks": float(tgt_stocks),
    "foreign": float(tgt_foreign),
    "fx": float(tgt_fx),
    "illiquid": float(tgt_illiquid),
    "sharpe_target": 0.0,        # use penalty vs 0 by default (so higher is always better)
    "service_target": 10.0,      # treat as "higher is better"; penalty is shortfall vs 10
}
weights_cfg = {
    "stocks": float(w_stocks),
    "foreign": float(w_foreign),
    "fx": float(w_fx),
    "illiquid": float(w_illiquid),
    "sharpe": float(w_sharpe),
    "service": float(w_service),
}

# Run computation
# (Keep it deterministic and "stable": no fast mode)
scored: List[Tuple[float, Candidate, Dict[str,float], Dict[str,float]]] = []
for cand in generate_candidates(df, combo_sizes=combo_sizes, step=step, single_provider_only=single_provider_only):
    expo = portfolio_exposures(df, cand.rows, cand.weights)
    s, parts, violated = score_candidate(expo, targets, weights_cfg, illiquid_max, hard_illiquid)
    if violated:
        continue
    scored.append((s, cand, expo, parts))

scored.sort(key=lambda x: x[0])
top3 = pick_top3_distinct(scored)

with tabs[1]:
    if not top3:
        st.warning("לא נמצאו חלופות שעומדות בתנאים. נסה לרכך מגבלות או לשנות יעד.")
    else:
        # KPI cards
        cols = st.columns(3)
        for i, (s, cand, expo, parts) in enumerate(top3, start=1):
            with cols[i-1]:
                st.markdown(f"### חלופה {i}")
                st.metric("סטייה כוללת (Score)", f"{s:.2f}")
                st.metric("מניות", f"{expo['stocks']:.1f}%")
                st.metric("חו״ל", f"{expo['foreign']:.1f}%")
                st.metric("ישראל (100-חו״ל)", f"{expo['israel']:.1f}%")
                st.metric("מט״ח", f"{expo['fx']:.1f}%")
                st.metric("לא-סחיר", f"{expo['illiquid']:.1f}%")
                st.metric("שארפ", f"{expo['sharpe']:.2f}")
                st.metric("שירות", f"{expo['service']:.2f}")
                st.caption(advantage_text(i, s, expo, targets))

        st.divider()
        # Full table: three alternatives stacked, but one unified table
        rows = []
        for i, (s, cand, expo, parts) in enumerate(top3, start=1):
            desc = describe_candidate(df, cand)
            for _, r in desc.iterrows():
                rows.append({
                    "חלופה": i,
                    "יתרון": advantage_text(i, s, expo, targets),
                    **r.to_dict()
                })
        out = pd.DataFrame(rows)

        # Make "מסלול" wider by placing it early; DataFrame width depends on container
        st.subheader("טבלת חלופות מלאה")
        st.dataframe(out, use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("איך דורג כל פתרון")
    st.write("הציון (Score) הוא סכום סטיות מוחלטות מהיעד בכל רכיב, מוכפלות במשקולות, בתוספת רכיבי שארפ ושירות.")
    st.write("מגבלת לא־סחיר: אם קשיחה – פתרונות מעל המקסימום נפסלים. אם לא – מקבלים קנס חזק.")

    if top3:
        for i, (s, cand, expo, parts) in enumerate(top3, start=1):
            with st.expander(f"חלופה {i} – פירוט"):
                st.write("**שקיפות חישוב – תוצאות משוקללות**")
                st.json({
                    "providers": list(cand.provider_set),
                    "rows": list(cand.rows),
                    "weights": [round(w*100,2) for w in cand.weights],
                    "exposures": {k: round(v,4) for k, v in expo.items()},
                    "score_parts": {k: round(v,4) for k, v in parts.items()},
                    "score_total": round(s,4),
                })
                st.write("**רכיבי החלופה**")
                st.dataframe(describe_candidate(df, cand), use_container_width=True, hide_index=True)


# Footer download helpers
st.divider()
st.download_button(
    "הורד תבנית אקסל לדוגמה",
    data=open("template_data.xlsx", "rb").read(),
    file_name="template_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True
)
