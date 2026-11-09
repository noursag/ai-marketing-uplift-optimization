"""Campaign planner dashboard (restyled). Run from the project root:

    streamlit run app/streamlit_app.py

Reads only files produced by the notebooks: data/processed/all_scores_crossfit.csv (every one of the
64,000 customers with honest, cross-fitted scores; notebook 05) and reports/results/*.json.
Falls back to data/processed/test_scores.csv if notebook 05 was not run.
Nothing is retrained here, so it starts instantly.

Three tabs:
  1. Campaign plan   - budget, cost and margin in; who to contact, with which email, and what it earns
  2. Customer lookup - real customers: profile, scores, recommendation, the reason, and the email text
  3. Evidence        - the evaluation behind the plan (Qini curves, tree rules, when targeting pays)

This version changes the PRESENTATION only (theme, layout, charts, tables). All numbers come from the
same functions and result files as before.
"""
import html
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ARMS, ARM_SHORT, COST_PER_CONTACT, MARGIN_RATE, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation import policy_incremental_at_k, uplift_at_k, uplift_curves, share_of_lift_at_k  # noqa: E402
from src.messaging import describe_customer, draft_message, explain_recommendation  # noqa: E402

# Header text (edit here).
APP_TITLE = "Campaign planner"
APP_SUBTITLE = "Who should get which email, and what it earns. Built on the Hillstrom randomized email experiment."

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ------------------------------------------------------------------ look and feel
INK, MUTED, GRID, PANEL, ACCENT = "#111827", "#6B7280", "#E5E7EB", "#F4F5FA", "#4F46E5"
# One colour per strategy, used on every chart and card.
COL = {"Mail everyone": "#9CA3AF", "Propensity": "#F59E0B", "T-learner": "#0D9488",
       "X-learner": "#0EA5E9", "Uplift tree": "#4F46E5"}
PREFIX = {"Uplift tree": "UT", "T-learner": "T", "X-learner": "X"}

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; padding-left: 2.5rem; padding-right: 2.5rem; max-width: 100%;}
.hero {display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.25rem 2.5rem;
        background: #312E81; border-radius: 12px; padding: 1.5rem 1.9rem; margin: 0 0 1.3rem 0;}
.hero-title {font-size: 2rem; font-weight: 800; letter-spacing: -0.025em; line-height: 1.15; color: #FFFFFF;}
.hero-sub {font-size: 1rem; color: #C7D2FE; margin-top: 0.4rem; max-width: 44rem;}
.hero-stats {display: flex; gap: 2.2rem;}
.hero-stats b {display: block; color: #FFFFFF; font-size: 1.5rem; font-weight: 700; line-height: 1.2;}
.hero-stats span {color: #C7D2FE; font-size: 0.82rem;}
h2, h3 {letter-spacing: -0.01em;}
div[data-testid="stMetricValue"] {font-size: 2rem; font-weight: 700;}
div[data-testid="stMetricLabel"] p {color: #6B7280; font-size: 0.85rem;}
.stTabs [data-baseweb="tab-list"] {gap: 1.75rem;}
.stTabs [data-baseweb="tab"] {font-size: 1.02rem; font-weight: 600; padding-left: 0; padding-right: 0;}
.verdict {border-left: 5px solid #4F46E5; background: #F4F5FA; padding: 1.1rem 1.4rem; border-radius: 8px; margin: 0.4rem 0 1rem 0;}
.verdict .big {font-size: 1.4rem; font-weight: 700; color: #111827; line-height: 1.3;}
.verdict .sub {color: #4B5563; margin: 0.35rem 0 0.7rem 0;}
.pill {display: inline-block; padding: 0.15rem 0.7rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem;}
.pill.warn {background: #FEF3C7; color: #92400E;}
.pill.ok {background: #DCFCE7; color: #166534;}
.dot {display: inline-block; width: 0.75rem; height: 0.75rem; border-radius: 50%; margin-right: 0.5rem;}
.cardtitle {font-weight: 600; font-size: 1rem; margin-bottom: 0.15rem;}
.email {border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden; background: #FFFFFF;}
.email .head {background: #F9FAFB; padding: 0.8rem 1rem; border-bottom: 1px solid #E5E7EB;}
.email .lbl {color: #6B7280; font-size: 0.78rem;}
.email .subj {font-weight: 600; color: #111827;}
.email .body {padding: 1rem 1.1rem; line-height: 1.6; color: #1F2937; font-size: 0.95rem;}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data (unchanged logic)
@st.cache_data
def load():
    f = PROCESSED_DIR / "all_scores_crossfit.csv"
    if f.exists():
        s, source = pd.read_csv(f, index_col="customer_id"), "all 64,000 customers, 5-fold cross-fitted scores (notebook 05)"
    elif (PROCESSED_DIR / "test_scores.csv").exists():
        s = pd.read_csv(PROCESSED_DIR / "test_scores.csv", index_col="customer_id")
        source = f"{len(s):,} held-out test customers (notebook 03)"
    else:
        return None, None, None
    res = {}
    for name in ("03_models", "04_evaluation", "05_robustness"):
        p = RESULTS_DIR / f"{name}.json"
        res[name] = json.loads(p.read_text()) if p.exists() else None
    return s, res, source


S, RES, SOURCE = load()
if S is None:
    st.error("No results found. Run the notebooks first:  python run_all.py")
    st.stop()
M3, V4, R5 = RES["03_models"], RES["04_evaluation"], RES["05_robustness"]
POP = len(S)
Y, ARM = S.visit.to_numpy(), S.arm.to_numpy()
VALUE_DEFAULT = float(V4["economics_visit"]["value"]) if V4 else 7.0
TREE_RULES = {e: pd.DataFrame(r) for e, r in M3["tree_rules"].items()} if M3 and "tree_rules" in M3 else None
SEG = pd.Series(np.select([(S.mens == 1) & (S.womens == 0), (S.mens == 0) & (S.womens == 1)],
                          ["men's only", "women's only"], "both lines"), index=S.index)


def strategies(model: str):
    """score and email per customer for each strategy; 'Mail everyone' and 'Propensity' send the default (men's) email."""
    p = PREFIX[model]
    ones = S.default_arm.to_numpy()
    return {"Mail everyone": (S.random_score.to_numpy(), ones), "Propensity": (S.propensity.to_numpy(), ones),
            model: (S[f"{p}_policy_score"].to_numpy(), S[f"{p}_best_arm"].to_numpy())}


@st.cache_data
def policy_estimate(name: str, model: str, k: float, n_boot: int = 100):
    """Extra visits per targeted customer (+ 95% bootstrap CI) when the strategy contacts its top k."""
    score, parm = strategies(model)[name]
    kk = 1.0 if name == "Mail everyone" else k
    est = policy_incremental_at_k(Y, ARM, parm, score, kk)
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(ARM == a) for a in np.unique(ARM)]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g)) for g in groups])
        vals.append(policy_incremental_at_k(Y[idx], ARM[idx], parm[idx], score[idx], kk))
    return est, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


@st.cache_data
def womens_estimate(name: str, model: str, k: float, n_boot: int = 100):
    """Women's email only (category campaign): uplift in the top k of the women's ranking, vs control."""
    m = np.isin(ARM, [0, 2])
    y, t = Y[m], (ARM[m] == 2).astype(int)
    col = {"Mail everyone": "random_score", "Propensity": "propensity"}.get(name, f"{PREFIX.get(name, 'UT')}_uplift_womens")
    s = S[col].to_numpy()[m]
    kk = 1.0 if name == "Mail everyone" else k
    est = uplift_at_k(y, t, s, kk)
    rng = np.random.default_rng(0)
    it, ic = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    vals = [uplift_at_k(y[i], t[i], s[i], kk) for i in (np.r_[rng.choice(it, len(it)), rng.choice(ic, len(ic))] for _ in range(n_boot))]
    return est, float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


QINI_MODELS = {"Mail everyone": "random_score", "Propensity": "propensity", "T-learner": "T_uplift_{a}", "Uplift tree": "UT_uplift_{a}"}


@st.cache_data
def qini_payload(arm: int):
    """Qini curves and 'effect kept by the best half' for one email (cached: it is the slowest part of the Evidence tab)."""
    sub = S[S.arm.isin([0, arm])]
    y, t = sub.visit.to_numpy(), (sub.arm == arm).astype(int).to_numpy()
    curves, kept = {}, {}
    for name, c in QINI_MODELS.items():
        sc = sub[c.format(a=ARM_SHORT[arm])].to_numpy()
        cur = uplift_curves(y, t, sc)
        curves[name] = (np.asarray(cur.frac), np.asarray(cur.qini) / len(sub))
        kept[name] = 0.5 if name == "Mail everyone" else share_of_lift_at_k(y, t, sc, 0.5)
    end = uplift_curves(y, t, sub.random_score.to_numpy()).qini.iloc[-1] / len(sub)
    return curves, float(end), kept


# ------------------------------------------------------------------ small helpers
def markdown_money_safe(message: str) -> str:
    """Escape currency dollar signs so Streamlit Markdown does not interpret them as math."""
    return message.replace("$", r"\$")


def html_safe(text: str) -> str:
    """Escape text for the custom HTML blocks (and keep $ from being read as math)."""
    return html.escape(text).replace("$", "&#36;")


def dollars(v: float) -> str:
    return f"{'−' if v < 0 else ''}${abs(v):,.0f}"


def money(u, n, value, margin, cost):
    return u * n * value * margin - cost * n


def short_profile(r: pd.Series) -> str:
    bought = {(1, 1): "both lines", (1, 0): "men's", (0, 1): "women's"}[(int(r.mens), int(r.womens))]
    who = "New" if int(r.newbie) == 1 else "Existing"
    return f"{who} · {str(r.zip_code).lower()} · {bought} · ${float(r.history):,.0f} · last bought {int(r.recency)} mo. ago"


def style_axes(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def money_axis(ax):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{'−' if v < 0 else ''}${abs(v):,.0f}"))


# ------------------------------------------------------------------ charts
def plan_chart(plan: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.1))
    fig.patch.set_facecolor("white")
    specs = [
    ("extra visits", "Estimated incremental visits", False),
    ("profit", "Estimated incremental profit", True),
    ]
    for ax, (col, title, is_money) in zip(axes, specs):
        vals = plan[col].to_numpy()
        lo, hi = plan[f"{col} (low)"].to_numpy(), plan[f"{col} (high)"].to_numpy()
        err = [np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)]
        x = np.arange(len(plan))
        ax.bar(x, vals, color=[COL[i] for i in plan.index], width=0.58, yerr=err, capsize=3,
               error_kw=dict(ecolor=INK, lw=1, alpha=0.75))
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" ", "\n") for n in plan.index], fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=10)
        style_axes(ax)
        if is_money:
            money_axis(ax)
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.text(0.995, 0.01, "Bars: estimate. Whiskers: 95% bootstrap interval.", ha="right", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def qini_chart(arm: int):
    curves, end, _ = qini_payload(arm)
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    fig.patch.set_facecolor("white")
    for name, (frac, q) in curves.items():
        ax.plot(frac, q, label=name, color=COL[name], lw=2.4 if name == "Uplift tree" else 1.4,
                alpha=1 if name in ("Uplift tree", "T-learner") else 0.85, zorder=3 if name == "Uplift tree" else 2)
    ax.plot([0, 1], [0, end], ls="--", c=INK, lw=1, label="Random (expected)")
    ax.set_xlabel("Share of customers contacted (best first)", fontsize=9, color=MUTED)
    ax.set_ylabel("Extra visits per customer in the base", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    style_axes(ax, grid="both")
    fig.tight_layout()
    return fig


def decision_chart(dm: pd.DataFrame, break_even: float):
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    fig.patch.set_facecolor("white")
    x = dm.cost_per_contact.to_numpy()
    best, every = dm.best_profit.to_numpy(), dm.everyone_profit.to_numpy()
    ax.fill_between(x, every, best, where=best > every, color=ACCENT, alpha=0.12, lw=0)
    ax.plot(x, best, color=ACCENT, lw=2.4, label="Best targeted plan")
    ax.plot(x, every, color=COL["Mail everyone"], lw=2.2, label="Mail everyone")
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(break_even, color=INK, lw=1, ls="--")
    ax.annotate(f"Mailing everyone breaks even at ${break_even:.2f} per contact", xy=(break_even, ax.get_ylim()[1]),
                xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=INK, va="top")
    ax.set_xlabel("Cost per contact", fontsize=9, color=MUTED)
    ax.set_ylabel("Incremental profit", fontsize=9, color=MUTED)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.2f}"))
    money_axis(ax)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.88))
    style_axes(ax, grid="both")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ header
st.markdown(
    f'<div class="hero"><div class="hero-text"><div class="hero-title">{html_safe(APP_TITLE)}</div>'
    f'<div class="hero-sub">{html_safe(APP_SUBTITLE)}</div></div>'
    f'<div class="hero-stats"><div><b>{POP:,}</b><span>customers</span></div>'
    f'<div><b>3</b><span>random groups</span></div><div><b>Visit</b><span>main outcome</span></div></div></div>',
    unsafe_allow_html=True)
tab_plan, tab_lookup, tab_evidence = st.tabs(["Campaign plan", "Customer lookup", "Evidence"])


# ================================================================== 1. campaign plan
def render_plan():
    ctrl, main = st.columns([1, 3], gap="large")
    with ctrl:
        with st.container(border=True):
            st.markdown("**Campaign**")
            campaign = st.radio("Which campaign?", ["Women's collection launch (one email)", "Spring campaign (both emails)"],
                                label_visibility="collapsed")
            model = st.selectbox("Uplift model", ["Uplift tree", "T-learner", "X-learner"], index=0,
                                 help="Uplift tree = interpretable deployment candidate chosen in notebook 03; T-learner = pre-registered main model.")
        with st.container(border=True):
            st.markdown("**Budget and money assumptions**")
            budget = st.number_input("Budget ($)", 100, 100_000, 5_000, step=250)
            cost = st.slider("Cost per contact ($)", 0.02, 0.60, float(COST_PER_CONTACT), 0.01,
                             help="Plain email ≈ $0.05-0.15; email with a coupon, SMS or a printed piece are far more expensive.")
            value = st.slider("Value of one extra visit ($ spend)", 2.0, 15.0, round(VALUE_DEFAULT, 2), 0.25,
                              help="Default = average spend per visitor in the training data (the visit-value bridge, notebook 04).")
            margin = st.slider("Gross margin (%)", 10, 60, int(MARGIN_RATE * 100), 5) / 100

    n_afford = int(min(POP, budget // cost))
    k = max(n_afford / POP, 0.02)
    womens_only = campaign.startswith("Women")

    names = ["Mail everyone", "Propensity", model]
    rows = []
    for name in names:
        n = POP if name == "Mail everyone" else n_afford
        kk = 1.0 if name == "Mail everyone" else k
        u, lo, hi = womens_estimate(name, model, kk) if womens_only else policy_estimate(name, model, kk)
        rows.append({"strategy": name, "contacts": n, "extra visits": u * n, "extra visits (low)": lo * n, "extra visits (high)": hi * n,
                     "cost": cost * n, "profit": money(u, n, value, margin, cost), "profit (low)": money(lo, n, value, margin, cost),
                     "profit (high)": money(hi, n, value, margin, cost), "cost per extra visit": (cost / u) if u > 0 else np.nan})
    plan = pd.DataFrame(rows).set_index("strategy")

    everyone, targeted, prop = plan.loc["Mail everyone"], plan.loc[model], plan.loc["Propensity"]
    over_budget = everyone.cost > budget
    affordable = plan[plan.cost <= budget + 1e-9]
    best_aff = affordable["profit"].idxmax() if not affordable.empty else None

    with main:
        st.caption(markdown_money_safe(
            f"Budget ${budget:,.0f} · ${cost:.2f} per contact · {n_afford:,} affordable contacts ({k:.0%} of customers) · "
            f"gross profit per extra visit ${value * margin:.2f} (${value:.2f} spend × {margin:.0%} margin)"))

        # ---- the one thing to read first
        if best_aff is None:
            st.markdown(
                '<div class="verdict"><div class="big">No strategy fits this budget</div>'
                '<div class="sub">Lower the cost per contact or raise the budget.</div></div>', unsafe_allow_html=True)
        else:
            b = plan.loc[best_aff]
            others = affordable.drop(index=best_aff)
            clear = len(others) > 0 and b["profit (low)"] > others["profit (high)"].max()
            positive = b["profit (low)"] > 0
            sub = (f"95% interval {dollars(b['profit (low)'])} to {dollars(b['profit (high)'])}. "
                   f"Mailing everyone: {dollars(everyone['profit'])}"
                   + (f" (it would cost {dollars(everyone['cost'])}, over budget)" if over_budget else "") + ".")
            pill_a = ( '<span class="pill ok">Higher interval than the other affordable options</span>' if clear
                      else '<span class="pill warn">Uncertain advantage over alternatives</span>')
            pill_b = ('<span class="pill ok">Profit above zero across the interval</span>' if positive
                      else '<span class="pill warn">Profit could be zero or negative</span>')
            st.markdown(
                f'<div class="verdict"><div class="big">{html_safe(best_aff)} has the highest estimated profit: {html_safe(dollars(b["profit"]))}</div>'
                f'<div class="sub">{html_safe(sub)}</div>{pill_a}{pill_b}</div>', unsafe_allow_html=True)

        # ---- three strategy cards
        cols = st.columns(3)
        for col, name in zip(cols, names):
            r = plan.loc[name]
            with col.container(border=True):
                st.markdown(f'<div class="cardtitle"><span class="dot" style="background:{COL[name]}"></span>{html_safe(name)}</div>',
                            unsafe_allow_html=True)
                st.metric("Estimated profit", dollars(r["profit"]))
                st.caption(markdown_money_safe(
                    f"{r['extra visits']:,.0f} extra visits from {int(r.contacts):,} contacts\n\n"
                    f"95% interval {dollars(r['profit (low)'])} to {dollars(r['profit (high)'])}"
                    + (f"\n\n${r['cost per extra visit']:.2f} per extra visit" if pd.notna(r["cost per extra visit"]) else "")))
                if r["cost"] > budget:
                    st.caption(markdown_money_safe(f":red[Over budget: needs {dollars(r['cost'])}]"))
                else:
                    st.caption("Within budget")  # keeps the three cards the same height

        # ---- chart
        st.pyplot(plan_chart(plan))

        # ---- who is on the list
        st.subheader("Who is on the list")
        score, parm = strategies(model)[model]
        if womens_only:
            score = S[f"{PREFIX[model]}_uplift_womens"].to_numpy()
            parm = np.full(POP, 2)
        top = np.argsort(-score, kind="mergesort")[:n_afford]
        who = pd.DataFrame({"segment": SEG.to_numpy()[top], "email": [ARMS[a] for a in parm[top]]})
        tab = who.groupby(["segment", "email"]).size().unstack(fill_value=0)
        tab["On the list"] = tab.sum(axis=1) / len(who) * 100
        base_share = (SEG.value_counts(normalize=True) * 100).rename("Whole customer base")
        shown = tab.join(base_share).reset_index().rename(columns={"segment": "Past purchases"})
        st.dataframe(
            shown, hide_index=True,
            column_config={
                "On the list": st.column_config.ProgressColumn("Share of the list", min_value=0, max_value=100, format="%.0f%%"),
                "Whole customer base": st.column_config.ProgressColumn("Share of the base", min_value=0, max_value=100, format="%.0f%%"),
            })
        st.caption("Segments reflect past merchandise purchases. They describe averages and do not predict any one customer's response.")

        with st.expander("How to read this plan"):
            if best_aff is None:
                st.write("No displayed strategy fits the current budget. Consider reducing the contact cost or adjusting the budget.")
            else:
                retained = ""
                if womens_only:
                    base_u = womens_estimate("Mail everyone", model, 1.0)[0]
                    lift_kept = targeted["extra visits"] / (base_u * POP) if base_u > 0 else np.nan
                    if np.isfinite(lift_kept):
                        retained = (f"The {model} ranking keeps an estimated {lift_kept:.0%} of the full-list extra visits with {k:.0%} of the contacts. "
                                    "The experiment suggests that women's-email effects differ by purchase history, with stronger responses among past women's-merchandise buyers. ")
                else:
                    retained = ("The men's email has a broad average effect, but the evidence does not establish that one targeting policy is "
                                "consistently more profitable. ")
                st.markdown(markdown_money_safe(
                    f"{retained}The {model} strategy is estimated at {dollars(targeted['profit'])}, propensity at {dollars(prop['profit'])}, "
                    f"mailing everyone at {dollars(everyone['profit'])}. 'Highest estimated profit' compares point estimates only; "
                    "overlapping intervals do not establish a reliable winner. This is an exploratory planning scenario."))
            st.markdown(
                "Extra visits are estimated from randomized experimental outcomes inside the selected groups. Budget, contact cost, value per visit "
                "and margin are **assumptions**: the dataset contains no campaign costs. Intervals come from the bootstrap shown and are not guarantees.")


# ================================================================== 2. customer lookup
def render_lookup():
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1, 1.6, 1.3, 1])
        model = f1.selectbox("Uplift model", ["T-learner", "Uplift tree", "X-learner"], index=0, key="lk_model")
        seg_pick = f2.multiselect("Past purchases", ["men's only", "women's only", "both lines"],
                                  default=["men's only", "women's only", "both lines"], key="lk_seg")
        newbie_pick = f3.selectbox("Customer type", ["all", "new (first purchase this year)", "existing"], key="lk_new")
        sample_n = f4.slider("Customers shown", 5, 40, 12, key="lk_n")
    p = PREFIX[model]

    pool = S[SEG.isin(seg_pick)]
    if newbie_pick.startswith("new"):
        pool = pool[pool.newbie == 1]
    elif newbie_pick == "existing":
        pool = pool[pool.newbie == 0]
    if pool.empty:
        st.info("No customers match these filters. Widen the filters above.")
        return
    sample = pool.sample(min(sample_n, len(pool)), random_state=7).sort_values(f"{p}_policy_score", ascending=False)

    show = pd.DataFrame({
        "Customer": [f"#{i}" for i in sample.index],
        "Profile": [short_profile(r) for _, r in sample.iterrows()],
        "Men's email": (sample[f"{p}_uplift_mens"] * 100).to_numpy(),
        "Women's email": (sample[f"{p}_uplift_womens"] * 100).to_numpy(),
        "Recommendation": sample[f"{p}_action"].str.replace(" E-Mail", " email").to_numpy()})

    def highlight(row):
        css = [""] * len(row)
        pick = {"Mens email": "Men's email", "Womens email": "Women's email"}.get(row["Recommendation"])
        if pick:
            css[list(row.index).index(pick)] = "background-color: #DCFCE7; font-weight: 600"
        return css

    styled = (show.style.apply(highlight, axis=1)
              .map(lambda v: "color: #B91C1C" if v < 0 else "", subset=["Men's email", "Women's email"])
              .format({"Men's email": "{:+.1f} pts", "Women's email": "{:+.1f} pts"}))
    st.dataframe(styled, hide_index=True, height=min(440, 38 * (len(show) + 1)),
                 column_config={"Profile": st.column_config.TextColumn(width="large")})
    st.caption("Points = estimated change in the chance of a visit if the customer gets that email. A model estimate for customers like this one, "
               "not a fact about the person. Green = the recommended email.")

    st.subheader("Open one customer")
    cid = st.selectbox("Customer", sample.index.tolist(), format_func=lambda i: f"#{i} · {short_profile(S.loc[i])}", label_visibility="collapsed")
    r = S.loc[cid]
    a, b = st.columns(2, gap="large")
    with a:
        with st.container(border=True):
            st.markdown(f"#### Customer #{cid}")
            st.write(describe_customer(r))
            c1, c2, c3 = st.columns(3)
            c1.metric("Men's email", f"{r[f'{p}_uplift_mens'] * 100:+.1f} pts")
            c2.metric("Women's email", f"{r[f'{p}_uplift_womens'] * 100:+.1f} pts")
            c3.metric("Recommended", r[f"{p}_action"].replace(" E-Mail", " email"))
            st.markdown("**Why.** " + explain_recommendation(r, TREE_RULES, score_prefix=p))
            st.caption(markdown_money_safe(
                f"In the experiment this record was randomly assigned to {r.segment} and "
                f"{'visited' if r.visit == 1 else 'did not visit'} the site in the following two weeks"
                + (f", with ${r.spend:,.2f} in spending." if r.spend > 0 else ".")))
            with st.expander("All models for this customer"):
                st.table(pd.DataFrame({m: [f"{r[f'{PREFIX[m]}_uplift_mens'] * 100:+.1f} pts", f"{r[f'{PREFIX[m]}_uplift_womens'] * 100:+.1f} pts",
                                           r[f"{PREFIX[m]}_action"]] for m in PREFIX if f"{PREFIX[m]}_action" in S},
                                     index=["men's email", "women's email", "recommendation"]))
    with b:
        st.markdown("#### The email this customer would get")
        use_llm = st.toggle("Write the copy with an LLM (needs ANTHROPIC_API_KEY)", value=False,
                            disabled=not os.environ.get("ANTHROPIC_API_KEY"))
        msg = draft_message(r, r[f"{p}_action"], llm=use_llm)
        body = html_safe(msg["body"]).replace("\n", "<br>")
        st.markdown(
            f'<div class="email"><div class="head"><div class="lbl">Subject</div><div class="subj">{html_safe(msg["subject"])}</div></div>'
            f'<div class="body">{body}</div></div>', unsafe_allow_html=True)
        st.caption(f"Source: {msg['source']}. The uplift model decides who gets which version; the template or the LLM only writes the words.")


# ================================================================== 3. evidence
def render_evidence():
    st.markdown("For the **women's email**, uplift models rank customers clearly better than propensity. "
                "For the **men's email**, the evidence shows no consistent advantage over simpler rankings.")

    st.subheader("Does the ranking find the customers who respond?")
    cols = st.columns(2, gap="large")
    for col, (arm, label) in zip(cols, ((2, "Women's email"), (1, "Men's email"))):
        with col.container(border=True):
            st.markdown(f"**{label}** vs no email")
            st.pyplot(qini_chart(arm))
            kept = qini_payload(arm)[2]
            st.caption("Share of the total effect kept by contacting the best half: "
                       + " · ".join(f"{n} {v:.0%}" for n, v in kept.items()) + ".")
    st.caption("The further a curve sits above the dashed line, the better the ranking. On the women's chart the uplift curves rise above "
               "propensity and random; on the men's chart no curve separates clearly.")

    st.subheader("The rules the honest uplift tree learned")
    if TREE_RULES:
        c1, c2 = st.columns(2, gap="large")
        for col, (email, rules) in zip((c2, c1), TREE_RULES.items()):
            with col.container(border=True):
                st.markdown(f"**{email}**")
                st.table(rules.assign(uplift=(rules.uplift * 100).round(1).map(lambda v: f"{v:+.1f} pts"), n=rules.n.astype(int))
                         .rename(columns={"n": "customers (estimation half)"}))
        st.caption("A split is kept only if the two sides' uplifts differ by at least 2 standard errors; leaf values are estimated on a separate half "
                   "of the data (honesty). For the men's email the leaf effects are similar, so read that split with caution.")

    st.subheader("Ranking quality with confidence intervals")
    src = R5 if R5 else V4
    rk = pd.DataFrame(src["ranking"])
    rk = rk.assign(**{"Qini (95% CI)": rk.apply(lambda r: f"{r.qini:.4f} [{r.qini_lo:.4f}, {r.qini_hi:.4f}]", axis=1)})
    extra = ["share_of_lift@50%"] if "share_of_lift@50%" in rk else []
    tbl = rk[["email", "model", "Qini (95% CI)"] + extra].copy()
    if extra:
        tbl["effect kept with top 50%"] = (rk["share_of_lift@50%"] * 100).round(0).astype(int).astype(str) + "%"
        tbl = tbl.drop(columns=extra)
    st.dataframe(tbl.rename(columns={"email": "Email", "model": "Model"}), hide_index=True)
    st.caption(("5-fold cross-fit on all 64,000 customers (notebook 05). " if R5 else "Held-out test set (notebook 04). ")
               + "Qini measures ranking quality relative to random targeting; 0 means no ranking gain.")

    if V4 and "economics_visit" in V4:
        st.subheader("When does targeting pay?")
        dm = pd.DataFrame(V4["economics_visit"]["decision_map"])
        dm = dm[dm.cost_per_contact <= 0.3]
        e = V4["economics_visit"]["everyone"]
        st.pyplot(decision_chart(dm, float(e["break_even_cost"])))
        st.caption(f"Scenario assumptions: base of 64,000, margin {MARGIN_RATE:.0%}, value per visit from the data. Cheap emails: mailing everyone is already "
                   "close to the best plan. Dearer contacts: targeting protects the profit. Confidence intervals are wide, so these thresholds are exploratory.")
        with st.expander("See the numbers"):
            table = pd.DataFrame({"Cost per contact": dm.cost_per_contact.map("${:.2f}".format),
                                  "Best plan": [f"{s} · top {k:.0%}" if s != "No campaign" else "do not send" for s, k in zip(dm.best_strategy, dm.best_k)],
                                  "Profit of best plan": dm.best_profit.map(dollars), "Profit of mailing everyone": dm.everyone_profit.map(dollars)})
            st.dataframe(table, hide_index=True)


with tab_plan:
    render_plan()
with tab_lookup:
    render_lookup()
with tab_evidence:
    render_evidence()

st.caption(f"Customer base: {SOURCE}. Per-customer effects are model estimates checked against the randomized experiment with held-out or "
           "out-of-fold predictions; they are not directly observed facts.")