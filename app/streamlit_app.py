"""Campaign planner dashboard: a real campaign on the real customer base. Run from the project root:

    streamlit run app/streamlit_app.py

Reads only files produced by the notebooks: data/processed/all_scores_crossfit.csv (every one of the
64,000 customers with honest, cross-fitted scores; notebook 05) and reports/results/*.json.
Falls back to data/processed/test_scores.csv (19,200 held-out customers) if notebook 05 was not run.
Nothing is retrained here, so it starts instantly.

Three views:
  1. Campaign plan   - budget, cost and margin in; who to contact, with which email, and what it earns out
  2. Customer lookup - real customers: profile, scores, recommendation, the reason, and the email text
  3. Evidence        - the held-out evaluation behind the plan (Qini, tree rules, decision map)
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ARMS, ARM_SHORT, COST_PER_CONTACT, MARGIN_RATE, PROCESSED_DIR, RESULTS_DIR  # noqa: E402
from src.evaluation import policy_incremental_at_k, uplift_at_k, uplift_curves, share_of_lift_at_k  # noqa: E402
from src.messaging import describe_customer, draft_message, explain_recommendation  # noqa: E402

st.set_page_config(page_title="Campaign planner · uplift targeting", layout="wide")
COL = {"Mail everyone": "#9CA3AF", "Propensity": "#D97706", "Uplift tree": "#DC2626", "T-learner": "#0F766E", "X-learner": "#7C3AED"}
PREFIX = {"Uplift tree": "UT", "T-learner": "T", "X-learner": "X"}


@st.cache_data
def load():
    f = PROCESSED_DIR / "all_scores_crossfit.csv"
    if f.exists():
        s, source = pd.read_csv(f, index_col="customer_id"), "all 64,000 customers, 5-fold cross-fitted scores (notebook 05)"
    elif (PROCESSED_DIR / "test_scores.csv").exists():
        s, source = pd.read_csv(PROCESSED_DIR / "test_scores.csv", index_col="customer_id"), f"{len(s):,} held-out test customers (notebook 03)"
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
SEG = pd.Series(np.select([(S.mens == 1) & (S.womens == 0), (S.mens == 0) & (S.womens == 1)], ["men's only", "women's only"], "both lines"), index=S.index)


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


def markdown_money_safe(message: str) -> str:
    """Escape currency dollar signs so Streamlit Markdown does not interpret them as math."""
    return message.replace("$", r"\$")


def money(u, n, value, margin, cost):
    return u * n * value * margin - cost * n


# ================================================================== sidebar
st.sidebar.title("Campaign planner")
view = st.sidebar.radio("View", ["1 · Campaign plan", "2 · Customer lookup", "3 · Evidence"])
st.sidebar.caption(f"Customer base: {SOURCE}. Every per-customer effect below is measured on the real Hillstrom experiment "
                   "using held-out or out-of-fold predictions; individual effects are estimates, not directly observed facts.")

# ================================================================== 1. campaign plan
if view.startswith("1"):
    st.header("Campaign plan: who gets which email, and what it earns")
    st.markdown("**The situation.** The retailer has a list of 64,000 past customers and two ready-made emails: a *men's* and a *women's* "
                "merchandise promotion. Marketing has a budget and wants to know **whom to contact, with which version, and whether "
                "targeting beats mailing everyone**. The experiment behind this dashboard tested exactly that on these customers.")
    with st.sidebar:
        st.subheader("Campaign")
        campaign = st.radio("Which campaign?", ["Women's collection launch (one email: who should get it?)", "Spring campaign (both versions: who gets which?)"])
        model = st.selectbox("Uplift model for targeting", ["Uplift tree", "T-learner", "X-learner"], index=0,
                             help="Uplift tree = interpretable deployment candidate chosen in notebook 03; T-learner = pre-registered main model.")
        st.subheader("Budget and money assumptions")
        budget = st.number_input("Budget ($)", 100, 100_000, 5_000, step=250)
        cost = st.slider("Cost per contact ($)", 0.02, 0.60, float(COST_PER_CONTACT), 0.01,
                         help="Plain email ≈ $0.05-0.15; email with a coupon, SMS or a printed piece are far more expensive.")
        value = st.slider("Value of one extra visit ($ spend)", 2.0, 15.0, round(VALUE_DEFAULT, 2), 0.25,
                          help="Default = average spend per visitor in the training data (the visit-value bridge, notebook 04).")
        margin = st.slider("Gross margin (%)", 10, 60, int(MARGIN_RATE * 100), 5) / 100
    n_afford = int(min(POP, budget // cost))
    k = max(n_afford / POP, 0.02)
    womens_only = campaign.startswith("Women")  # first option
    st.caption(markdown_money_safe(
        f"Budget: ${budget:,.0f} | Cost per contact: ${cost:.2f} | "
        f"Affordable contacts: {n_afford:,} ({k:.0%} of customers) | "
        f"Assumed gross profit per incremental visit: ${value * margin:.2f} "
        f"(${value:.2f} spend × {margin:.0%} margin)."
    ))

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
    best = plan["profit"].idxmax()

    everyone, targeted, prop = plan.loc["Mail everyone"], plan.loc[model], plan.loc["Propensity"]
    over_budget = everyone.cost > budget
    affordable = plan[plan.cost <= budget + 1e-9]
    best_aff = affordable["profit"].idxmax() if not affordable.empty else None

    def dollars(v):
        return f"{'−' if v < 0 else ''}${abs(v):,.0f}"

    c = st.columns(3)
    for col, name in zip(c, names):
        r = plan.loc[name]
        label = name + (" · highest point estimate" if name == best_aff else "")
        col.metric(label, dollars(r["profit"]),
                   f"{r['extra visits']:,.0f} estimated extra visits · {int(r.contacts):,} contacts",
                   delta_color="off")
        interval = (f"95% bootstrap interval: {dollars(r['profit (low)'])} to "
                    f"{dollars(r['profit (high)'])}")
        if pd.notna(r["cost per extra visit"]):
            interval += f" · ${r['cost per extra visit']:.2f} per estimated extra visit"
        col.caption(markdown_money_safe(interval))
        if r["cost"] > budget:
            col.caption(markdown_money_safe(f"Over budget: requires {dollars(r['cost'])}."))

    if best_aff is None:
        verdict = ("No displayed strategy fits the current budget. "
                   "Consider reducing contact cost or adjusting the campaign budget.")
    elif womens_only:
        base_u = womens_estimate("Mail everyone", model, 1.0)[0]
        lift_kept = targeted["extra visits"] / (base_u * POP) if base_u > 0 else np.nan
        retained = (f"The {model} ranking retains an estimated {lift_kept:.0%} of the "
                    f"full-list incremental visits with {k:.0%} of the contacts. "
                    if np.isfinite(lift_kept) else "")
        verdict = (
            f"**Women's-email campaign:** Among the affordable options shown, "
            f"**{best_aff}** has the highest estimated profit ({dollars(plan.loc[best_aff, 'profit'])}). "
            f"The {model} strategy is estimated at {dollars(targeted['profit'])}; "
            f"propensity at {dollars(prop['profit'])}; and mailing everyone at "
            f"{dollars(everyone['profit'])}"
            + (f" (but mailing everyone costs {dollars(everyone['cost'])}, above budget)" if over_budget else "")
            + ". " + retained
            + "The experiment suggests that women's-email effects differ by purchase history, "
            "with stronger responses among past women's-merchandise buyers. "
            "These are scenario estimates, not guaranteed profits; compare the uncertainty intervals above."
        )
    else:
        verdict = (
            f"**Two-email campaign:** Among the affordable options shown, **{best_aff}** "
            f"has the highest estimated profit ({dollars(plan.loc[best_aff, 'profit'])}). "
            f"The {model} strategy is estimated at {dollars(targeted['profit'])}; "
            f"propensity at {dollars(prop['profit'])}; and mailing everyone at "
            f"{dollars(everyone['profit'])}"
            + (f" (requiring {dollars(everyone['cost'])}, above budget)" if over_budget else "")
            + ". The men's email has a broad average effect, but the evidence does not "
            "establish that one targeting policy is consistently more profitable. "
            "These results depend on the chosen costs, visit value and margin."
        )
    st.markdown(markdown_money_safe(verdict))
    st.caption("'Highest estimated profit' compares point estimates only. Overlapping confidence "
               "intervals do not establish a reliable winner; this is an exploratory planning scenario.")

    left, right = st.columns([3, 2])
    with left:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3))
        axes[0].bar(plan.index, plan["extra visits"], color=[COL[i] for i in plan.index],
                    yerr=[plan["extra visits"] - plan["extra visits (low)"], plan["extra visits (high)"] - plan["extra visits"]], capsize=4)
        axes[0].set_title("Extra visits caused by the campaign (95% range)")
        axes[1].bar(plan.index, plan["profit"], color=[COL[i] for i in plan.index],
                    yerr=[plan["profit"] - plan["profit (low)"], plan["profit (high)"] - plan["profit"]], capsize=4)
        axes[1].axhline(0, c="black", lw=.8); axes[1].set_title("Incremental profit ($)")
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False); ax.tick_params(axis="x", labelsize=8)
        plt.tight_layout(); st.pyplot(fig)
    with right:
        st.markdown("**Who is on the list** (top of the ranking vs the whole base)")
        score, parm = strategies(model)[model]
        if womens_only:
            score = S[f"{PREFIX[model]}_uplift_womens"].to_numpy(); parm = np.full(POP, 2)
        top = np.argsort(-score, kind="mergesort")[:n_afford]
        who = pd.DataFrame({"segment": SEG.to_numpy()[top], "email": [ARMS[a] for a in parm[top]]})
        tab = who.groupby(["segment", "email"]).size().unstack(fill_value=0)
        tab["share of list"] = tab.sum(axis=1) / len(who)
        base_share = SEG.value_counts(normalize=True).rename("share of base")
        st.dataframe(tab.join(base_share).style.format({"share of list": "{:.0%}", "share of base": "{:.0%}"}))
        st.caption("Segments reflect past merchandise purchases. On average, the women's email shows stronger estimated effects among past women's-merchandise buyers, while the men's email has a broad positive average effect. These patterns do not predict every individual response.")
    st.info("Incremental visits are estimated from randomized experimental outcomes within selected groups. "
            "Budget, contact cost, value per visit and margin are assumptions; the dataset does not contain campaign costs. "
            "Profit intervals reflect the displayed bootstrap procedure and should not be treated as guaranteed outcomes.")

# ================================================================== 2. customer lookup
elif view.startswith("2"):
    st.header("Customer lookup: a dataset record, the recommendation, and why")
    model = st.sidebar.selectbox("Uplift model", ["T-learner", "Uplift tree", "X-learner"], index=0)
    p = PREFIX[model]
    st.sidebar.subheader("Find customers")
    seg_pick = st.sidebar.multiselect("Past purchases", ["men's only", "women's only", "both lines"], default=["men's only", "women's only", "both lines"])
    newbie_pick = st.sidebar.selectbox("Customer type", ["all", "new (first purchase this year)", "existing"])
    sample_n = st.sidebar.slider("Customers shown", 5, 40, 12)
    pool = S[SEG.isin(seg_pick)]
    if newbie_pick.startswith("new"):
        pool = pool[pool.newbie == 1]
    elif newbie_pick == "existing":
        pool = pool[pool.newbie == 0]
    sample = pool.sample(min(sample_n, len(pool)), random_state=7).sort_values(f"{p}_policy_score", ascending=False)
    show = pd.DataFrame({
        "customer": [f"#{i}" for i in sample.index], "who": [describe_customer(r) for _, r in sample.iterrows()],
        "men's email": (sample[f"{p}_uplift_mens"] * 100).round(1).map(lambda v: f"{v:+.1f} pts"),
        "women's email": (sample[f"{p}_uplift_womens"] * 100).round(1).map(lambda v: f"{v:+.1f} pts"),
        "recommendation": sample[f"{p}_action"].to_numpy()}, index=sample.index)
    st.dataframe(show.set_index("customer"), use_container_width=True, height=min(420, 38 * (len(show) + 1)))
    st.caption("Points = estimated change in the chance of a visit if this customer receives that email (a model estimate for customers like this one, not a fact about the person).")

    cid = st.selectbox("Open a customer", sample.index.tolist(), format_func=lambda i: f"#{i} · {describe_customer(S.loc[i])[:70]}…")
    r = S.loc[cid]
    a, b = st.columns([1, 1])
    with a:
        st.subheader(f"Customer #{cid}")
        st.write(describe_customer(r))
        c1, c2, c3 = st.columns(3)
        c1.metric("Men's email", f"{r[f'{p}_uplift_mens'] * 100:+.1f} pts")
        c2.metric("Women's email", f"{r[f'{p}_uplift_womens'] * 100:+.1f} pts")
        c3.metric("Recommended", r[f"{p}_action"].replace(" E-Mail", " email"))
        st.markdown("**Why:** " + explain_recommendation(r, TREE_RULES, score_prefix=p))
        st.markdown(markdown_money_safe(
            f"**Observed outcome for this dataset record:** randomly assigned to *{r.segment}*; "
            f"{'visited' if r.visit == 1 else 'did not visit'} the site in the following two weeks"
            + (f", with ${r.spend:,.2f} in spending." if r.spend > 0 else ".")
        ))
        with st.expander("All models for this customer"):
            st.table(pd.DataFrame({m: [f"{r[f'{PREFIX[m]}_uplift_mens'] * 100:+.1f} pts", f"{r[f'{PREFIX[m]}_uplift_womens'] * 100:+.1f} pts", r[f"{PREFIX[m]}_action"]]
                                   for m in PREFIX if f"{PREFIX[m]}_action" in S}, index=["men's email", "women's email", "recommendation"]))
    with b:
        st.subheader("The email this customer would get")
        use_llm = st.toggle("Generate the copy with an LLM (needs ANTHROPIC_API_KEY in the environment)", value=False,
                            disabled=not os.environ.get("ANTHROPIC_API_KEY"))
        msg = draft_message(r, r[f"{p}_action"], llm=use_llm)
        st.text_input("Subject", msg["subject"], disabled=True)
        st.text_area("Body", msg["body"], height=230, disabled=True)
        st.caption(f"Source: {msg['source']}. The copy is personalised on the same features the model uses (purchase history, recency, "
                  "channel, customer type); the *decision* who gets which version comes from the uplift model, the *words* from the template or the LLM.")

# ================================================================== 3. evidence
else:
    st.header("Evidence: why the plan is built this way")
    st.markdown("These charts use out-of-fold predictions for the customer base (or held-out results if cross-fitted scores are unavailable). "
                "For the women's email, uplift models show better ranking than propensity in the reported analysis. "
                "For the men's email, the evidence does not show a consistent uplift-model advantage over simpler rankings.")
    models = {"Mail everyone": "random_score", "Propensity": "propensity", "T-learner": "T_uplift_{a}", "Uplift tree": "UT_uplift_{a}"}
    cols = st.columns(2)
    for col, arm in zip(cols, (2, 1)):
        sub = S[S.arm.isin([0, arm])]
        y, t = sub.visit.to_numpy(), (sub.arm == arm).astype(int).to_numpy()
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for name, c in models.items():
            cur = uplift_curves(y, t, sub[c.format(a=ARM_SHORT[arm])].to_numpy())
            ax.plot(cur.frac, cur.qini / len(sub), label=name, color=COL[name], lw=2 if name == "Uplift tree" else 1.3)
        end = uplift_curves(y, t, sub.random_score.to_numpy()).qini.iloc[-1] / len(sub)
        ax.plot([0, 1], [0, end], ls="--", c="black", lw=1, label="random (expected)")
        ax.set_title(f"Qini curve: {ARMS[arm]} vs no email"); ax.set_xlabel("share of customers contacted"); ax.set_ylabel("extra visits per customer in base")
        ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
        col.pyplot(fig)
        kept = {name: (0.5 if name == "Mail everyone" else share_of_lift_at_k(y, t, sub[c.format(a=ARM_SHORT[arm])].to_numpy(), 0.5)) for name, c in models.items()}
        col.caption("Share of the email's total effect kept when contacting the best half: " + ", ".join(f"{n} {v:.0%}" for n, v in kept.items()) + ".")

    st.subheader("The rules the honest uplift tree learned (training data)")
    if TREE_RULES:
        c1, c2 = st.columns(2)
        for col, (email, rules) in zip((c2, c1), TREE_RULES.items()):
            col.markdown(f"**{email}**")
            col.table(rules.assign(uplift=(rules.uplift * 100).round(1).map(lambda v: f"{v:+.1f} pts"), n=rules.n.astype(int)).rename(columns={"n": "customers (estimation half)"}))
        st.caption("A split is kept only if the two sides' uplifts differ by at least 2 standard errors; leaf values are estimated on a separate half of the data (honesty). "
                   "For the men's email, the displayed leaf effects are relatively similar; interpret the split cautiously.")

    st.subheader("Ranking quality with confidence intervals")
    src = R5 if R5 else V4
    rk = pd.DataFrame(src["ranking"])
    rk = rk.assign(**{"Qini (95% CI)": rk.apply(lambda r: f"{r.qini:.4f} [{r.qini_lo:.4f}, {r.qini_hi:.4f}]", axis=1)})
    extra = ["share_of_lift@50%"] if "share_of_lift@50%" in rk else []
    tbl = rk[["email", "model", "Qini (95% CI)"] + extra].copy()
    if extra:
        tbl["effect kept with top 50%"] = (rk["share_of_lift@50%"] * 100).round(0).astype(int).astype(str) + "%"
        tbl = tbl.drop(columns=extra)
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.caption(("5-fold cross-fit on all 64,000 customers (notebook 05). " if R5 else "Held-out test set (notebook 04). ")
               + "Qini measures ranking quality relative to random targeting; 0 indicates no ranking gain. "
               "For the women's email, uplift rankings outperform propensity in the reported comparisons. "
               "For the men's email, propensity performs better in the cross-fitted ranking results. "
               "Individual model differences and uncertainty should be checked before choosing a deployment policy.")

    if V4 and "economics_visit" in V4:
        st.subheader("When does targeting pay? (decision map, scenario assumptions)")
        dm = pd.DataFrame(V4["economics_visit"]["decision_map"])
        dm = dm[dm.cost_per_contact <= 0.3]
        show = pd.DataFrame({"cost per contact": dm.cost_per_contact.map("${:.2f}".format),
                             "best plan": [f"{s} · top {k:.0%}" if s != "No campaign" else "do not send" for s, k in zip(dm.best_strategy, dm.best_k)],
                             "profit of best plan": dm.best_profit.map("${:,.0f}".format), "profit of mailing everyone": dm.everyone_profit.map("${:,.0f}".format)})
        st.dataframe(show, hide_index=True, use_container_width=True)
        e = V4["economics_visit"]["everyone"]
        st.caption(f"Visit-value bridge, base of 64,000, margin {MARGIN_RATE:.0%}: mailing everyone breaks even at ${e['break_even_cost']:.3f} per contact. "
                   "The preferred strategy depends on the assumed contact cost and the evaluated targeting fractions. "
                   "For the women's-only campaign, targeting can improve estimated economics, but confidence intervals are wide; "
                   "a positive point estimate does not establish reliable profitability. These threshold choices are exploratory.")
