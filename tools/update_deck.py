"""Refresh the numbers and claims in 'Uplift Promotion Targeting - Presentation.pptx' from reports/results/*.json.

Run from the project root after run_all.py:  python tools/update_deck.py
Edits text in place (keeping the deck's formatting), updates the native charts, and adds the uplift-tree rows.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.util import Pt

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "Uplift Promotion Targeting - Presentation.pptx"
RES = ROOT / "reports" / "results"
M = json.loads((RES / "03_models.json").read_text())
V = json.loads((RES / "04_evaluation.json").read_text())
R = json.loads((RES / "05_robustness.json").read_text())
W, Mn = "Womens E-Mail", "Mens E-Mail"
rank = {(r["email"], r["model"]): r for r in V["ranking"]}
paired = {(r["email"], r["comparison"]): r for r in V["paired"]}
lift = {(r["email"], r["model"], r["k"]): r for r in V["lift_captured"]}
leaves = V["tree_leaves_womens_test"]
rules_w = M["tree_rules"][W]
ev = V["economics_visit"]
dmap = {round(r["cost_per_contact"], 3): r for r in ev["decision_map"]}
wcamp = {(r["strategy"], r["share"]): r for r in ev["womens_campaign"]}
ver = {r["k"]: r for r in V["version_choice"]}
pol = {(r["strategy"], r["k"]): r for r in V["policy_ci"]}
pcmp = {(r["outcome"], r["k"], r["comparison"]): r for r in V["policy_cmp"]}
rrank = {(r["email"], r["model"]): r for r in R["ranking"]}
rpol = {(r["strategy"], r["k"]): r for r in R["policy"]}
rcmp = {(r["k"], r["comparison"]): r for r in R["policy_cmp"]}
pts = lambda v, d=1: f"{v * 100:+.{d}f} pts"
usd = lambda v, d=0: ("-" if v < 0 else "") + f"${abs(v):,.{d}f}"

prs = Presentation(DECK)


def shape(slide_no, name):
    for sh in prs.slides[slide_no - 1].shapes:
        if sh.name == name:
            return sh
    raise KeyError((slide_no, name))


def set_paragraphs(sh, texts):
    """Replace paragraph texts in order, keeping each paragraph's first-run formatting.
    Extra texts are added by cloning the last paragraph; extra paragraphs are removed."""
    tf = sh.text_frame
    paras = [p for p in tf.paragraphs]
    while len(paras) < len(texts):
        new = copy.deepcopy(paras[-1]._p)
        paras[-1]._p.addnext(new)
        paras = [p for p in tf.paragraphs]
    for p, t in zip(paras, texts):
        runs = p.runs
        if runs:
            runs[0].text = t
            for r in runs[1:]:
                r._r.getparent().remove(r._r)
        else:
            p.text = t
    for p in paras[len(texts):]:
        p._p.getparent().remove(p._p)


def set_table(sh, rows, font_pt=None):
    tbl = sh.table
    while len(tbl.rows) < len(rows):
        tr = copy.deepcopy(tbl._tbl.tr_lst[-1])
        tbl._tbl.append(tr)
    while len(tbl.rows) > len(rows):
        tbl._tbl.remove(tbl._tbl.tr_lst[-1])
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = tbl.cell(r_i, c_i)
            p = cell.text_frame.paragraphs[0]
            if p.runs:
                p.runs[0].text = val
                for r in p.runs[1:]:
                    r._r.getparent().remove(r._r)
            else:
                p.text = val
            if font_pt:
                for r in p.runs:
                    r.font.size = Pt(font_pt)


# ---------------------------------------------------------------- slide 3: research question
set_paragraphs(shape(3, "Text 2"), ["Can uplift modeling find the customers whose behaviour an email changes (and which version, men's or women's), does targeting them "
                                    "beat random or propensity targeting, and under which cost conditions does targeting pay compared with mailing everyone?"])

# ---------------------------------------------------------------- slide 6: models
set_paragraphs(shape(6, "Text 10"), ["X-learner (advanced)", "Imputes each customer's effect with the other arm's model, then models the effect directly.",
                                     "S-learner (comparison)", "One model with the arm as a feature.",
                                     "Honest uplift tree (new)", "Splits directly where the uplift differs, only if the two sides differ by 2 standard errors; "
                                     "leaf effects estimated on a separate half of the data. Output: a few readable rules."])
set_paragraphs(shape(6, "Text 13"), ["Same 7 features, same split, same base learner (gradient-boosted trees) for the meta-learners. Tuning by log-loss (calibration), "
                                     "then an uplift-aware check: 5-fold cross-fitted Qini on train picks the deployment candidate before the test set is touched."])

# ---------------------------------------------------------------- slide 7: tests count
set_paragraphs(shape(7, "Text 18"), ["No leakage: one stratified split created once; encoder, tuning, models and the deployment candidate fixed on train only; "
                                     "test outcomes used once. Evaluation functions covered by 22 unit tests with hand-computed answers."])

# ---------------------------------------------------------------- slide 8: women's email
S = pd.read_csv(ROOT / "data" / "processed" / "test_scores.csv", index_col="customer_id")
import sys
sys.path.insert(0, str(ROOT))
from src.evaluation import uplift_curves  # noqa: E402

sub = S[S.arm.isin([0, 2])]
y, t = sub.visit.to_numpy(), (sub.arm == 2).astype(int).to_numpy()
grid = np.linspace(0, 1, 21)
cd = CategoryChartData()
cd.categories = [f"{int(g * 100)}%" for g in grid]
for name, col in [("Random", "random_score"), ("Propensity", "propensity"), ("T-learner", "T_uplift_womens"), ("X-learner", "X_uplift_womens"), ("Uplift tree", "UT_uplift_womens")]:
    c = uplift_curves(y, t, sub[col].to_numpy())
    vals = [float(np.interp(g, c.frac, c.qini) / len(sub)) for g in grid]
    cd.add_series(name, vals)
shape(8, "Chart 0").chart.replace_data(cd)
tq, pq = rank[(W, "T-learner")]["qini"], rank[(W, "Propensity")]["qini"]
set_paragraphs(shape(8, "Text 2"), [f"{tq / pq:.1f}×"])
set_paragraphs(shape(8, "Text 3"), [f"higher Qini than propensity ({tq:.4f} vs {pq:.4f}, T-learner, test set)"])
set_paragraphs(shape(8, "Text 4"), ["Statistically convincing",
                                    f"Paired bootstrap, T − propensity: {paired[(W, 'T-learner - Propensity')]['diff']:.4f} (95% CI {paired[(W, 'T-learner - Propensity')]['lo']:.4f} to "
                                    f"{paired[(W, 'T-learner - Propensity')]['hi']:.4f}); tree − propensity CI {paired[(W, 'Uplift tree - Propensity')]['lo']:.4f} to {paired[(W, 'Uplift tree - Propensity')]['hi']:.4f}.",
                                    f"The tree's rule: {rules_w[0]['rule']} {pts(rules_w[0]['uplift'])}, {rules_w[1]['rule']} {pts(rules_w[1]['uplift'])}. "
                                    f"On unseen customers: {pts(leaves[0]['diff'])} vs {pts(leaves[1]['diff'])}. Top half of the list keeps {lift[(W, 'Uplift tree', 0.5)]['share_of_lift']:.0%} of the effect."])

# ---------------------------------------------------------------- slide 10: model comparison
set_paragraphs(shape(10, "Text 0"), ["Model comparison: differences between uplift models are within noise"])
rows = [["Model", "Women's: test split", "Women's: 64k cross-fit", "Men's: test split", "Men's: 64k cross-fit"]]
for m, label in [("Random", "Random"), ("Propensity", "Propensity"), ("T-learner", "T-learner (main)"), ("X-learner", "X-learner"), ("S-learner", "S-learner"), ("Uplift tree", "Uplift tree (deployment)")]:
    rw, rm, cw, cm = rank[(W, m)], rank[(Mn, m)], rrank[(W, m)], rrank[(Mn, m)]
    rows.append([label, f"{rw['qini']:.4f} [{rw['qini_lo']:.4f}, {rw['qini_hi']:.4f}]", f"{cw['qini']:.4f} [{cw['qini_lo']:.4f}, {cw['qini_hi']:.4f}]",
                 f"{rm['qini']:.4f} [{rm['qini_lo']:.4f}, {rm['qini_hi']:.4f}]", f"{cm['qini']:.4f} [{cm['qini_lo']:.4f}, {cm['qini_hi']:.4f}]"])
tbl_shape = [sh for sh in prs.slides[9].shapes if sh.has_table][0]
set_table(tbl_shape, rows, font_pt=10)
set_paragraphs(shape(10, "Text 3"), ["Within noise (women's)", f"X − T {paired[(W, 'X-learner - T-learner')]['diff']:+.4f} (CI {paired[(W, 'X-learner - T-learner')]['lo']:+.4f} to "
                                     f"{paired[(W, 'X-learner - T-learner')]['hi']:+.4f}); tree − T {paired[(W, 'Uplift tree - T-learner')]['diff']:+.4f} (CI {paired[(W, 'Uplift tree - T-learner')]['lo']:+.4f} to "
                                     f"{paired[(W, 'Uplift tree - T-learner')]['hi']:+.4f}). All beat propensity; none beats random for the men's email."])
set_paragraphs(shape(10, "Text 6"), ["Why T stays main, why the tree would be deployed",
                                     f"T was chosen in advance (no peeking). The tree was picked on training data by cross-fitted Qini; on all 64,000 customers it ranks first "
                                     f"({rrank[(W, 'Uplift tree')]['qini']:.4f}) and T last among uplift models ({rrank[(W, 'T-learner')]['qini']:.4f}). Two rules, nothing to over-fit."])

# ---------------------------------------------------------------- slide 11: whole campaign
set_paragraphs(shape(11, "Text 0"), ["Whole campaign (who + which email): the borderline gain does not replicate"])
cd = CategoryChartData()
ks = [0.1, 0.2, 0.3, 0.5, 1.0]
cd.categories = [f"top {int(k * 100)}%" for k in ks]
for name in ("Random", "Propensity", "T-learner", "Uplift tree"):
    cd.add_series(name, [rpol[(name, k)]["visit"] * 100 for k in ks])
ch = shape(11, "Chart 0").chart
ch.replace_data(cd)
ch.chart_title.text_frame.text = "Extra visits per 100 targeted customers (all 64,000 customers, 5-fold cross-fit)"
set_paragraphs(shape(11, "Text 2"), [f"{rpol[('T-learner', 0.3)]['visit'] * 100:.1f} vs {rpol[('Random', 0.3)]['visit'] * 100:.1f}"])
set_paragraphs(shape(11, "Text 3"), [f"extra visits per 100 targeted: T-learner top 30% vs random, on all 64,000 customers ({pol[('T-learner', 0.3)]['visit'] * 100:.1f} vs {pol[('Random', 0.3)]['visit'] * 100:.1f} on the test split)"])
set_paragraphs(shape(11, "Text 4"), ["Not proven, and we know why",
                                     f"T − random {pts(rcmp[(0.3, 'T-learner - Random')]['diff'])} (CI {pts(rcmp[(0.3, 'T-learner - Random')]['lo'])} to {pts(rcmp[(0.3, 'T-learner - Random')]['hi'])}); "
                                     f"propensity top 30% is the best whole-campaign ranking ({rpol[('Propensity', 0.3)]['visit'] * 100:.1f}).",
                                     f"The men's email is at least as good as the women's in every segment, so choosing the version per customer adds {pts(ver[0.3]['diff'])} (CI includes 0). "
                                     "'Men's email to the most likely visitors' is as good as any uplift policy."])

# ---------------------------------------------------------------- slide 12: economics
set_paragraphs(shape(12, "Text 0"), ["From visits to money: when does targeting pay?"])
costs = [c for c in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3) if c in dmap]
table_costs = [c for c in (0.1, 0.15, 0.2, 0.3) if c in dmap]
table_csv = pd.read_csv(RES / "04_strategy_table.csv")
VALUE, MARGIN, POP = ev["value"], V["economics"]["margin"], V["economics"]["population"]
cd = CategoryChartData()
cd.categories = [f"${c:.2f}" for c in costs]
for strat, k, label in [("Random", 1.0, "Email everyone (men's)"), ("Propensity", 0.3, "Propensity top 30%"), ("T-learner", 0.3, "T-learner top 30%"), ("Uplift tree", 0.3, "Uplift tree top 30%")]:
    r = table_csv[(table_csv.strategy == strat) & (np.isclose(table_csv.k, k))].iloc[0]
    n = np.ceil(k * POP)
    cd.add_series(label, [float(r.inc_visit_per_targeted * n * VALUE * MARGIN - c * n) for c in costs])
ch = shape(12, "Chart 0").chart
ch.replace_data(cd)
ch.chart_title.text_frame.text = "Incremental profit vs cost per contact (visit-value bridge, 30% margin, 64,000 customers)"
rows = [["Cost", "Best plan", "Profit · everyone"]]
for c in table_costs:
    d = dmap[c]
    plan = "do not send" if d["best_strategy"] == "No campaign" else ("mail everyone" if d["best_k"] == 1.0 else f"{d['best_strategy']} top {d['best_k']:.0%}")
    rows.append([f"${c:.2f}", plan, f"{usd(d['best_profit'])} · {usd(d['everyone_profit'])}"])
tbl_shape = [sh for sh in prs.slides[11].shapes if sh.has_table][0]
set_table(tbl_shape, rows, font_pt=10)
set_paragraphs(shape(12, "Text 1"), ["Visit-value bridge",
                                     f"1 extra visit ≈ {usd(VALUE, 2)} spend ≈ {usd(VALUE * MARGIN, 2)} margin (conservative vs observed spend). Mailing everyone breaks even at {usd(ev['everyone']['break_even_cost'], 3)} per contact.",
                                     f"Women's campaign to everyone: {usd(wcamp[('Everyone', 1.0)]['profit'])}; to the tree's top half: {usd(wcamp[('Uplift tree', 0.5)]['profit'])} ({wcamp[('Uplift tree', 0.5)]['share_of_lift']:.0%} of the effect, half the cost). Observed-spend profits: all CIs overlap."])

# ---------------------------------------------------------------- slide 13: recommendations
set_paragraphs(shape(13, "Text 4"), [f"Women's email: send it only to past women's-merchandise buyers (the tree's rule): ~{lift[(W, 'Uplift tree', 0.5)]['share_of_lift']:.0%} of the effect at half the cost.",
                                     "Men's email: send it broadly while emails are cheap; its effect is the same for everyone.",
                                     f"Above ≈ {usd(ev['everyone']['break_even_cost'], 2)} per contact only short targeted lists pay; above ≈ $0.25 nothing does.",
                                     "Before scaling, run a new A/B test of the targeting policy itself."])
set_paragraphs(shape(13, "Text 8"), ["Money rests on the visit-value bridge and assumed cost and margin.",
                                     "Individual uplift is noisy; segments are model-inferred; differences between uplift models are within noise.",
                                     "One 2-week experiment, one retailer, email only.",
                                     "The tree is greedy: it can miss interactions (e.g. both-lines buyers for the men's email)."])

# ---------------------------------------------------------------- slide 14: what we built
set_paragraphs(shape(14, "Text 3"), ["Campaign plan", "Budget, cost, margin in → who to contact, which version, extra visits and profit (95% ranges) vs mail everyone and propensity."])
set_paragraphs(shape(14, "Text 6"), ["Customer lookup", "Real customers: profile, effect of each email, the rule that applies, what happened in the experiment, and the email text (template or LLM)."])
set_paragraphs(shape(14, "Text 9"), ["Evidence", "Qini curves, the tree's rules, CIs on all 64,000 customers, decision map."])
tbl_shape = [sh for sh in prs.slides[13].shapes if sh.has_table][0]
set_table(tbl_shape, [["notebooks/01-05", "EDA → baseline → uplift models + honest tree → evaluation & economics → 64,000-customer robustness"],
                      ["src/", "preprocessing, models, crossfit, evaluation, simulation, messaging"],
                      ["tests/", "22 unit tests (hand-computed answers)"],
                      ["run_all.py", "reruns everything from scratch in about 10 minutes"],
                      ["reports/project_guide.pdf", "full explanation + defense Q&A"]])

# ---------------------------------------------------------------- slide 15: closing
set_paragraphs(shape(15, "Text 0") if False else [sh for sh in prs.slides[14].shapes if sh.name == "Text 0"][1],
               [f"Women's email: two rules keep ~{lift[(W, 'Uplift tree', 0.5)]['share_of_lift']:.0%} of the effect at half the cost · Men's email: send broadly · "
                f"Targeting pays above ≈ {usd(ev['everyone']['break_even_cost'], 2)} per contact, or when the content is fixed"])

prs.save(DECK)
print("Deck updated:", DECK.name)
