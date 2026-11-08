"""The 'AI in marketing' last mile: turn a score into a decision a marketer can read and act on.

* describe_customer()        -> one plain-English line about who the customer is
* explain_recommendation()   -> why the model picks this email (uses the honest uplift tree's rules)
* draft_message()            -> the personalised email text for the recommended version

Message generation has two modes:
  - template (default, deterministic, no internet): segment-aware copy built from the features.
  - LLM (optional): if the environment variable ANTHROPIC_API_KEY is set and llm=True, the
    template brief is sent to Claude (claude-haiku) through the public Messages API with the
    standard library only (no extra package). Any error falls back to the template.
Only the customer's features and the recommended email go into the prompt: no names, no IDs.
"""
from __future__ import annotations

import json
import os
import urllib.request

import pandas as pd

from .config import ARMS

SHOP = "Hillstrom Outfitters"   # fictional retailer name used in the demo copy


def describe_customer(r: pd.Series) -> str:
    bought = {(1, 1): "men's and women's merchandise", (1, 0): "men's merchandise", (0, 1): "women's merchandise"}[(int(r.mens), int(r.womens))]
    months = int(r.recency)
    when = "last month" if months <= 1 else f"{months} months ago"
    status = "new customer (first purchase in the last 12 months)" if int(r.newbie) == 1 else "existing customer"
    return (f"{status.capitalize()}, {str(r.zip_code).lower()} area, last purchase {when}, "
            f"${float(r.history):,.0f} spent in the past year on {bought}, usually buys by {str(r.channel).lower()}.")


def explain_recommendation(r: pd.Series, tree_rules: dict | None = None, score_prefix: str = "T") -> str:
    """Plain-language reason for the recommended email, from the two estimated uplifts
    and (if available) the matching leaf of the honest uplift tree."""
    um, uw = float(r[f"{score_prefix}_uplift_mens"]), float(r[f"{score_prefix}_uplift_womens"])
    action = r[f"{score_prefix}_action"]
    parts = [f"Estimated effect on the chance of a visit: men's email {um * 100:+.1f} pts, women's email {uw * 100:+.1f} pts."]
    if action == "No E-Mail":
        parts.append("Neither email is expected to change this customer's behaviour, so the contact would be wasted.")
    else:
        other = "women's" if action == "Mens E-Mail" else "men's"
        parts.append(f"The {'men’s' if action == 'Mens E-Mail' else 'women’s'} version is expected to add more visits than the {other} one.")
    if int(r.womens) == 1 and int(r.mens) == 0:
        parts.append("Past women's-merchandise buyers are the group the women's email actually moves; for them both versions work.")
    elif int(r.womens) == 0:
        parts.append("Customers who never bought women's merchandise barely react to the women's email; the men's email works for everyone.")
    else:
        parts.append("Customers who bought in both categories react most strongly to the men's email (about double the average effect).")
    if tree_rules is not None and action in tree_rules:
        parts.append("Matching rule from the honest uplift tree: " + _matching_rule(r, tree_rules[action]))
    return " ".join(parts)


def _matching_rule(r: pd.Series, rules: pd.DataFrame) -> str:
    """Find the leaf whose condition the customer satisfies (string rules from HonestUpliftTree.rules())."""
    feats = {"mens": int(r.mens), "womens": int(r.womens), "newbie": int(r.newbie), "recency": float(r.recency),
             "history": float(r.history), f"zip_code_{r.zip_code}": 1, f"channel_{r.channel}": 1}
    for _, row in rules.iterrows():
        if _satisfies(row["rule"], feats):
            return f"“{row['rule']}” → {row['uplift'] * 100:+.1f} pts (estimated on {int(row['n']):,} training customers)."
    return "no matching leaf."


def _satisfies(rule: str, feats: dict) -> bool:
    if rule == "everyone":
        return True
    for cond in rule.split(" and "):
        if " = " in cond:
            name, val = cond.split(" = ")
            if feats.get(name, 0) != (1 if val == "yes" else 0):
                return False
        elif " <= " in cond:
            name, thr = cond.split(" <= ")
            if not feats.get(name, 0) <= float(thr):
                return False
        elif " > " in cond:
            name, thr = cond.split(" > ")
            if not feats.get(name, 0) > float(thr):
                return False
    return True


# --------------------------------------------------------------------------- message copy
def _brief(r: pd.Series, email: str) -> dict:
    line = "men's" if email == "Mens E-Mail" else "women's"
    months = int(r.recency)
    hook = ("Welcome back: here is what's new this season" if months <= 2 else
            f"It has been {months} months. We kept a few things aside for you" if months <= 6 else
            "We miss you: come and see what has changed")
    tier = "VIP" if float(r.history) >= 350 else "loyal" if float(r.history) >= 100 else "new"
    channel = {"Web": "online", "Phone": "by phone", "Multichannel": "online or by phone"}.get(str(r.channel), "online")
    return {"line": line, "hook": hook, "tier": tier, "channel": channel, "newbie": int(r.newbie) == 1,
            "both": int(r.mens) == 1 and int(r.womens) == 1}


def draft_message(r: pd.Series, email: str, llm: bool = False) -> dict:
    """Return {'subject', 'body', 'source'} for the recommended email version."""
    if email == "No E-Mail":
        return {"subject": "(no email)", "body": "Hold this customer out of the campaign: no version is expected to change their behaviour.", "source": "rule"}
    b = _brief(r, email)
    if llm and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _llm_message(b)
        except Exception:
            pass
    return _template_message(b)


def _template_message(b: dict) -> dict:
    subject = f"{b['hook']} · new {b['line']} arrivals at {SHOP}"
    first = ("Thanks for your first order with us this year." if b["newbie"] else
             f"As one of our {b['tier']} customers, you get first look." if b["tier"] != "new" else "Good to have you back.")
    cross = (" Since you shop both collections, we also added a few matching pieces from the other line." if b["both"] else "")
    body = (f"Hi there,\n\n{first} Our new {b['line']} collection has just landed, and the pieces you tend to like are "
            f"back in stock in your size.{cross}\n\nBrowse the collection {b['channel']} this week and enjoy free delivery on your next order.\n\n"
            f"See you soon,\nThe {SHOP} team")
    return {"subject": subject, "body": body, "source": "template"}


def _llm_message(b: dict, model: str = "claude-haiku-4-5") -> dict:
    prompt = (f"Write a short marketing email (subject line + 70-90 word body) for the retailer {SHOP}. "
              f"Promote the new {b['line']} collection. Customer context: {'new customer' if b['newbie'] else b['tier'] + ' customer'}, "
              f"shops {b['channel']}, {'buys both men’s and women’s lines' if b['both'] else 'buys one line'}. Opening angle: '{b['hook']}'. "
              "Warm, concrete, no discounts beyond free delivery, no exclamation marks. Return JSON with keys subject and body.")
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", method="POST",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
        data=json.dumps({"model": model, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}).encode())
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = json.loads(resp.read())["content"][0]["text"]
    text = text[text.find("{"): text.rfind("}") + 1]
    out = json.loads(text)
    return {"subject": out["subject"], "body": out["body"], "source": f"llm:{model}"}
