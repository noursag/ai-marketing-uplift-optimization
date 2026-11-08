# AI-Powered Personalized Promotion Targeting

### Uplift Modeling for Email Marketing Optimization

An AI in Marketing project that applies **causal machine learning (uplift modeling)** to the Hillstrom Email Marketing dataset to optimize promotional email targeting.

Instead of predicting which customers are most likely to visit, the project estimates **which customers are more likely to visit because of receiving an email**.

## Problem & Objective

Traditional marketing models prioritize customers with high purchase or visit probabilities, even when those customers would have responded without a promotion.

This project addresses three questions:

1. **Who to target?** Identify customers with the highest estimated incremental response.
2. **Which email to send?** Choose between men's and women's promotional emails.
3. **When is targeting profitable?** Compare the estimated economic impact of targeting strategies under different campaign costs.

## Dataset

**Hillstrom Email Marketing Dataset** — 64,000 customers randomly assigned to:

- No E-Mail (control)
- Mens E-Mail
- Womens E-Mail

**Features:** Purchase history, recency, product preferences, customer type, location category, and purchase channel.

**Outcomes:** Website visits (primary), conversions, and spending (secondary).

The data is split into:

- Training: 60% (38,400 customers)
- Validation: 20% (12,800 customers)
- Testing: 20% (12,800 customers)

## Methodology

The project compares:

| Model              | Purpose                                      |
| ------------------ | -------------------------------------------- |
| Random targeting   | Baseline                                     |
| Propensity model   | Predict likelihood of visiting               |
| T-Learner          | Main uplift model                            |
| X-Learner          | Alternative causal uplift estimator          |
| S-Learner          | Single-model uplift estimator                |
| Honest Uplift Tree | Interpretable, segment-based uplift modeling |

**Evaluation:**

- Qini curves and coefficients
- AUUC and uplift@K
- Five-fold cross-fitting
- Bootstrap confidence intervals
- Inverse-propensity-weighted policy evaluation
- Validation-based targeting threshold selection
- Incremental visits and estimated campaign profitability

## Key Findings

- **Women's email:** Uplift models identify customers with stronger incremental responses. The honest uplift tree achieves a cross-fitted Qini of **0.0039**, compared with **0.0008** for propensity.
- **Targeting efficiency:** Contacting the top 50% identified by the uplift tree retains approximately **81%** of the women's email's incremental visit effect.
- **Men's email:** Propensity performs better than the uplift models in the cross-fitted ranking evaluation.
- **Full campaign:** Uplift-based personalization does not demonstrate a consistent advantage over simpler targeting strategies.
- **Economics:** Higher uplift ranking quality does not necessarily translate into higher campaign profit. Financial results depend on campaign costs, margins, and uncertainty.

**Conclusion:** Uplift modeling is most useful when customer treatment effects differ substantially. Its value must be evaluated against simpler marketing strategies using randomized experimental evidence.

## Interactive Dashboard

Built with **Streamlit**, the dashboard includes three views:

**1. Campaign Plan**

- Select the campaign and targeting model
- Adjust budget, contact cost, visit value, and margin
- Compare expected incremental visits and profit
- Explore recommended targeting strategies

**2. Customer Lookup**

- Explore individual customer profiles
- Compare predicted uplift for both email versions
- View model recommendations and explanations
- Preview personalized promotional email copy

**3. Evidence**

- Qini curves and confidence intervals
- Interpretable uplift tree rules
- Ranking comparisons and economic decision maps

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/noursag/ai-marketing-uplift-optimization.git
cd ai-marketing-uplift-optimization
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows (CMD):

```bat
.venv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the notebooks

Execute the notebooks in order:

1. `01_eda.ipynb`
2. `02_baseline_propensity.ipynb`
3. `03_uplift_models.ipynb`
4. `04_evaluation_economics.ipynb`
5. `05_robustness_crossfit.ipynb`

Alternatively, run the automated pipeline:

```bash
python run_all.py
```

### 5. Launch the dashboard

```bash
python -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**.

### 6. Run tests

```bash
python -m pytest -q
```

The current project version passes **22 unit tests**.

## Project Structure

```text
├── app/                # Streamlit dashboard
├── data/               # Datasets and predictions
├── models/             # Trained models
├── notebooks/          # Analysis and evaluation
├── reports/            # Figures and results
├── src/                # ML and evaluation code
├── tests/              # Unit tests
├── tools/              # Execution utilities
├── requirements.txt
├── requirements-lock.txt
├── run_all.py
└── README.md
```

## Limitations

- Low conversion rates make spending-based evaluation noisy.
- Individual treatment effects are estimates, not directly observable outcomes.
- Financial simulations depend on assumed campaign costs and margins.
- Cross-fitting improves robustness but does not replace an independent experiment.
- A future randomized campaign would be needed to confirm real-world profitability.

---

**Tech stack:** Python, Pandas, Scikit-learn, Scikit-uplift, Streamlit, Matplotlib
