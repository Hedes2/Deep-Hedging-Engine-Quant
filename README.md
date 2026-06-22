# Delta Hedging & Adversarial Stock Path Generation Simulator

An end-to-end quantitative finance simulator comparing **classical Black-Scholes delta hedging** against a **CVaR-trained deep hedging neural network**, evaluated on both real NIFTY options market data and synthetic Monte Carlo price paths.

> **TL;DR:** A GRU-based deep hedger, trained to directly minimize Conditional Value at Risk (CVaR) under realistic transaction costs, achieves a **27% lower P&L standard deviation** and **19% lower CVaR** than textbook Black-Scholes delta hedging on out-of-sample synthetic test paths — by learning to trade less reactively and absorb fewer transaction costs, at the cost of a small amount of pure hedging precision.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Project Architecture](#project-architecture)
3. [Part 1 — Real Market Data Hedging](#part-1--real-market-data-hedging)
4. [Part 2 — Synthetic Path Generation (GBM)](#part-2--synthetic-path-generation-gbm)
5. [Part 3 — Black-Scholes Baseline](#part-3--black-scholes-baseline)
6. [Part 4 — Deep Hedging Network](#part-4--deep-hedging-network)
7. [Part 5 — Training Objective: CVaR + Transaction Costs](#part-5--training-objective-cvar--transaction-costs)
8. [A Major Debugging Story: The Constant-Output Collapse](#a-major-debugging-story-the-constant-output-collapse)
9. [Results](#results)
10. [Key Insights](#key-insights)
11. [Limitations & Future Work](#limitations--future-work)
12. [References](#references)
13. [Repository Structure](#repository-structure)

---

## Motivation

Delta hedging is the standard technique for neutralizing the directional risk of an options position by continuously rebalancing a position in the underlying asset. The classical approach, **Black-Scholes delta hedging**, assumes:

- Continuous (frictionless) rebalancing
- Constant volatility
- Zero transaction costs

None of these assumptions hold in real markets. **Deep Hedging** (Buehler et al., 2019) reframes hedging as a sequential decision problem solved by a neural network trained to directly minimize a *risk measure* (such as CVaR) of the realized hedging P&L — allowing the network to learn cost-aware, data-driven hedging policies that can outperform Black-Scholes once frictions are introduced.

This project implements both approaches side by side and stress-tests the comparison under transaction costs, using real NIFTY index options/futures data for context and 1,000+ Monte Carlo–simulated price paths for training and evaluation.

<!-- IMAGE SUGGESTION: A simple before/after diagram contrasting "Black-Scholes: fixed formula, no cost awareness" vs "Deep Hedger: learns from simulated P&L, cost-aware" -->

---

## Project Architecture

```
┌─────────────────────┐      ┌──────────────────────┐      ┌────────────────────────┐
│  Real NIFTY Market   │      │   GBM Synthetic Path  │      │   Black-Scholes Greeks │
│  Data (minute-level) │ ───► │     Generator         │ ───► │   & Implied Vol Solver │
│  Futures + Options   │      │   (1,000 paths)       │      │                        │
└─────────────────────┘      └──────────────────────┘      └───────────┬────────────┘
                                                                         │
                                                  ┌──────────────────────┴───────────────────────┐
                                                  ▼                                                ▼
                                    ┌──────────────────────────┐                  ┌─────────────────────────────┐
                                    │  Black-Scholes Delta      │                  │   GRU Deep Hedger            │
                                    │  Hedging (baseline)       │                  │   (trained on CVaR + costs)  │
                                    └──────────────┬────────────┘                  └───────────────┬─────────────┘
                                                   │                                                │
                                                   └───────────────────┬────────────────────────────┘
                                                                       ▼
                                                       ┌───────────────────────────────┐
                                                       │  P&L Distribution Comparison   │
                                                       │  Mean / Std / VaR / CVaR       │
                                                       └───────────────────────────────┘
```

<!-- IMAGE SUGGESTION: Insert your actual pipeline diagram / flowchart image here -->

---

## Part 1 — Real Market Data Hedging

**Data:** Minute-level NIFTY futures and options last-trade-prices for 5 Feb 2026 (`NIFTY26FEBFUT` and the full option chain), with strikes parsed from symbol strings (e.g. `NIFTY2621025700CE` → strike `25700`, type `call`).

**Implied Volatility (IV):** Since volatility is not directly observable, it is reverse-engineered from observed option prices at every minute via **bisection search**:

$$\text{Find } \sigma \text{ such that } C_{BS}(S, K, T, r, \sigma) = C_{\text{market}}$$

This is necessarily re-solved at every timestep because real market sentiment (and therefore implied volatility) genuinely drifts intraday — a constant-volatility assumption would not match observed prices.

> **Important methodological note:** dynamically re-solving IV at every minute technically *violates* Black-Scholes's own foundational assumption of constant volatility. This is a well-known, deliberate practitioner convention — Black-Scholes is used less as "the true model of price dynamics" and more as a **quoting/translation layer** that converts an observed price into a single comparable number (IV) at each instant. This tension is the entire reason stochastic-volatility models (e.g. Heston) and deep hedging approaches exist.

**Delta Hedging Loop (real data):** At each minute *t*, compute IV from the observed option price, plug into Black-Scholes to get δ*ₜ*, and accumulate trading P&L:

$$\text{P\&L}_{\text{trading}} = \sum_t \delta_t \cdot (S_{t+1} - S_t)$$

$$\text{PL}_T = -Z_T + \text{P\&L}_{\text{trading}} - C_T(\delta)$$

where *Z_T* is the option payoff owed at expiry (the hedger is short the option) and *C_T(δ)* is the transaction cost (zero in the simplest version).

![Deep Hedger GRU Architecture](images/model_architecture.png)

---

## Part 2 — Synthetic Path Generation (GBM)

To train and evaluate a deep hedger at scale, 1,000 synthetic price paths are generated via **Geometric Brownian Motion (GBM)**, seeded from the real futures price and matching the real data's time grid (375 one-minute steps, 09:16–15:30):

$$S_{i} = S_{i-1} \cdot \exp\left[\left(\mu - \tfrac{1}{2}\sigma^2\right) dt + \sigma \sqrt{dt}\, Z_i\right], \quad Z_i \sim \mathcal{N}(0,1)$$

- **μ = r** (risk-free rate) — the risk-neutral drift assumption used for no-arbitrage option pricing
- **σ = 0.6** — fixed annualized volatility (deliberately kept constant, since the generating process defines the "true" volatility for this synthetic world — no IV-solving needed here, unlike Part 1)
- **dt** — derived from the real data's actual elapsed time divided into 375 steps, so each simulated step represents one real-world minute
- The **−½σ²dt** term is the Itô correction, necessary because GBM models the *log* of price as normally distributed; without it, the *average* of many simulated paths would drift upward faster than the risk-free rate due to a known bias from exponentiating a normal random variable

Paths are split **800 / 200** (train/test) for the deep hedger.

<!-- IMAGE SUGGESTION: overlay plot of ~30-50 sample GBM paths fanning out from S0 -->

---

## Part 3 — Black-Scholes Baseline

For every one of the 1,000 paths, at every one of the 375 timesteps, the closed-form Black-Scholes price, delta, gamma, and theta are computed using the **same fixed σ = 0.6** that generated the paths (ensuring an internally consistent ground truth):

$$d_1 = \frac{\ln(S/K) + (r + \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

$$C = S\,\Phi(d_1) - K e^{-rT}\Phi(d_2), \qquad \Delta_{\text{call}} = \Phi(d_1)$$

This produces the `all_bs_deltas` array (shape `1000 × 375`) used both as a **feature** fed into the deep hedger and as the **baseline** it is benchmarked against.

---

## Part 4 — Deep Hedging Network

### Architecture

A single **GRU (Gated Recurrent Unit)** layer replaces a naive step-by-step feedforward network, processing the *entire* 375-step price path in one forward pass rather than looping manually and feeding a previous-delta input back by hand.

```
Input: S_path (375, 1)  ──┐
Input: BS_delta_path (375, 1) ──┼──► Concatenate ──► GRU(64, return_sequences=True) ──► TimeDistributed(Dense(1, sigmoid)) ──► δ_path (375, 1)
Input: T_remaining_path (375, 1) ──┘
```

| Layer | Output Shape | Parameters |
|---|---|---|
| S_path_input | (None, 375, 1) | 0 |
| BS_delta_path_input | (None, 375, 1) | 0 |
| T_remaining_input | (None, 375, 1) | 0 |
| Concatenate | (None, 375, 3) | 0 |
| GRU (64 units) | (None, 375, 64) | ~13,248 |
| TimeDistributed(Dense(1, sigmoid)) | (None, 375, 1) | 65 |

**Why GRU over a plain RNN or manual loop?** Sequences here run ~375 steps; plain RNNs suffer significant vanishing-gradient degradation over sequences this long. GRU's gating mechanism mitigates this with fewer parameters than an LSTM, making it a good efficiency/expressiveness trade-off for this problem size.

**Why `sigmoid`, not `tanh`, on the output?** A call option's delta is mathematically bounded to **(0, 1)** — never negative. `tanh` (range −1 to 1) wastes capacity on impossible outputs and was implicated in early training instability; `sigmoid` constrains the output to the valid range by construction.

**Inputs, and why each matters:**
- **Sₜ (normalized as `log(S/K)`)** — the moneyness of the option, in log-space. *(See the [debugging story](#a-major-debugging-story-the-constant-output-collapse) below for why raw, unnormalized prices ~25,785 broke training entirely.)*
- **BS deltaₜ** — gives the network a sensible reference point ("learn the correction to the textbook answer" rather than learning option theory from scratch)
- **T_remaining** — explicit time-to-expiry, so the network can learn how urgency-to-hedge changes as expiry approaches, without having to infer it implicitly from its position in the sequence

<!-- IMAGE SUGGESTION: model architecture diagram (Keras plot_model output or a hand-drawn box diagram) -->

---

## Part 5 — Training Objective: CVaR + Transaction Costs

### Why CVaR instead of MSE-only?

**Value at Risk (VaR)** at level α answers: *"what loss threshold is exceeded only α% of the time?"* — a single cutoff point.

**Conditional VaR (CVaR / Expected Shortfall)** answers the more informative question: *"given that we're in the worst α% of outcomes, what's the average loss?"* CVaR captures tail severity, not just where the tail begins, and is the risk measure used for training in this project (α = 0.05):

$$\text{CVaR}_\alpha = -\mathbb{E}\left[\,\text{P\&L} \mid \text{P\&L} \le \text{VaR}_\alpha\,\right]$$

### Transaction Costs

Black-Scholes assumes frictionless, continuous rebalancing — unrealistic in practice. A proportional transaction cost term was added directly into the P&L calculation, charged whenever the hedge position changes:

$$\text{Cost}_t = c \cdot |\delta_t - \delta_{t-1}| \cdot S_t$$

$$\text{Total P\&L} = -Z_T + \sum_t \delta_t (S_{t+1}-S_t) - \sum_t \text{Cost}_t$$

This term is applied **identically to both the deep hedger and the Black-Scholes baseline** during evaluation — a fair comparison requires both strategies to pay the same real-world friction; giving one side free trades would invalidate the comparison entirely.

### Combined Loss Function

$$\mathcal{L} = \lambda_{\text{MSE}} \cdot \text{MSE}(\delta_{\text{pred}}, \delta_{BS}) + \lambda_{\text{CVaR}} \cdot \text{CVaR}_\alpha(\text{P\&L})$$

The MSE anchor term (against the Black-Scholes delta) was added for **training stability** — see below for why pure CVaR alone proved insufficient. This does not contradict the deep hedging literature: CVaR minimization remains the actual target; MSE serves as a regularizer/warm-start mechanism, a standard and well-documented technique to stabilize sparse-gradient objectives, not a deviation from the core idea.

---

## A Major Debugging Story: The Constant-Output Collapse

This is worth documenting because it's a genuine, instructive failure mode in deep hedging implementations, and the diagnostic process is itself a meaningful part of this project.

### Symptom
After training, the network output the **exact same delta value** (e.g. `0.5758`) for every path and every timestep — completely ignoring its inputs. Standard deviation of predictions across the entire test set was exactly `0.0000`.

### What was ruled out, in order
| Attempt | Change | Result |
|---|---|---|
| 1 | Baseline: `batch_size=32`, `tanh` output, CVaR-only loss | Collapsed to a constant after ~4 epochs |
| 2 | `batch_size=128` (denser CVaR tail), `sigmoid` output | Slower collapse (~19 epochs), same outcome |
| 3 | Added MSE anchor term (`loss = MSE + CVaR`) | **Still collapsed** — the key clue |

Attempt 3 was the breakthrough diagnostic: MSE against a target that genuinely varies from 0 to 1 *should* produce a strong, clean gradient pushing the network away from a flat output. The fact that it still collapsed proved the problem was **not** in the loss function design at all — it had to be further upstream, in how gradients were flowing through the network itself.

### Root Cause: Saturated GRU Gates from Unnormalized Inputs

The underlying price `Sₜ ≈ 25,785` was being fed **raw** into the GRU layer. Keras's GRU relies internally on `sigmoid` and `tanh` activations for its gates. With small, randomly-initialized weights multiplied by an input of magnitude ~25,000, the pre-activation values become enormous (e.g. ±5,000) — and:

$$\sigma(5000) \approx 1.0, \qquad \frac{d\sigma}{dx}\bigg|_{x=5000} \approx 0.0$$

The gates **saturate on the very first forward pass**. Since the derivative of a saturated sigmoid/tanh is essentially zero, gradients backpropagating through the GRU are multiplied by ~0 at every step — by the chain rule, **even a perfectly clean MSE gradient gets annihilated the moment it passes through the dead GRU layer.** This is why adding MSE alone could not fix it: the bug wasn't downstream in the loss, it was upstream in the network's first real computation.

### The Fix

Normalize the price input before it ever reaches the network — specifically `log(S/K)` (a standard, theoretically meaningful normalization, since this is exactly the moneyness term that appears inside the Black-Scholes *d₁* formula itself):

```python
normalized_S_paths = tf.math.log(batch_S_paths / tf.constant(option_K, dtype=tf.float32))
predicted_deltas = deep_hedger_gru_model([normalized_S_paths, batch_BS_deltas, batch_T])
```

Post-fix diagnostic confirmed the collapse was resolved:

| Metric | Pre-fix | Post-fix |
|---|---|---|
| Std of predicted deltas | 0.0000 | **0.2769** |
| Min / Max delta | 0.576 / 0.576 (constant) | **0.061 / 0.964** |
| Mean \|delta change\| per step | 0.0000 | **0.0082** |

**Lesson:** input scale matters enormously for any architecture relying on saturating activations (sigmoid, tanh) — this is a classic, well-documented deep learning pitfall, but it's easy to overlook when raw financial price levels (tens of thousands) look unremarkable until you consider what a `tanh` or `sigmoid` gate actually does with them.

<!-- IMAGE SUGGESTION: side-by-side histogram of predicted deltas before vs after the fix, visually showing a single spike vs a real distribution -->

---

## Results

Evaluated on the **held-out 200-path test set**, with realistic proportional transaction costs (`cost_rate = 0.0005`) applied identically to both strategies:

| Metric | Deep Hedger (GRU) | Black-Scholes | Improvement |
|---|---|---|---|
| Mean P&L | -258.91 | -312.77 | +17.2% (less negative) |
| **Std Dev P&L** | **27.81** | 37.99 | **−26.8%** |
| VaR (5%) | -304.67 | -376.99 | +19.2% |
| **CVaR (5%)** | **316.60** | 388.69 | **−18.6%** |

Same pattern holds on the training set (316.60 vs 388.69 test CVaR; 317.50 vs 376.83 train CVaR) — consistent, not a lucky split.

**Behavioral comparison** (why the deep hedger wins):

| | Deep Hedger | Black-Scholes |
|---|---|---|
| Mean \|delta change\| per step | 0.0082 | 0.0174 |
| Max \|delta change\| per step | 0.2675 | 0.7988 |

The deep hedger trades roughly **half as reactively** as Black-Scholes, and never makes the large, sudden delta swings Black-Scholes is prone to near the strike. It sacrifices a small amount of pure hedging precision in exchange for substantially lower transaction costs and tail risk — precisely the trade-off deep hedging is designed to discover, and one Black-Scholes structurally cannot make since it has no concept of transaction costs at all.

<!-- IMAGE SUGGESTION: the 2x2 P&L distribution histogram (Deep Hedger vs BS, Train vs Test) from the notebook -->
<!-- IMAGE SUGGESTION: bar chart comparing Std Dev and CVaR side by side for both strategies -->

---

## Key Insights

1. **A risk-aware loss function changes what "good hedging" means.** Black-Scholes optimizes for *replication accuracy* under an idealized frictionless world; a CVaR-trained network optimizes directly for *tail-risk reduction* under realistic frictions — these are different objectives, and the "better" hedger depends entirely on which one actually matters for the use case.

2. **Sparse-tail objectives (CVaR) are prone to training collapse**, since with `alpha=0.05` and a batch of *N* paths, only `⌊0.05N⌋` paths contribute any gradient signal at all — the rest are computed but ignored. A dense regularizer (MSE) or larger batch size is often necessary to prevent the optimizer from finding a "safe constant" shortcut. Larger batch size alone was tried first and was *not* sufficient in this project — it only slowed the collapse, it didn't prevent it.

3. **Input normalization is not optional for architectures using saturating activations.** Financial price levels (tens of thousands) interact catastrophically with `sigmoid`/`tanh`-gated recurrent layers if fed in raw. `log(S/K)` — moneyness — is both numerically well-behaved and economically meaningful, since it's the same quantity Black-Scholes itself is built around.

4. **A fair benchmark requires identical frictions on both sides.** Comparing a cost-aware deep hedger against a cost-blind Black-Scholes baseline would be a meaningless, rigged comparison; both were evaluated under the exact same `cost_rate` in this project specifically to avoid that pitfall.

5. **Dynamically re-estimating implied volatility, while standard market practice, is a known violation of Black-Scholes's own constant-volatility assumption.** This project deliberately separates the *real-data IV-based hedging* (practical, market-realistic, theoretically inconsistent) from the *synthetic constant-σ hedging* (theoretically clean, since the data-generating process and the pricing model agree on σ by construction) to keep this distinction explicit.

---

## Limitations & Future Work

- **Adversarial price-process generation was not implemented.** This project's title and inspiration draw from Hirano, Minami & Imajo's "Adversarial Deep Hedging" (2023), but only the **deep hedging component** (GRU hedger trained on a CVaR objective) was built here — the paper's core contribution, a GAN-style generator that produces adversarial market scenarios without assuming a price process at all, is a natural and substantial next step, and would let the hedger be stress-tested against scenarios harder than plain GBM.
- **Fixed volatility regime (σ=0.6) for the GBM/deep-hedging comparison** — does not test the deep hedger under stochastic volatility or regime changes, which is where deep hedging's theoretical advantages over Black-Scholes are expected to be largest.
- **Single option (one strike, one expiry)** — a natural extension is a full hedging book across multiple strikes/expiries simultaneously.
- **GRU rather than LSTM or attention-based architectures** — GRU was chosen for parameter efficiency at this sequence length; an ablation against LSTM/Transformer-style architectures was not performed.
- **cost_rate is a single assumed constant** — a sensitivity analysis across a range of plausible transaction cost levels would strengthen the robustness of the conclusion.

---

## References

- Hirano, M., Minami, K., Imajo, K. (2023). *Adversarial Deep Hedging: Learning to Hedge without Price Process Modeling.* — this project implements the **deep hedging component** (GRU-based hedger trained on a CVaR risk objective) described in this paper; the adversarial market-scenario generator (the GAN-style price-process-free training loop that is the paper's main contribution) was not implemented here and is noted as future work below.
- Buehler, H., Gonon, L., Teichmann, J., Wood, B. (2019). *Deep Hedging.* Quantitative Finance. — the foundational deep hedging framework (CVaR-based training objective) this project's loss function builds on.
- Hull, J. C. *Options, Futures, and Other Derivatives.*
- Black, F., Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.

---

## Repository Structure

```
.
├── README.md
├── HFT_CAMP_CODE.ipynb                     # full pipeline: real data → GBM → BS baseline → GRU deep hedger → evaluation
├── 20260205_option_minute_prices_expiry.csv  # minute-level NIFTY futures + options data, 5 Feb 2026
└── images/
    ├── model_architecture.png              # Keras plot_model() output of the GRU deep hedger
    ├── delta_iv_price_plot.png             # real-data hedging: delta / IV / underlying price over time
    ├── gbm_sample_paths.png                # sample of simulated GBM price paths
    ├── pnl_distributions.png               # deep hedger vs Black-Scholes P&L histograms (train/test)
    └── delta_collapse_before_after.png     # debugging story: predicted-delta distribution before/after the input-normalization fix
```

---

*Built as a self-project exploring the intersection of classical quantitative finance and modern deep learning for risk management — May–June 2026.*
