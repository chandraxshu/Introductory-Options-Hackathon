# META Option Pricing & Risk Analysis  
**IO Hackathon – Understanding Option Prices and Risk**

---

## Project Overview

This project analyzes a **live call option on META (Meta Platforms Inc.)** to demonstrate how option prices are formed, how risks are quantified using Greeks, and how a hedge fund can manage those risks in practice.

Using a **simple Black–Scholes pricing framework** and **12 months of historical market data**, the project:
- Estimates a fair value for the option
- Compares model price with the observed market price
- Interprets key option Greeks in intuitive terms
- Identifies major risks in the position
- Proposes a basic hedging strategy
- Presents findings in a consulting-style memo and presentation

The analysis is intentionally designed for a **non-technical portfolio manager**, prioritizing clarity, intuition, and practical decision-making over mathematical complexity.

---

## Objectives Covered

This project fully addresses the hackathon objectives:

### 1. Option Pricing Basics
- Used the Black–Scholes model to estimate a theoretical price for a META call option
- Compared the model-estimated price with the observed market price
- Explained differences driven by implied vs historical volatility and market expectations

### 2. Introduction to Greeks
- Calculated and interpreted:
  - **Delta** – sensitivity to stock price movements
  - **Theta** – impact of time decay
  - **Vega** – sensitivity to volatility changes
- Explained each Greek using simple numerical examples

### 3. Risk Awareness & Hedging
- Identified key risks using Greeks (directional, time decay, volatility risk)
- Proposed a **delta-hedging strategy** using the underlying META stock

### 4. Consulting Recommendation
- Summarized findings in a one-page memo
- Provided actionable recommendations and monitoring guidance for a hedge fund manager

---

## Repository Contents

### Python Code
**`META_option_analysis.py`**

The Python script acts as the analytical engine for the project:
- Downloads 12 months of META stock data using `yfinance`
- Computes historical (realized) volatility
- Selects a future-dated at-the-money call option
- Prices the option using the Black–Scholes model
- Computes implied volatility from the market price
- Calculates Delta, Theta, and Vega
- Generates charts and summary outputs

---

### Presentation Deck
**`META_option_deck.pptx` / `META_option_deck.pdf`**

A professional slide deck (≤9 slides) explaining:
- Option pricing methodology
- Model vs market price comparison
- Interpretation of Greeks
- Risk assessment and hedging strategy
- Final recommendation for a hedge fund manager

---

### One-Page Consulting Memo
**`META_one_page_memo.pdf`**

A concise, non-technical memo that:
- Assesses whether the option appears cheap or expensive
- Highlights key risks using Greeks
- Recommends a delta hedge and monitoring plan
- Is suitable for direct review by a portfolio manager

---

### Visual Outputs
- **`meta_price_history.png`** – META historical price chart  
- **`meta_volatility.png`** – Rolling volatility vs long-term realized volatility  

These visuals support volatility estimation and pricing assumptions.

---

## Methodology Summary

- **Data Source:**  
  - Stock and option data retrieved via `yfinance`

- **Volatility Estimation:**  
  - 12 months of daily log returns  
  - Annualized historical volatility

- **Pricing Model:**  
  - Black–Scholes (European option approximation)
  - Used as a transparent baseline model

- **Greeks Calculation:**  
  - Closed-form Black–Scholes Greeks
  - Vega expressed per **1 volatility point (1%)** for intuitive interpretation

- **Hedging Strategy:**  
  - Delta hedge using underlying META shares
  - Hedge size derived directly from option Delta

---

## Key Assumptions & Limitations

- Black–Scholes assumes:
  - European-style options
  - Constant volatility and interest rates
- Early exercise, discrete dividends, and transaction costs are not explicitly modeled
- Liquidity and bid–ask spreads are discussed qualitatively

These limitations are acknowledged in both the memo and presentation.

---

## How to Reproduce the Analysis (Optional)

To rerun the analysis locally:

```bash
pip install yfinance numpy pandas scipy matplotlib python-pptx
python META_option_analysis.py
