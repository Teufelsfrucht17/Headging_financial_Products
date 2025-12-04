# [nicht in PDF]
import numpy as np
import pandas as pd

# ================================
# CONFIG – Exxon CO2 Hedging Model
# ================================

YEARS = list(range(2025, 2031))  # 2025–2030 inkl.

# Globaler Scope-1-Anker (grob basierend auf Exxon-Daten, 2016–2024 Trend & 2030 Ziel)
SCOPE1_ANCHOR_START_YEAR = 2024
SCOPE1_ANCHOR_START_T    = 103_000_000   # tCO2e global Scope 1 (≈ Exxon 2024, gerundet)
SCOPE1_ANCHOR_END_YEAR   = 2030
SCOPE1_ANCHOR_END_T      = 90_000_000    # tCO2e 2030 (modellierte Zielgröße, leichter Rückgang)

# Modellierte größten Märkte
MARKETS = ["USA", "Canada", "EU_UK"]

# Marktanteile an globalen Scope-1-Emissionen (Approx. aus Produktionsanteilen abgeleitet)
MARKET_SHARE = {
    "USA":   0.45,   # ~45 % der Produktion in den USA
    "Canada":0.20,   # Kanada + weitere Amerika ≈ 20 % für Modell
    "EU_UK": 0.05,   # EU/UK sehr klein, 5 % als obere Schätzung
}
# Rest (~30 %) = „Other“, wird im Modell ignoriert (kein Hedge)

# Anteil der Emissionen, die effektiv einem CO2-Preis unterliegen (Coverage)
# – konservativ, aber an Systemlogik orientiert
COVERAGE_2025 = {
    "USA":   0.30,   # nur Teile (Kalifornien, Washington, RGGI)
    "Canada":0.70,   # föderaler Mindestpreis + Provinzsysteme
    "EU_UK": 0.90,   # nahezu alle großen Anlagen im ETS/UK ETS
}
COVERAGE_2030 = {
    "USA":   0.50,   # mehr Staaten / stärkere Regulierung angenommen
    "Canada":0.90,   # fast vollständige Abdeckung der großen Emittenten
    "EU_UK": 1.00,   # vollständige Abdeckung angenommen
}

# CO2-Preisanker 2025/2030 (lokale Währung)
# USA & Canada hier bereits in USD, EU/UK in EUR (wird unten mit FX in USD umgerechnet)
PRICE_2025 = {
    "USA":   30.0,   # USD/t – angenähert an CCA/US-ETS-Größenordnung
    "Canada":69.0,   # USD/t – aus 95 CAD ≈ 69.38 USD (Bundes-Backstop 2025)  [oai_citation:0‡icapcarbonaction.com](https://icapcarbonaction.com/system/files/ets_pdfs/icap-etsmap-factsheet-135.pdf?utm_source=chatgpt.com)
    "EU_UK": 80.0,   # EUR/t – in der Nähe aktueller EUA-Dec25-Futures (~78–83 EUR)  [oai_citation:1‡Investing.com](https://www.investing.com/commodities/european-union-allowance-eua-year-futures-historical-data?utm_source=chatgpt.com)
}
PRICE_2030 = {
    "USA":   50.0,   # USD/t – moderates Wachstumsszenario
    "Canada":124.0,  # USD/t – aus 170 CAD ≈ 124.15 USD (Bundes-Backstop 2030)  [oai_citation:2‡icapcarbonaction.com](https://icapcarbonaction.com/system/files/ets_pdfs/icap-etsmap-factsheet-135.pdf?utm_source=chatgpt.com)
    "EU_UK": 90.0,   # EUR/t – leicht höheres EUA-Niveau in 2030 (Szenario)
}

# FX-Annahme für EUR → USD
FX_EURUSD = 1.08   # 1 EUR = 1.08 USD (grobe Annahme)

# Fixpreise der Futures (Hedge-Preis K) in USD/t
# – etwas über den 2025-Spots (Risk Premium)
FUTURES_PRICE_USD = {
    "USA":   35.0,                 # USD/t – CO2-Futures auf US-Systeme
    "Canada":75.0,                 # USD/t – leicht über 2025-Backstop
    "EU_UK": 85.0 * FX_EURUSD,     # EUR 85 → USD (EUA-Future-Niveau leicht über Spot)
}

# Volatilität (annualisierte sigma) pro Markt
SIGMA = {
    "USA":   0.20,   # 20 % – US-Carbonpreise (CCA/RGGI) historisch moderat volatil
    "Canada":0.25,   # 25 % – fragmentierter Markt, etwas höhere Unsicherheit  [oai_citation:3‡clearbluemarkets.com](https://www.clearbluemarkets.com/knowledge-base/navigating-canadas-carbon-markets-ahead-of-the-2026-federal-benchmark-review?utm_source=chatgpt.com)
    "EU_UK": 0.35,   # 35 % – EUAs historisch ~30–40 % Jahresvola  [oai_citation:4‡Investing.com](https://www.investing.com/commodities/european-union-allowance-eua-year-futures-historical-data?utm_source=chatgpt.com)
}

# Simulationseinstellungen
N_SIMS = 20_000     # Anzahl Monte-Carlo-Pfade
RANDOM_SEED = 42    # Reproduzierbarkeit

# Hedge-Ratios, die du durchsimulierst (global, gleiche Quote für alle Märkte)
HEDGE_RATIOS_TO_TEST = [0.0, 0.3, 0.5, 0.7, 1.0]

# „Other“ nicht explizit modellieren
INCLUDE_OTHER = False
# ============================================================
# HELPERS
# ============================================================

# [nicht in PDF]
def linear_interp(years, x0_year, x0_value, x1_year, x1_value) -> dict:
    """Linear interpolation for yearly series."""
    years = list(years)
    out = {}
    for y in years:
        w = (y - x0_year) / (x1_year - x0_year)
        out[y] = (1 - w) * x0_value + w * x1_value
    return out

# [nicht in PDF]
def build_scope1_global(years) -> dict:
    """Build global Scope 1 series for modeled years."""
    return linear_interp(
        years,
        SCOPE1_ANCHOR_START_YEAR, SCOPE1_ANCHOR_START_T,
        SCOPE1_ANCHOR_END_YEAR, SCOPE1_ANCHOR_END_T
    )

# [nicht in PDF]
def build_scope1_by_market(scope1_global: dict, include_other=False) -> dict:
    """Allocate global Scope 1 to markets via market shares."""
    shares_sum = sum(MARKET_SHARE.values())
    if shares_sum > 1.0 + 1e-9:
        raise ValueError("MARKET_SHARE sums to > 1.0. Fix shares.")

    scope1_market = {y: {} for y in scope1_global.keys()}

    for y, total in scope1_global.items():
        for m in MARKETS:
            scope1_market[y][m] = total * MARKET_SHARE[m]

        if include_other:
            scope1_market[y]["Other"] = total * (1.0 - shares_sum)

    return scope1_market

# [nicht in PDF]
def build_coverage(years) -> dict:
    """Coverage ratio per year and market (linear 2025->2030)."""
    cov = {y: {} for y in years}
    for m in MARKETS:
        series = linear_interp(years, 2025, COVERAGE_2025[m], 2030, COVERAGE_2030[m])
        for y in years:
            cov[y][m] = float(series[y])
    return cov

# [nicht in PDF]
def to_usd_price(market: str, price_local: float) -> float:
    """Convert EU_UK EUR price into USD; USA/Canada assumed already USD."""
    if market == "EU_UK":
        return price_local * FX_EURUSD
    return price_local

# [nicht in PDF]
def build_price_anchors_usd() -> tuple[dict, dict]:
    """Convert 2025 and 2030 anchor prices into USD per market."""
    p25 = {m: to_usd_price(m, PRICE_2025[m]) for m in MARKETS}
    p30 = {m: to_usd_price(m, PRICE_2030[m]) for m in MARKETS}
    return p25, p30

# [nicht in PDF]
def simulate_prices_gbm_targeted(years, n_sims, p0_usd, pT_usd, sigma, seed=42) -> dict:
    """
    Simulate GBM price paths with drift chosen to hit pT in expectation.
    prices[m] shape: (n_sims, n_years)
    """
    np.random.seed(seed)
    n_years = len(years)
    T = n_years - 1  # steps from first to last year

    prices = {}

    for m in MARKETS:
        s0 = p0_usd[m]
        sT = pT_usd[m]
        vol = sigma[m]

        # Choose mu such that E[S_T] = S_0 * exp(mu*T) = sT  -> mu = ln(sT/s0)/T
        mu = np.log(sT / s0) / max(T, 1)

        paths = np.zeros((n_sims, n_years))
        paths[:, 0] = s0

        for t in range(1, n_years):
            dt = 1.0
            shock = np.random.normal(
                (mu - 0.5 * vol**2) * dt,
                vol * np.sqrt(dt),
                size=n_sims
            )
            paths[:, t] = paths[:, t-1] * np.exp(shock)

        prices[m] = paths

    return prices

# [nicht in PDF]
def compute_costs(prices_usd, scope1_by_market, coverage_by_year, hedge_ratio, futures_price_usd) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (total_cost_no_hedge, total_cost_with_hedge) for all sims.
    Costs in USD.
    """
    n_sims = next(iter(prices_usd.values())).shape[0]
    total_no = np.zeros(n_sims)
    total_hedged = np.zeros(n_sims)

    years = list(scope1_by_market.keys())

    for j, y in enumerate(years):
        for m in MARKETS:
            emissions_total = scope1_by_market[y][m]  # Scope 1 for market
            coverage = coverage_by_year[y][m]
            emissions_priced = emissions_total * coverage  # effective priced emissions

            spot = prices_usd[m][:, j]  # USD/tCO2
            total_no += emissions_priced * spot

            # Futures hedge (fixed price on hedge_ratio of priced emissions)
            hedged_vol = hedge_ratio * emissions_priced
            unhedged_vol = (1.0 - hedge_ratio) * emissions_priced
            K = futures_price_usd[m]

            total_hedged += hedged_vol * K + unhedged_vol * spot

    return total_no, total_hedged

# [nicht in PDF]
def summarize_costs(no_hedge, with_hedge) -> dict:
    """Return summary metrics."""
    saving = no_hedge - with_hedge
    return {
        "mean_no_hedge": float(no_hedge.mean()),
        "mean_with_hedge": float(with_hedge.mean()),
        "mean_saving": float(saving.mean()),
        "std_no_hedge": float(no_hedge.std()),
        "std_with_hedge": float(with_hedge.std()),
        "p95_no_hedge": float(np.quantile(no_hedge, 0.95)),
        "p95_with_hedge": float(np.quantile(with_hedge, 0.95)),
        "p05_no_hedge": float(np.quantile(no_hedge, 0.05)),
        "p05_with_hedge": float(np.quantile(with_hedge, 0.05)),
    }

# [nicht in PDF]
def run_hedge_ratio_sweep():
    # Build emissions & coverage
    scope1_global = build_scope1_global(YEARS)
    scope1_market = build_scope1_by_market(scope1_global, include_other=INCLUDE_OTHER)
    coverage = build_coverage(YEARS)

    # Build price anchors and simulate prices
    p25_usd, p30_usd = build_price_anchors_usd()
    prices_usd = simulate_prices_gbm_targeted(
        years=YEARS,
        n_sims=N_SIMS,
        p0_usd=p25_usd,
        pT_usd=p30_usd,
        sigma=SIGMA,
        seed=RANDOM_SEED
    )

    rows = []
    # No-hedge baseline computed once (hedge_ratio=0 equals no hedge in this setup)
    # But we compute baseline explicitly for clarity.
    base_no, _ = compute_costs(
        prices_usd=prices_usd,
        scope1_by_market=scope1_market,
        coverage_by_year=coverage,
        hedge_ratio=0.0,
        futures_price_usd=FUTURES_PRICE_USD
    )

    for hr in HEDGE_RATIOS_TO_TEST:
        no, hedged = compute_costs(
            prices_usd=prices_usd,
            scope1_by_market=scope1_market,
            coverage_by_year=coverage,
            hedge_ratio=hr,
            futures_price_usd=FUTURES_PRICE_USD
        )

        # no here == base_no; using it anyway for readability
        s = summarize_costs(no, hedged)
        s["hedge_ratio"] = hr
        rows.append(s)

    df = pd.DataFrame(rows).sort_values("hedge_ratio").reset_index(drop=True)

    # Add some interpretable deltas
    df["p95_reduction"] = df["p95_no_hedge"] - df["p95_with_hedge"]
    df["std_reduction"] = df["std_no_hedge"] - df["std_with_hedge"]

    return df


def make_plots(sweep: pd.DataFrame):
    """Create quick diagnostic plots for hedge ratios."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Headless friendly backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots wurden nicht erzeugt: matplotlib nicht installiert (pip install matplotlib).")
        return

    try:
        plt.style.use("seaborn-v0_8")
    except OSError:
        pass  # style not available; continue with default

    hr = sweep["hedge_ratio"]
    saved = []

    # Erwartete Kosten (Mrd. USD)
    fig1, ax1 = plt.subplots()
    ax1.plot(hr, sweep["mean_no_hedge"] / 1e9, marker="o", label="Mean ohne Hedge")
    ax1.plot(hr, sweep["mean_with_hedge"] / 1e9, marker="o", label="Mean mit Hedge")
    ax1.set_xlabel("Hedge-Ratio")
    ax1.set_ylabel("Kosten (Mrd. USD)")
    ax1.set_title("Erwartete Kosten vs. Hedge-Ratio")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fname = "plot_costs_mean.png"
    fig1.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig1)

    # Risiko: 95. Perzentil (Mrd. USD)
    fig2, ax2 = plt.subplots()
    ax2.plot(hr, sweep["p95_no_hedge"] / 1e9, marker="o", label="P95 ohne Hedge")
    ax2.plot(hr, sweep["p95_with_hedge"] / 1e9, marker="o", label="P95 mit Hedge")
    ax2.set_xlabel("Hedge-Ratio")
    ax2.set_ylabel("95%-Perzentil Kosten (Mrd. USD)")
    ax2.set_title("Risiko (P95) vs. Hedge-Ratio")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fname = "plot_risk_p95.png"
    fig2.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig2)

    # Erwartete Einsparung (Mrd. USD)
    fig3, ax3 = plt.subplots()
    ax3.bar(hr, sweep["mean_saving"] / 1e9, width=0.15)
    ax3.set_xlabel("Hedge-Ratio")
    ax3.set_ylabel("Erwartete Einsparung (Mrd. USD)")
    ax3.set_title("Einsparung durch Hedge vs. Hedge-Ratio")
    ax3.grid(True, axis="y", alpha=0.3)
    fig3.tight_layout()
    fname = "plot_savings.png"
    fig3.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig3)

    # Risk-Return (Std vs. Mean) für Hedged-Kosten
    fig4, ax4 = plt.subplots()
    x_std = sweep["std_with_hedge"] / 1e9
    y_mean = sweep["mean_with_hedge"] / 1e9
    scatter = ax4.scatter(x_std, y_mean, c=hr, cmap="viridis", s=80)
    for i, ratio in enumerate(hr):
        ax4.annotate(f"{ratio:.1f}", (x_std.iloc[i], y_mean.iloc[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    cbar = fig4.colorbar(scatter, ax=ax4)
    cbar.set_label("Hedge-Ratio")
    ax4.set_xlabel("Std dev Kosten (Mrd. USD)")
    ax4.set_ylabel("Erwartete Kosten (Mrd. USD)")
    ax4.set_title("Risk-Return der Hedge-Ratios")
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    fname = "plot_risk_return.png"
    fig4.savefig(fname, dpi=150)
    saved.append(fname)
    plt.close(fig4)

    print("Plots gespeichert:", ", ".join(saved))


# ============================================================
# RUN
# ============================================================

# [nicht in PDF]
if __name__ == "__main__":
    sweep = run_hedge_ratio_sweep()

    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("\n=== Exxon CO2 Hedge Ratio Sweep (USD) ===")
    print(sweep[[
        "hedge_ratio",
        "mean_no_hedge", "mean_with_hedge", "mean_saving",
        "p95_no_hedge", "p95_with_hedge", "p95_reduction",
        "std_no_hedge", "std_with_hedge", "std_reduction"
    ]])

    output_path = "hedge_ratio_sweep.xlsx"
    output_path_csv = "hedge_ratio_sweep.csv"
    try:
        excel_written = False
        try:
            sweep.to_excel(output_path, index=False, decimal=",")
            excel_written = True
        except TypeError:
            sweep.to_excel(output_path, index=False)
            excel_written = True
            print("Hinweis: pandas-Version unterstützt decimal=',' nicht; Excel nutzt Dezimalpunkt '.'.")

        sweep.to_csv(output_path_csv, index=False, sep=";", decimal=",", float_format="%.2f")

        if excel_written:
            print(f"\nExcel export written to {output_path}")
        print(f"Simpler CSV export written to {output_path_csv} (separator=';', decimal=',')")
    except ImportError as exc:
        print(f"\nCould not export to Excel ({exc}). Install openpyxl or xlsxwriter to enable Excel export.")

    make_plots(sweep)
