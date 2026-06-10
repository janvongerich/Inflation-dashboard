
import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Euro Area Inflation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ── Component definitions ─────────────────────────────────────
HICP_COMPONENTS = {
    'TOTAL':          'Headline HICP',
    'TOT_X_NRG_FOOD': 'Core (ex energy & food)',
    'NRG':            'Energy',
    'ELC_GAS':        'Electricity & gas',
    'FUEL':           'Fuels for transport',
    'FOOD':           'Food (total)',
    'FOOD_NP':        'Food - unprocessed',
    'FOOD_P':         'Food - processed',
    'IGD_NNRG':       'Non-energy industrial goods (NEIG)',
    'IGD_NNRG_D':     'NEIG - durable',
    'IGD_NNRG_ND':    'NEIG - non-durable',
    'SERV':           'Services (total)',
    'SERV_HOUS':      'Services - housing & rents',
    'SERV_TRA':       'Services - transport',
    'SERV_REC':       'Services - recreation & culture',
    'SERV_REC_HOA':   'Services - hotels & accommodation',
    'SERV_COM':       'Services - communications',
    'SERV_MSC':       'Services - miscellaneous',
}

KEY_COMPONENTS = [
    'Headline HICP',
    'Core (ex energy & food)',
    'Energy',
    'Food (total)',
    'Food - processed',
    'Food - unprocessed',
    'Non-energy industrial goods (NEIG)',
    'Services (total)',
    'Services - housing & rents',
    'Services - transport',
    'Services - hotels & accommodation',
    'Services - recreation & culture',
    'Services - communications',
    'Services - miscellaneous',
]

# ── Data fetching ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def fetch_hicp_data():
    url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/prc_hicp_minr/?format=TSV&compressed=false"
    r = requests.get(url)
    df = pd.read_csv(io.StringIO(r.text), sep='\t')

    id_col = 'freq,unit,coicop18,geo\\TIME_PERIOD'
    split = df[id_col].str.split(',', expand=True)
    split.columns = ['freq', 'unit', 'coicop18', 'geo']
    df = pd.concat([split, df.drop(columns=[id_col])], axis=1)

    mask = (
        (df['geo'] == 'EA20') &
        (df['unit'] == 'I15') &
        (df['coicop18'].isin(HICP_COMPONENTS.keys()))
    )
    ea = df[mask].copy()

    date_cols = [c for c in ea.columns if c not in ['freq','unit','coicop18','geo']]
    ea_long = ea.melt(id_vars='coicop18', value_vars=date_cols,
                      var_name='date', value_name='index')
    ea_long['date'] = pd.to_datetime(ea_long['date'].str.strip(), format='%Y-%m')
    ea_long['index'] = pd.to_numeric(
        ea_long['index'].astype(str).str.strip().replace(':', None), errors='coerce')
    ea_long = ea_long.dropna(subset=['index'])
    ea_long['name'] = ea_long['coicop18'].map(HICP_COMPONENTS)

    hicp = ea_long.pivot(index='date', columns='name', values='index').sort_index()
    return hicp

# ── Seasonal adjustment ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def seasonal_adjust(_hicp):
    hicp_sa = pd.DataFrame(index=_hicp.index)
    for col in _hicp.columns:
        s = _hicp[col].dropna()
        if len(s) < 24:
            hicp_sa[col] = _hicp[col]
            continue
        stl = STL(s, period=12, robust=True)
        res = stl.fit()
        sa = s - res.seasonal
        hicp_sa[col] = sa
    return hicp_sa

# ── Metric calculations ───────────────────────────────────────
def calc_metrics(hicp):
    def yoy(s):
        return s.pct_change(12, fill_method=None) * 100
    def mom_ann(s, m):
        return ((s / s.shift(m)) ** (12/m) - 1) * 100
    def mom_3m3m_ann(s):
        avg_recent = s.rolling(3).mean()
        avg_prior  = avg_recent.shift(3)
        return ((avg_recent / avg_prior) ** 4 - 1) * 100

    results = {}
    for col in hicp.columns:
        s = hicp[col]
        results[col] = {
            'YoY':       yoy(s),
            '3m/3m ann': mom_3m3m_ann(s),
            '3m ann':    mom_ann(s, 3),
            '6m ann':    mom_ann(s, 6),
        }
    return results

# ── Projection ────────────────────────────────────────────────
def run_projection(results, optimal_lag=8):
    energy  = results['Energy']['YoY'].dropna()
    core    = results['Core (ex energy & food)']['YoY'].dropna()
    services = results['Services (total)']['YoY'].dropna()

    energy_lagged = energy.shift(optimal_lag)
    reg_df = pd.DataFrame({
        'energy_lagged': energy_lagged,
        'core':          core,
        'services':      services,
    }).dropna()

    model_core = LinearRegression().fit(
        reg_df['energy_lagged'].values.reshape(-1,1), reg_df['core'].values)
    model_serv = LinearRegression().fit(
        reg_df['energy_lagged'].values.reshape(-1,1), reg_df['services'].values)

    last_date = energy.index[-1]
    proj_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        end='2026-12-01', freq='MS')

    proj_energy = []
    for d in proj_dates:
        input_date = d - pd.DateOffset(months=optimal_lag)
        if input_date in energy.index:
            proj_energy.append(energy[input_date])
        else:
            proj_energy.append(energy.dropna().iloc[-1])
    proj_energy = pd.Series(proj_energy, index=proj_dates)

    proj_core = model_core.intercept_ + model_core.coef_[0] * proj_energy
    proj_serv = model_serv.intercept_ + model_serv.coef_[0] * proj_energy

    return proj_core, proj_serv, model_core, model_serv, reg_df, energy_lagged, last_date

# ── App layout ────────────────────────────────────────────────
st.title("📊 Euro Area Inflation Dashboard")

# Sidebar
st.sidebar.header("Settings")

if st.sidebar.button("🔄 Refresh data"):
    st.cache_data.clear()
    st.rerun()

use_sa = st.sidebar.checkbox("Seasonal adjustment (STL)", value=True)
history_months = st.sidebar.selectbox(
    "History window", [12, 24, 36, 60], index=2)

module = st.sidebar.radio("Select module", [
    "Summary table",
    "Headline vs Core vs Services",
    "Momentum dashboard",
    "Services breakdown",
    "Energy pass-through",
    "Projection",
])

# Load data
with st.spinner("Loading HICP data from Eurostat..."):
    hicp_raw = fetch_hicp_data()

if use_sa:
    with st.spinner("Applying STL seasonal adjustment..."):
        hicp = seasonal_adjust(hicp_raw)
else:
    hicp = hicp_raw

results = calc_metrics(hicp)
latest_date = hicp.index[-1].strftime("%B %Y")
st.sidebar.markdown(f"**Latest data:** {latest_date}")

cutoff = hicp.index[-1] - pd.DateOffset(months=history_months)

# ── Module 1: Summary table ───────────────────────────────────
if module == "Summary table":
    st.subheader("Inflation momentum — all components")
    summary = pd.DataFrame({
        'YoY %':       {k: results[k]['YoY'].iloc[-1] for k in KEY_COMPONENTS},
        '3m/3m ann %': {k: results[k]['3m/3m ann'].iloc[-1] for k in KEY_COMPONENTS},
        '3m ann %':    {k: results[k]['3m ann'].iloc[-1] for k in KEY_COMPONENTS},
        '6m ann %':    {k: results[k]['6m ann'].iloc[-1] for k in KEY_COMPONENTS},
    }).round(2)

    def color_cells(val):
        if val > 4:   return 'background-color: #ff4444; color: white'
        elif val > 2: return 'background-color: #ffaa00; color: white'
        elif val < 0: return 'background-color: #4444ff; color: white'
        else:         return 'background-color: #44bb44; color: white'

    st.dataframe(summary.style.applymap(color_cells), use_container_width=True)

# ── Module 2: Headline vs Core vs Services ────────────────────
elif module == "Headline vs Core vs Services":
    st.subheader("Headline vs Core vs Services — YoY %")
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, color, ls in [
        ('Headline HICP',          'black', '-'),
        ('Core (ex energy & food)', 'red',   '--'),
        ('Services (total)',        'blue',  ':'),
        ('Energy',                  'orange','-.'),
    ]:
        s = results[label]['YoY']
        s = s.loc[s.index >= cutoff]
        ax.plot(s.index, s, color=color, linewidth=2, linestyle=ls, label=label)
    ax.axhline(2, color='grey', linewidth=1, linestyle='--', alpha=0.5, label='ECB target')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('%')
    st.pyplot(fig)

# ── Module 3: Momentum dashboard ─────────────────────────────
elif module == "Momentum dashboard":
    st.subheader("Momentum — 3m/3m annualised % (seasonally adjusted)")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    agg = ['Headline HICP', 'Core (ex energy & food)', 'Services (total)', 'Energy']
    colors = ['black', 'red', 'blue', 'orange']
    for label, color in zip(agg, colors):
        s = results[label]['3m/3m ann']
        s = s.loc[s.index >= cutoff]
        axes[0].plot(s.index, s, color=color, linewidth=2, label=label)
    axes[0].axhline(2, color='grey', linewidth=1, linestyle='--', alpha=0.5)
    axes[0].set_title('Key aggregates')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    food = ['Food (total)', 'Food - processed', 'Food - unprocessed']
    food_colors = ['green', 'limegreen', 'darkgreen']
    for label, color in zip(food, food_colors):
        s = results[label]['3m/3m ann']
        s = s.loc[s.index >= cutoff]
        axes[1].plot(s.index, s, color=color, linewidth=2, label=label)
    axes[1].axhline(2, color='grey', linewidth=1, linestyle='--', alpha=0.5)
    axes[1].set_title('Food components')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

# ── Module 4: Services breakdown ──────────────────────────────
elif module == "Services breakdown":
    st.subheader("Services subcomponents — 3m/3m annualised %")
    services = [
        'Services - housing & rents',
        'Services - transport',
        'Services - hotels & accommodation',
        'Services - recreation & culture',
        'Services - communications',
        'Services - miscellaneous',
    ]
    svc_colors = ['#2ca02c','#98df8a','#b5cf6b','#637939','#17becf','#9edae5']
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, color in zip(services, svc_colors):
        s = results[label]['3m/3m ann']
        s = s.loc[s.index >= cutoff]
        ax.plot(s.index, s, color=color, linewidth=2,
                label=label.replace('Services - ', ''))
    ax.axhline(2, color='grey', linewidth=1, linestyle='--', alpha=0.5, label='ECB target')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylabel('%')
    st.pyplot(fig)

# ── Module 5: Energy pass-through ────────────────────────────
elif module == "Energy pass-through":
    st.subheader("Energy price pass-through — lag correlation analysis")
    energy  = results['Energy']['YoY'].dropna()
    core    = results['Core (ex energy & food)']['YoY'].dropna()
    services = results['Services (total)']['YoY'].dropna()

    df_trans = pd.DataFrame({
        'Energy':   energy,
        'Core':     core,
        'Services': services,
    }).dropna()

    lags = range(0, 19)
    corr_core = [df_trans['Energy'].shift(l).corr(df_trans['Core']) for l in lags]
    corr_serv = [df_trans['Energy'].shift(l).corr(df_trans['Services']) for l in lags]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(list(lags), corr_core, marker='o', color='red', linewidth=2, label='Core')
    axes[0].plot(list(lags), corr_serv, marker='o', color='blue', linewidth=2, label='Services')
    axes[0].axhline(0, color='grey', linewidth=0.8)
    axes[0].set_xlabel('Lag (months)')
    axes[0].set_ylabel('Correlation')
    axes[0].set_title('Correlation of energy YoY with core/services at different lags')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    optimal_lag = int(pd.Series(corr_core).abs().idxmax())
    energy_shifted = df_trans['Energy'].shift(optimal_lag)
    energy_shifted = energy_shifted.loc[energy_shifted.index >= cutoff]

    ax2 = axes[1]
    ax2.plot(energy_shifted.index, energy_shifted,
             color='orange', linewidth=2, label=f'Energy YoY (lagged {optimal_lag}m)')
    ax2_twin = ax2.twinx()
    core_plot = df_trans['Core'].loc[df_trans['Core'].index >= cutoff]
    serv_plot = df_trans['Services'].loc[df_trans['Services'].index >= cutoff]
    ax2_twin.plot(core_plot.index, core_plot,
                  color='red', linewidth=2, linestyle='--', label='Core YoY (right)')
    ax2_twin.plot(serv_plot.index, serv_plot,
                  color='blue', linewidth=2, linestyle=':', label='Services YoY (right)')
    ax2.set_ylabel('Energy YoY %', color='orange')
    ax2_twin.set_ylabel('Core / Services YoY %')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title(f'Energy (lagged {optimal_lag}m) vs Core & Services')

    plt.tight_layout()
    st.pyplot(fig)

# ── Module 6: Projection ──────────────────────────────────────
elif module == "Projection":
    st.subheader("Core & Services projection — energy pass-through model")
    proj_core, proj_serv, model_core, model_serv, reg_df, energy_lagged, last_date = run_projection(results)

    r2_core = model_core.score(
        reg_df['energy_lagged'].values.reshape(-1,1), reg_df['core'].values)
    r2_serv = model_serv.score(
        reg_df['energy_lagged'].values.reshape(-1,1), reg_df['services'].values)

    col1, col2 = st.columns(2)
    col1.metric("Core model R²", f"{r2_core:.2f}")
    col2.metric("Services model R²", f"{r2_serv:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    configs = [
        ('Core (ex energy & food)', proj_core, model_core, 'core', 'red'),
        ('Services (total)',        proj_serv, model_serv, 'services', 'blue'),
    ]

    for ax, (label, proj, model, reg_key, color) in zip(axes, configs):
        hist = results[label]['YoY'].loc[results[label]['YoY'].index >= cutoff]
        ax.plot(hist.index, hist, color=color, linewidth=2.5, label=f'{label} — actual')

        fitted = model.intercept_ + model.coef_[0] * energy_lagged
        fitted_recent = fitted.loc[fitted.index >= cutoff]
        ax.plot(fitted_recent.index, fitted_recent,
                color=color, linewidth=1.5, linestyle=':', alpha=0.6, label='Model fit')

        last_actual = pd.Series([hist.iloc[-1]], index=[hist.index[-1]])
        proj_connected = pd.concat([last_actual, proj])
        ax.plot(proj_connected.index, proj_connected,
                color=color, linewidth=2, linestyle='--', label='Projection')

        resid_std = (reg_df[reg_key] - (model.intercept_ + model.coef_[0] * reg_df['energy_lagged'])).std()
        ax.fill_between(proj.index,
                        proj - 1.5 * resid_std,
                        proj + 1.5 * resid_std,
                        color=color, alpha=0.1, label='±1.5 std band')

        ax.axhline(2, color='grey', linewidth=1, linestyle='--', alpha=0.5, label='ECB target')
        ax.axvline(last_date, color='grey', linewidth=1.5, linestyle=':', alpha=0.7)
        ax.set_title(f'{label} — YoY % with projection')
        ax.set_ylabel('%')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)
