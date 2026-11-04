"""
INSIGHT ATLAS  -  one-file interactive dashboard for the Data Visualization Lab (BISL504-C)
Dataset : global_socioeconomic_trends.csv  (194 countries, 1994-2024, practice dataset)

Run     :  pip install streamlit plotly pandas numpy scikit-learn statsmodels jinja2 matplotlib
           streamlit run insight_atlas.py
Place global_socioeconomic_trends.csv (or the slim copy) next to this file, or upload it on the first screen.

The app is a guided tour: one slide at a time (Back / Next / Contents in the top bar),
chart on the left, story panel (what it is, how it is made, takeaway) on the right.
"""
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Insight Atlas", page_icon="\U0001F310", layout="wide", initial_sidebar_state="collapsed")

# ====================================================================== design tokens
TEAL, BLUE, VIOLET, GOLD, ROSE, SLATE = "#2DD4BF", "#60A5FA", "#A78BFA", "#F5C451", "#FB7185", "#94A3B8"
CONT_COL = {"Africa": ROSE, "Asia": BLUE, "Europe": TEAL, "North America": GOLD, "Oceania": VIOLET, "South America": SLATE}
INCOME_ORDER = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
MAP_SCALE = ["#14213d", "#1f4e79", "#2a7f9e", "#2DD4BF", "#b7e36a", "#F5C451"]
FONT = "Inter, Segoe UI, sans-serif"
px.defaults.color_discrete_sequence = [BLUE, TEAL, GOLD, ROSE, VIOLET, "#38BDF8", "#F472B6", "#A3E635"]
UP, DN, ARW = "▲", "▼", " → "
PLOTLY_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif !important; }
.stApp { background:
   radial-gradient(900px 520px at 8% -8%, rgba(96,165,250,.16), transparent 60%),
   radial-gradient(800px 480px at 100% 0%, rgba(45,212,191,.13), transparent 58%),
   radial-gradient(700px 500px at 60% 110%, rgba(167,139,250,.10), transparent 60%), #070B14; color: #E6EAF2; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.block-container { padding-top: .8rem; padding-bottom: 2.2rem; max-width: 1480px; }
h1, h2, h3, h4 { letter-spacing: -.01em; }

/* ---- top bar */
.st-key-topbar { position: sticky; top: .4rem; z-index: 60; padding-bottom: 22px !important; background: rgba(9,14,26,.78); backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
   border: 1px solid rgba(255,255,255,.09); border-radius: 18px; padding: 10px 18px; margin-bottom: 14px; box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.brand { display: flex; align-items: center; gap: 12px; }
.brand .nm { font-weight: 800; font-size: 1.15rem; line-height: 1.1; background: linear-gradient(90deg,#fff,#a5f3fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.brand .sb { font-size: .72rem; color: #94A3B8; letter-spacing: .06em; text-transform: uppercase; }
.stepinfo { text-align: center; }
.stepinfo .grp { font-size: .68rem; letter-spacing: .14em; color: #2DD4BF; font-weight: 700; text-transform: uppercase; }
.stepinfo .ttl { font-size: 1.05rem; font-weight: 700; color: #fff; }
.prog { display: flex; gap: 4px; margin-top: 6px; }
.prog i { flex: 1; height: 4px; border-radius: 4px; background: rgba(255,255,255,.10); }
.prog i.done { background: linear-gradient(90deg,#2DD4BF,#60A5FA); opacity: .55; }
.prog i.on { background: linear-gradient(90deg,#F5C451,#fb923c); box-shadow: 0 0 12px rgba(245,196,81,.7); }
.stButton > button { border-radius: 12px; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.05); color: #E6EAF2; font-weight: 600; transition: all .2s; }
.stButton > button:hover { border-color: #2DD4BF; color: #fff; transform: translateY(-1px); }
.st-key-next_top button, .st-key-next_bot button, .st-key-start_btn button { background: linear-gradient(90deg,#2DD4BF,#60A5FA) !important; color: #05121c !important; border: 0 !important; font-weight: 700 !important; box-shadow: 0 6px 22px rgba(45,212,191,.28); }
.st-key-next_top button:hover, .st-key-next_bot button:hover, .st-key-start_btn button:hover { filter: brightness(1.08); transform: translateY(-2px); }

/* ---- cards */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 20px !important; border: 1px solid rgba(255,255,255,.09) !important; background: linear-gradient(160deg, rgba(255,255,255,.055), rgba(255,255,255,.02)); }
.kick { font-size: .72rem; letter-spacing: .16em; font-weight: 700; color: #F5C451; text-transform: uppercase; }
.h { font-size: 1.55rem; font-weight: 800; line-height: 1.15; margin: 2px 0 6px 0; color: #fff; }
.d { color: #AEB9CC; font-size: .95rem; line-height: 1.5; margin-bottom: 10px; }
.pill { display: inline-block; font-size: .74rem; padding: 3px 11px; border-radius: 30px; margin: 0 6px 6px 0; background: rgba(96,165,250,.12); border: 1px solid rgba(96,165,250,.28); color: #cfe3ff; }
.pill b { color: #fff; font-weight: 600; margin-right: 4px; }
.pill.g { background: rgba(245,196,81,.10); border-color: rgba(245,196,81,.30); color: #fde7b0; }
.take { margin-top: 12px; padding: 12px 15px; border-radius: 14px; background: linear-gradient(135deg, rgba(45,212,191,.12), rgba(96,165,250,.06)); border: 1px solid rgba(45,212,191,.25); color: #E3EEF6; font-size: .92rem; line-height: 1.5; }
.take span { display: block; font-size: .66rem; letter-spacing: .16em; color: #2DD4BF; font-weight: 700; text-transform: uppercase; margin-bottom: 3px; }
.take b { color: #F5C451; }

/* ---- hero, tiles, kpi */
.hero { position: relative; overflow: hidden; border-radius: 26px; padding: 34px 40px; margin-bottom: 14px; border: 1px solid transparent;
   background: linear-gradient(135deg,#0d1730,#0b2230 55%,#1b1a35) padding-box, linear-gradient(120deg, rgba(45,212,191,.7), rgba(96,165,250,.45), rgba(167,139,250,.6)) border-box; }
.hero:before { content:""; position:absolute; right:-80px; top:-90px; width:380px; height:380px; border-radius:50%; background: radial-gradient(circle, rgba(96,165,250,.28), transparent 65%); }
.hero:after { content:""; position:absolute; left:30%; bottom:-140px; width:420px; height:300px; border-radius:50%; background: radial-gradient(circle, rgba(167,139,250,.18), transparent 65%); }
.hero .row { display:flex; align-items:center; gap:30px; position:relative; z-index:2; }
.hero h1 { margin:0; font-size:3.1rem; font-weight:800; background: linear-gradient(90deg,#fff 10%,#a5f3fc 55%,#F5C451); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.05; }
.hero p { color:#C3CEE0; font-size:1.08rem; max-width:760px; margin:10px 0 14px 0; line-height:1.55; }
.chip { display:inline-block; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14); border-radius:30px; padding:4px 14px; font-size:.8rem; color:#eef2fb; margin:0 8px 6px 0; }
.globe { animation: bob 6s ease-in-out infinite; filter: drop-shadow(0 14px 30px rgba(45,212,191,.35)); }
.orb { transform-box: view-box; transform-origin: 50% 50%; animation: spin 16s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
@keyframes bob { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-8px); } }
.tile { border-radius:16px; padding:14px 16px; background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.08); }
.tile small { display:block; color:#94A3B8; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; }
.tile b { font-size:1.6rem; font-weight:800; color:#fff; }
.kpi { border-radius:18px; padding:14px 16px 8px 16px; background: linear-gradient(160deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border:1px solid rgba(255,255,255,.09); border-top:3px solid var(--c,#2DD4BF); transition: transform .2s, box-shadow .2s; }
.kpi:hover { transform: translateY(-4px); box-shadow: 0 14px 34px rgba(0,0,0,.35); }
.kpi .l { color:#94A3B8; font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; }
.kpi .v { font-size:1.75rem; font-weight:800; color:#fff; line-height:1.2; }
.kpi .dl { font-size:.78rem; margin-top:2px; }
.kpi svg { display:block; margin-top:4px; }
.step { display:flex; gap:12px; align-items:flex-start; margin-bottom:12px; }
.step .n { flex:0 0 30px; height:30px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-weight:800; color:#07121b; background:linear-gradient(135deg,#F5C451,#fb923c); }
.step div { color:#C3CEE0; font-size:.93rem; line-height:1.45; } .step b { color:#fff; }
.spec { display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin: 2px 0 14px 0; }
.spec div { border-radius:14px; padding:11px 14px; background:rgba(96,165,250,.07); border:1px solid rgba(96,165,250,.20); }
.spec small { display:block; color:#7FB1F2; font-size:.64rem; letter-spacing:.16em; text-transform:uppercase; font-weight:700; }
.spec b { color:#fff; font-size:.95rem; font-weight:600; }
.spot { border-radius:16px; padding:14px 16px; background: linear-gradient(135deg, rgba(96,165,250,.14), rgba(45,212,191,.10)); border:1px solid rgba(96,165,250,.30); }
.spot h4 { margin:0 0 6px 0; color:#F5C451; }
.foot { text-align:center; color:#64748B; font-size:.76rem; margin-top:18px; }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:rgba(255,255,255,.04); padding:5px; border-radius:14px; border:1px solid rgba(255,255,255,.08); }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:7px 16px; height:auto; color:#9fb0c8; font-weight:600; }
.stTabs [aria-selected="true"] { background:linear-gradient(90deg,#2DD4BF,#60A5FA); color:#05121c !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
div[data-baseweb="select"] > div { border-radius:12px; background: rgba(255,255,255,.05); }
.stMultiSelect div[data-baseweb="tag"] { background: rgba(96,165,250,.25) !important; color:#fff; border-radius:8px; }
[data-testid="stExpander"] { border-radius:16px; border:1px solid rgba(255,255,255,.09); background: rgba(255,255,255,.03); }
@media (max-width: 900px) { .hero h1 { font-size:2.1rem; } .spec { grid-template-columns: 1fr; } .hero .row { flex-direction:column; align-items:flex-start; } }
</style>
""",
    unsafe_allow_html=True,
)


def logo(size=64, animate=False):
    cls = "globe" if animate else ""
    return f"""<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<defs><radialGradient id="gb" cx="34%" cy="28%" r="80%"><stop offset="0" stop-color="#9be7ff"/><stop offset=".45" stop-color="#2a7fd0"/><stop offset="1" stop-color="#0b1f4d"/></radialGradient>
<linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2DD4BF"/><stop offset="1" stop-color="#60A5FA"/></linearGradient></defs>
<circle cx="50" cy="50" r="33" fill="url(#gb)"/>
<path d="M31 36c5-6 13-8 17-4 3 3-1 6 3 9s9 0 8 6-8 6-10 12-8 7-12 3-6-9-4-14-5-8-2-12z" fill="#2DD4BF" fill-opacity=".72"/>
<path d="M58 28c4-2 9 0 9 4s-5 5-8 3-3-5-1-7z M62 52c4-1 8 2 7 6s-6 5-8 2-1-7 1-8z" fill="#2DD4BF" fill-opacity=".6"/>
<ellipse cx="50" cy="50" rx="33" ry="11" fill="none" stroke="#d6f4ff" stroke-opacity=".38" stroke-width="1"/>
<ellipse cx="50" cy="50" rx="13" ry="33" fill="none" stroke="#d6f4ff" stroke-opacity=".38" stroke-width="1"/>
<circle cx="50" cy="50" r="33" fill="none" stroke="url(#rg)" stroke-width="2.2"/>
<g class="orb"><ellipse cx="50" cy="50" rx="46" ry="15" fill="none" stroke="#F5C451" stroke-opacity=".85" stroke-width="1.6" transform="rotate(-24 50 50)"/>
<circle cx="93" cy="38" r="3.6" fill="#F5C451" transform="rotate(-24 50 50)"/></g></svg>"""


def style(fig, h=440, legend=True):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=6, r=6, t=34, b=6), font=dict(family=FONT, size=12.5, color="#CBD5E1"), showlegend=legend,
                      legend=dict(orientation="h", y=-0.16, x=0, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
                      hoverlabel=dict(bgcolor="#0B1220", bordercolor="#2DD4BF", font=dict(family=FONT, size=12)), title_font=dict(size=14))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False, linecolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zeroline=False)
    return fig


COLS = ["Country", "ISO3_Code", "Continent", "Income_Group", "Development_Status", "Year", "Economic_Sector", "Population_Total",
        "GDP_Per_Capita_USD", "Life_Expectancy_Years", "Human_Development_Index", "Happiness_Index", "Internet_Penetration_Pct",
        "CO2_Emissions_Per_Capita_Tons", "Literacy_Rate_Pct", "Urban_Population_Pct", "Health_Expenditure_Pct_GDP",
        "Fertility_Rate_Births_Per_Woman", "Gini_Coefficient", "Corruption_Perceptions_Index", "Sector_GDP_Contribution_Pct",
        "Unemployment_Rate_Pct", "Physicians_Per_1000", "Access_To_Electricity_Pct", "Education_Expenditure_Pct_GDP",
        "Median_Age_Years", "Renewable_Energy_Share_Pct", "Democracy_Index"]
NICE = {
    "Human_Development_Index": "Human Development Index", "Life_Expectancy_Years": "Life expectancy (years)",
    "GDP_Per_Capita_USD": "GDP per capita (USD)", "Internet_Penetration_Pct": "Internet users (%)",
    "CO2_Emissions_Per_Capita_Tons": "CO2 per person (tons)", "Happiness_Index": "Happiness index",
    "Literacy_Rate_Pct": "Literacy rate (%)", "Urban_Population_Pct": "Urban population (%)",
    "Health_Expenditure_Pct_GDP": "Health spending (% GDP)", "Fertility_Rate_Births_Per_Woman": "Fertility rate",
    "Gini_Coefficient": "Gini (inequality)", "Corruption_Perceptions_Index": "Corruption perception index",
    "Unemployment_Rate_Pct": "Unemployment (%)", "Physicians_Per_1000": "Doctors per 1000", "Access_To_Electricity_Pct": "Electricity access (%)",
    "Education_Expenditure_Pct_GDP": "Education spending (% GDP)", "Median_Age_Years": "Median age", "Renewable_Energy_Share_Pct": "Renewable energy (%)",
    "Democracy_Index": "Democracy index",
}


@st.cache_data(show_spinner="Loading dataset...")
def load(src):
    df = pd.read_csv(src, usecols=lambda c: c in COLS)
    cy = df.drop_duplicates(["Country", "Year"]).drop(columns=["Economic_Sector", "Sector_GDP_Contribution_Pct"]).reset_index(drop=True)
    sec = df[df.Economic_Sector != "Overall_National"][["Country", "Continent", "Income_Group", "Year", "Economic_Sector", "Sector_GDP_Contribution_Pct"]]
    try:
        if hasattr(src, "seek"):
            src.seek(0)
        ncols = len(pd.read_csv(src, nrows=0).columns)
        if ncols < 100:  # slimmed copy of the dataset: the original file has 106 columns
            ncols = 106
    except Exception:
        ncols = df.shape[1]
    return cy, sec, len(df), ncols


def find_csv():
    here = os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(here, "global_socioeconomic_trends.csv"), os.path.join(here, "global_socioeconomic_trends_slim.csv"),
              "global_socioeconomic_trends.csv", "global_socioeconomic_trends_slim.csv", r"C:\DVP\global_socioeconomic_trends.csv"]:
        if os.path.exists(p):
            return p
    return None


path, up = find_csv(), None
if path is None:
    st.markdown(f'<div class="hero"><div class="row">{logo(90, True)}<div><h1>Insight Atlas</h1><p>Upload the dataset to begin.</p></div></div></div>', unsafe_allow_html=True)
    up = st.file_uploader("Upload global_socioeconomic_trends.csv", type="csv")
    if up is None:
        st.stop()
cy, sec, n_rows, n_cols_raw = load(up if up is not None else path)
y0, y1 = int(cy.Year.min()), int(cy.Year.max())
ALLC, ALLI = sorted(cy.Continent.unique()), INCOME_ORDER

# ====================================================================== guided tour
STEPS = [
    ("Welcome", "Welcome"), ("The world at a glance", "Insights"), ("Income and life expectancy", "Insights"), ("Life expectancy by continent", "Insights"),
    ("The digital divide", "Insights"), ("Development across income groups", "Insights"), ("The sector mix of economies", "Insights"),
    ("Indicators linked to life expectancy", "Insights"), ("Emissions and income", "Insights"), ("Leaders and biggest improvers", "Insights"),
    ("Population and wealth", "Insights"), ("Country Explorer", "Explorer"), ("Predict life expectancy", "Machine learning"),
    ("Group countries into tiers", "Machine learning"), ("Forecast to 2030", "Machine learning"), ("Conclusions", "Conclusions"),
]
N = len(STEPS)
try:
    _qs = int(st.query_params.get("step", 0))
except Exception:
    _qs = 0
for k_, v_ in {"step": max(0, min(N - 1, _qs)), "f_cont": ALLC, "f_inc": ALLI, "f_yr": (y0, y1), "f_fy": y1}.items():
    if k_ not in st.session_state:
        st.session_state[k_] = v_


def goto(n):
    st.session_state.step = max(0, min(N - 1, n))
    st.query_params["step"] = str(st.session_state.step)


def _reset():
    st.session_state.f_cont, st.session_state.f_inc, st.session_state.f_yr, st.session_state.f_fy = ALLC, ALLI, (y0, y1), y1


step = st.session_state.step
with st.container(key="topbar"):
    c1, c2, c3, c4, c5 = st.columns([2.1, 0.9, 3.6, 0.9, 1.0], vertical_alignment="center")
    c1.markdown(f'<div class="brand">{logo(46)}<div><div class="nm">Insight Atlas</div><div class="sb">Global socioeconomic trends</div></div></div>', unsafe_allow_html=True)
    c2.button("\u2190 Back", key="back_top", on_click=goto, args=(step - 1,), disabled=step == 0, width="stretch")
    bars = "".join(f'<i class="{"on" if i == step else ("done" if i < step else "")}"></i>' for i in range(N))
    c3.markdown(f'<div class="stepinfo"><div class="grp">{STEPS[step][1]} \u00b7 {step + 1} of {N}</div><div class="ttl">{STEPS[step][0]}</div><div class="prog">{bars}</div></div>', unsafe_allow_html=True)
    c4.button("Next \u2192", key="next_top", on_click=goto, args=(step + 1,), disabled=step == N - 1, width="stretch")
    with c5.popover("\u2630 Contents", width="stretch"):
        for i, (t_, g_) in enumerate(STEPS):
            st.button(f"{i + 1:02d}  {t_}", key=f"jump{i}", on_click=goto, args=(i,), width="stretch", type="primary" if i == step else "secondary")

with st.expander("\U0001F39B  Filters  (apply to the 10 insight slides)", expanded=False):
    fc = st.columns([1.5, 1.5, 1.4, 1.2, 0.8])
    conts = fc[0].multiselect("Continent", ALLC, key="f_cont")
    incs = fc[1].multiselect("Income group", ALLI, key="f_inc")
    yr = fc[2].slider("Year range", y0, y1, key="f_yr")
    fy = fc[3].slider("Focus year", y0, y1, key="f_fy")
    fc[4].markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
    fc[4].button("\u21BA Reset", on_click=_reset, width="stretch")

mask = cy.Continent.isin(conts) & cy.Income_Group.isin(incs) & cy.Year.between(*yr)
f = cy[mask]
fyr = f[f.Year == min(max(fy, yr[0]), yr[1])]
fsec = sec[sec.Continent.isin(conts) & sec.Income_Group.isin(incs) & sec.Year.between(*yr)]
st.markdown(f'<div style="margin:-4px 0 10px 4px;color:#7F8CA3;font-size:.8rem">Selection: <b style="color:#cfe3ff">{len(conts)}</b> continents \u00b7 <b style="color:#cfe3ff">{len(incs)}</b> income groups \u00b7 '
            f'<b style="color:#cfe3ff">{yr[0]}\u2013{yr[1]}</b> \u00b7 focus year <b style="color:#cfe3ff">{fy}</b> \u00b7 {len(f):,} country-years</div>', unsafe_allow_html=True)


# ====================================================================== helpers
def spark(vals, color="#F5C451", w=160, h=34):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2 or np.nanmax(v) == np.nanmin(v):
        return ""
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 3 - (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) * (h - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="{pts}"/></svg>'


def kpi(col, label, value, delta=None, good_up=True, series=None, color=TEAL):
    d = ""
    if delta is not None:
        up_ = delta >= 0
        c = "#34D399" if (up_ == good_up) else "#FB7185"
        d = f'<div class="dl" style="color:{c}">{UP if up_ else DN} {abs(delta):,.2f} vs all data</div>'
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div><div class="v">{value}</div>{d}{spark(series, color) if series is not None else ""}</div>', unsafe_allow_html=True)


def pills(items):
    return "".join(f'<span class="pill{" g" if i % 2 else ""}"><b>{a}</b>{b}</span>' for i, (a, b) in enumerate(items))


def panel(n, title, desc, chips):
    st.markdown(f'<div class="kick">Insight {n:02d}</div><div class="h">{title}</div><div class="d">{desc}</div><div>{pills(chips)}</div>', unsafe_allow_html=True)


def take(text):
    st.markdown(f'<div class="take"><span>Takeaway</span>{text}</div>', unsafe_allow_html=True)


def spec(items):
    st.markdown('<div class="spec">' + "".join(f"<div><small>{a}</small><b>{b}</b></div>" for a, b in items) + "</div>", unsafe_allow_html=True)


def slide(ratio=(2.25, 1)):
    return st.columns(list(ratio), gap="large")


# ====================================================================== ML (cached)
FEATS = ["GDP_Per_Capita_USD", "Literacy_Rate_Pct", "Urban_Population_Pct", "Internet_Penetration_Pct", "Health_Expenditure_Pct_GDP", "Fertility_Rate_Births_Per_Woman"]
FNAMES = ["ln(GDP per capita)", "Literacy rate (%)", "Urban population (%)", "Internet users (%)", "Health spending (% GDP)", "Fertility rate"]
TARGET = "Life_Expectancy_Years"


def design(d):
    X = d[FEATS].copy()
    X["GDP_Per_Capita_USD"] = np.log(X["GDP_Per_Capita_USD"])
    X.columns = FNAMES
    return X


@st.cache_resource(show_spinner="Training and comparing models (first visit only)...")
def train_models(_cy, key):
    X, y, grp = design(_cy), _cy[TARGET], _cy.Country
    cv = GroupKFold(5)
    models = {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(1.0)),
        "k-Nearest Neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(10)),
        "Random Forest": RandomForestRegressor(150, min_samples_leaf=3, n_jobs=-1, random_state=1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=.06, random_state=1),
    }
    rows, preds = [], {}
    for n, m in models.items():
        p = cross_val_predict(m, X, y, groups=grp, cv=cv)
        preds[n] = p
        rows.append({"Model": n, "R-squared": r2_score(y, p), "MAE (years)": mean_absolute_error(y, p)})
    res = pd.DataFrame(rows).sort_values("R-squared", ascending=False).reset_index(drop=True)
    ridge = models["Ridge regression"].fit(X, y)
    sc, rg = ridge.named_steps["standardscaler"], ridge.named_steps["ridge"]
    coef = pd.Series(rg.coef_ / sc.scale_, index=FNAMES)
    icpt = rg.intercept_ - float((rg.coef_ * sc.mean_ / sc.scale_).sum())
    return res, preds["Ridge regression"], coef, icpt, X.mean(), X.std()


@st.cache_data(show_spinner=False)
def build_clusters(_cy, k, key):
    prof_cols = ["GDP_Per_Capita_USD", "Life_Expectancy_Years", "Literacy_Rate_Pct", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons",
                 "Happiness_Index", "Gini_Coefficient", "Urban_Population_Pct"]
    last = _cy[_cy.Year >= _cy.Year.max() - 2].groupby(["Country", "Continent", "ISO3_Code"], as_index=False)[prof_cols + ["Human_Development_Index"]].mean()
    Z = last[prof_cols].copy()
    Z["GDP_Per_Capita_USD"] = np.log(Z["GDP_Per_Capita_USD"])
    Z["CO2_Emissions_Per_Capita_Tons"] = np.log1p(Z["CO2_Emissions_Per_Capita_Tons"])
    Zs = StandardScaler().fit_transform(Z)
    km = KMeans(k, n_init=10, random_state=1).fit(Zs)
    last["cl"] = km.labels_
    order = last.groupby("cl").Human_Development_Index.mean().sort_values().index.tolist()
    names_by_k = {3: ["Developing", "Emerging", "Advanced"], 4: ["Developing", "Emerging", "Upper tier", "Advanced"],
                  5: ["Struggling", "Developing", "Emerging", "Upper tier", "Advanced"], 6: ["Struggling", "Developing", "Catching up", "Emerging", "Upper tier", "Advanced"]}
    mp = {c: f"Tier {i + 1}: {names_by_k[k][i]}" for i, c in enumerate(order)}
    last["Cluster"] = last.cl.map(mp)
    pc = PCA(2, random_state=1).fit(Zs)
    last[["PC1", "PC2"]] = pc.transform(Zs)
    sil = silhouette_score(Zs, km.labels_)
    prof = last.groupby("Cluster")[prof_cols + ["Human_Development_Index"]].mean()
    prof.insert(0, "Countries", last.groupby("Cluster").size())
    return last, prof, sil, pc.explained_variance_ratio_.sum()


@st.cache_data(show_spinner=False)
def forecast_table(_cy, metric, horizon, key):
    g = _cy.groupby(["Continent", "Year"], as_index=False)[metric].mean()
    rows, bt = [], []
    for c_, d_ in g.groupby("Continent"):
        d_ = d_.sort_values("Year")
        tr = d_[d_.Year >= d_.Year.max() - 14]
        mdl = sm.OLS(tr[metric].values, sm.add_constant(tr.Year.values.astype(float))).fit()
        fut = np.arange(d_.Year.max() + 1, horizon + 1).astype(float)
        pr = mdl.get_prediction(sm.add_constant(fut, has_constant="add")).summary_frame(alpha=0.2)
        for yy, m_, lo, hi in zip(fut, pr["mean"], pr["obs_ci_lower"], pr["obs_ci_upper"]):
            rows.append({"Continent": c_, "Year": int(yy), "Value": m_, "Lower": lo, "Upper": hi})
        tr2, te2 = d_[(d_.Year >= 2004) & (d_.Year <= 2018)], d_[d_.Year >= 2019]
        m2 = sm.OLS(tr2[metric].values, sm.add_constant(tr2.Year.values.astype(float))).fit()
        p2 = m2.predict(sm.add_constant(te2.Year.values.astype(float), has_constant="add"))
        bt.append({"Continent": c_, "Trend-model error": mean_absolute_error(te2[metric], p2),
                   "'Same as last year' error": mean_absolute_error(te2[metric], np.repeat(tr2[metric].iloc[-1], len(te2)))})
    return pd.DataFrame(rows), pd.DataFrame(bt)


# ====================================================================== slides
def s_welcome():
    st.markdown(f'<div class="hero"><div class="row">{logo(150, True)}<div><div class="kick">Data Visualization Lab \u00b7 BISL504-C</div><h1>Insight Atlas</h1>'
                f'<p>Thirty years of global development in one guided tour. Explore ten interactive graphs, compare countries, then let machine learning predict, group and forecast.</p>'
                f'<span class="chip">\U0001F4CA {n_rows:,} rows</span><span class="chip">\U0001F9E9 {n_cols_raw} indicators</span><span class="chip">\U0001F30D {cy.Country.nunique()} countries</span>'
                f'<span class="chip">\U0001F4C5 {y0}\u2013{y1}</span><span class="chip">\U0001F916 3 ML models</span></div></div></div>', unsafe_allow_html=True)
    a, b = st.columns([1.35, 1], gap="large")
    with a:
        with st.container(border=True):
            st.markdown('<div class="kick">About</div><div class="h">What is Insight Atlas</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="d">Insight Atlas turns a large socioeconomic table into a story you can click through. It answers questions such as <i>does a richer country really live longer</i> and '
                f'<i>did the digital divide close</i>, and then uses machine learning to <b>predict</b> life expectancy, <b>group</b> countries into development tiers and <b>forecast</b> the next years.<br><br>'
                f'The dataset has <b>{n_rows:,} rows</b> because every country-year is stored nine times, once per economic sector. The app works with the <b>{len(cy):,} unique country-years</b>. '
                f'It is a practice dataset, so every finding describes this data and not official statistics.</div>', unsafe_allow_html=True)
            t = st.columns(4)
            for col, (l_, v_) in zip(t, [("Country-years", f"{len(cy):,}"), ("Countries", cy.Country.nunique()), ("Continents", cy.Continent.nunique()), ("Years", f"{y1 - y0 + 1}")]):
                col.markdown(f'<div class="tile"><small>{l_}</small><b>{v_}</b></div>', unsafe_allow_html=True)
    with b:
        with st.container(border=True):
            st.markdown('<div class="kick">The tour</div><div class="h">How to use it</div>', unsafe_allow_html=True)
            for n_, t_ in [("1", "<b>Press Next</b> to move one slide at a time, or open <b>Contents</b> to jump anywhere."),
                           ("2", "Each insight slide has the <b>chart</b> on the left and the <b>story panel</b> on the right: what it shows, how it is made, and the takeaway."),
                           ("3", "Open <b>Filters</b> under the top bar to focus on continents, income groups and years."),
                           ("4", "The last slides use <b>machine learning</b> and end with the conclusions.")]:
                st.markdown(f'<div class="step"><div class="n">{n_}</div><div>{t_}</div></div>', unsafe_allow_html=True)
            st.button("Start the tour \u2192", key="start_btn", on_click=goto, args=(1,), width="stretch")
    st.markdown('<div class="kick" style="margin:14px 0 6px 2px">Snapshot of the current selection</div>', unsafe_allow_html=True)
    allm = cy[["GDP_Per_Capita_USD", "Life_Expectancy_Years", "CO2_Emissions_Per_Capita_Tons", "Happiness_Index", "Human_Development_Index"]].mean()
    cur = f[allm.index].mean()
    ser = f.groupby("Year")[list(allm.index)].mean()
    k = st.columns(5)
    kpi(k[0], "Avg GDP per capita", f"${cur.GDP_Per_Capita_USD:,.0f}", cur.GDP_Per_Capita_USD - allm.GDP_Per_Capita_USD, series=ser.GDP_Per_Capita_USD, color=TEAL)
    kpi(k[1], "Avg life expectancy", f"{cur.Life_Expectancy_Years:.1f} yrs", cur.Life_Expectancy_Years - allm.Life_Expectancy_Years, series=ser.Life_Expectancy_Years, color=BLUE)
    kpi(k[2], "Avg CO2 per person", f"{cur.CO2_Emissions_Per_Capita_Tons:.2f} t", cur.CO2_Emissions_Per_Capita_Tons - allm.CO2_Emissions_Per_Capita_Tons, good_up=False, series=ser.CO2_Emissions_Per_Capita_Tons, color=ROSE)
    kpi(k[3], "Avg happiness", f"{cur.Happiness_Index:.2f}", cur.Happiness_Index - allm.Happiness_Index, series=ser.Happiness_Index, color=GOLD)
    kpi(k[4], "Avg HDI", f"{cur.Human_Development_Index:.3f}", cur.Human_Development_Index - allm.Human_Development_Index, series=ser.Human_Development_Index, color=VIOLET)
    with st.expander("Peek at the data"):
        st.dataframe(f.sample(min(200, len(f)), random_state=1).sort_values(["Country", "Year"]).head(100), width="stretch", height=280)


def s_map():
    mopts = {NICE[k]: k for k in ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons", "Happiness_Index", "Literacy_Rate_Pct"]}
    msel = st.session_state.get("mapm", list(mopts)[0])
    mc = mopts[msel]
    fmap = fyr.copy()
    yfocus = int(fmap.Year.iloc[0]) if len(fmap) else fy
    L, R = slide((2.4, 1))
    with L, st.container(border=True):
        fig = px.choropleth(fmap, locations="ISO3_Code", color=mc, hover_name="Country", custom_data=["Country"],
                            color_continuous_scale="Inferno" if mc == "CO2_Emissions_Per_Capita_Tons" else MAP_SCALE, labels={mc: msel})
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", landcolor="#18223a", showland=True)
        style(fig, 560, False)
        fig.update_layout(coloraxis_colorbar=dict(title="", thickness=12, len=.7), margin=dict(l=0, r=0, t=0, b=0))
        ev = st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG, on_select="rerun", selection_mode="points", key="worldmap")
    sel_c = None
    try:
        pts = ev.selection.points
        if pts:
            sel_c = pts[0].get("customdata", [None])[0] or pts[0].get("hovertext")
    except Exception:
        sel_c = None
    with R:
        panel(1, "The world at a glance", "Every country coloured by the measure you choose. Click a country, or use the picker, to open its profile.",
              [("Chart", "Choropleth map"), ("Library", "Plotly"), ("Use", "Click, hover, change measure")])
        st.selectbox("Measure on the map", list(mopts), key="mapm")
        spot_pick = st.selectbox("Spotlight a country", ["-"] + sorted(fyr.Country.unique()), key="spotpick")
        if not sel_c and spot_pick != "-":
            sel_c = spot_pick
        top, bot = fmap.nlargest(1, mc).iloc[0], fmap.nsmallest(1, mc).iloc[0]
        take(f"In {yfocus}, <b>{top.Country}</b> leads on {msel.lower()} ({top[mc]:,.2f}) and <b>{bot.Country}</b> is lowest ({bot[mc]:,.2f}). Dark to bright shows how unevenly development is spread.")
        if sel_c and sel_c in set(cy.Country):
            cc = cy[cy.Country == sel_c].sort_values("Year")
            rw = cc[cc.Year == yfocus].iloc[0] if (cc.Year == yfocus).any() else cc.iloc[-1]
            rank_ = int((fmap[mc] > rw[mc]).sum()) + 1
            st.markdown(f'<div class="spot" style="margin-top:12px"><h4>\U0001F4CD {sel_c}</h4><span class="pill">{rw.Continent}</span><span class="pill g">{rw.Income_Group}</span><br>'
                        f'<span style="font-size:1.6rem;font-weight:800;color:#fff">{rw[mc]:,.2f}</span> <span style="color:#94A3B8">{msel}</span><br>'
                        f'<span style="color:#C3CEE0;font-size:.88rem">Rank <b>{rank_}</b> of {len(fmap)} in {yfocus}<br>Life expectancy {rw.Life_Expectancy_Years:.1f} yrs \u00b7 HDI {rw.Human_Development_Index:.2f} \u00b7 GDP/capita ${rw.GDP_Per_Capita_USD:,.0f}</span></div>', unsafe_allow_html=True)
            mini = px.line(cc, x="Year", y=mc)
            mini.update_traces(line=dict(color=GOLD, width=3))
            style(mini, 130, False)
            mini.update_layout(margin=dict(l=0, r=0, t=8, b=0), yaxis_title=None, xaxis_title=None)
            st.plotly_chart(mini, width="stretch", config=PLOTLY_CFG)


def s_bubble():
    g = f.dropna(subset=["Population_Total"]).sort_values("Year")
    L, R = slide((2.4, 1))
    with L, st.container(border=True):
        fig = px.scatter(g, x="GDP_Per_Capita_USD", y="Life_Expectancy_Years", size="Population_Total", color="Continent", hover_name="Country", animation_frame="Year",
                         log_x=True, size_max=50, color_discrete_map=CONT_COL, range_y=[35, 90], range_x=[150, 120000],
                         labels={"GDP_Per_Capita_USD": "GDP per capita (USD, log scale)", "Life_Expectancy_Years": "Life expectancy (years)"})
        style(fig, 580)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    r = np.corrcoef(np.log(f.GDP_Per_Capita_USD), f.Life_Expectancy_Years)[0, 1]
    with R:
        panel(2, "Income and life expectancy", "Each bubble is a country, sized by population. Press Play to watch the years go by.",
              [("Chart", "Animated bubble chart"), ("Library", "Plotly"), ("Use", "Play, hover, log scale")])
        take(f"The correlation between log GDP and life expectancy is <b>{r:.2f}</b>. Life expectancy climbs steeply up to about $10,000 per person and then flattens: extra income helps far less once the basics are covered.")


def s_life():
    g = f.groupby(["Year", "Continent"], as_index=False).Life_Expectancy_Years.mean()
    L, R = slide()
    with L, st.container(border=True):
        fig = px.line(g, x="Year", y="Life_Expectancy_Years", color="Continent", color_discrete_map=CONT_COL, labels={"Life_Expectancy_Years": "Life expectancy (years)"})
        fig.update_traces(line=dict(width=3.5))
        style(fig, 540)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    gain = g.pivot(index="Year", columns="Continent", values="Life_Expectancy_Years")
    d = (gain.iloc[-1] - gain.iloc[0]).sort_values(ascending=False)
    with R:
        panel(3, "Life expectancy by continent", "Average years of life for each continent, year by year.", [("Chart", "Line chart"), ("Library", "Plotly"), ("Use", "Hover, click legend to hide")])
        take(f"From {gain.index[0]} to {gain.index[-1]}, <b>{d.index[0]}</b> gained the most (+{d.iloc[0]:.1f} years) and <b>{d.index[-1]}</b> the least (+{d.iloc[-1]:.1f}). Every line dips together near 2020, a sign of one shared shock.")


def s_internet():
    g = f.groupby(["Year", "Continent"], as_index=False).Internet_Penetration_Pct.mean()
    L, R = slide()
    with L, st.container(border=True):
        fig = go.Figure()
        for c_ in sorted(g.Continent.unique()):
            s_ = g[g.Continent == c_]
            fig.add_trace(go.Scatter(x=s_.Year, y=s_.Internet_Penetration_Pct, name=c_, mode="lines", fill="tozeroy", line=dict(color=CONT_COL[c_], width=2.5), opacity=.8))
        fig.update_yaxes(title="Internet users (% of population)")
        style(fig, 540)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    last = g[g.Year == g.Year.max()].set_index("Continent").Internet_Penetration_Pct.sort_values()
    with R:
        panel(4, "The digital divide", "Share of the population online, by continent.", [("Chart", "Area chart"), ("Library", "Plotly"), ("Use", "Hover, click legend")])
        take(f"Internet use starts near zero everywhere. By {int(g.Year.max())} it reaches <b>{last.iloc[-1]:.0f}%</b> in {last.index[-1]} but only <b>{last.iloc[0]:.0f}%</b> in {last.index[0]}. The S-shaped curves show a slow start, a fast middle and a slowing finish.")


def s_hdi():
    g = f.groupby("Income_Group", as_index=False).Human_Development_Index.mean()
    g["Income_Group"] = pd.Categorical(g.Income_Group, INCOME_ORDER, ordered=True)
    g = g.sort_values("Income_Group")
    L, R = slide()
    with L, st.container(border=True):
        fig = px.bar(g, x="Income_Group", y="Human_Development_Index", color="Income_Group", text_auto=".2f", color_discrete_sequence=[ROSE, GOLD, TEAL, BLUE], labels={"Human_Development_Index": "Average HDI"})
        fig.update_traces(textposition="outside", marker_line_width=0)
        style(fig, 540, False)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    hv = g.set_index("Income_Group").Human_Development_Index
    with R:
        panel(5, "Development across income groups", "Average Human Development Index for each income group.", [("Chart", "Bar chart"), ("Library", "Plotly"), ("Use", "Hover for values")])
        hv_txt = ARW.join(f"{v:.2f}" for v in hv.dropna().values)
        take(f"HDI rises step by step with income: {hv_txt}. Income group is a reliable shorthand for development here, though countries inside each group still differ widely.")


def s_sector():
    g = fsec.groupby(["Income_Group", "Economic_Sector"], as_index=False).Sector_GDP_Contribution_Pct.mean()
    g["Income_Group"] = pd.Categorical(g.Income_Group, INCOME_ORDER, ordered=True)
    L, R = slide()
    with L, st.container(border=True):
        fig = px.bar(g.sort_values("Income_Group"), x="Income_Group", y="Sector_GDP_Contribution_Pct", color="Economic_Sector", barmode="stack",
                     color_discrete_sequence=["#60A5FA", "#2DD4BF", "#F5C451", "#FB7185", "#A78BFA", "#34D399", "#F472B6", "#94A3B8"], labels={"Sector_GDP_Contribution_Pct": "Share of GDP (%)"})
        style(fig, 540)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    sv = g.groupby("Economic_Sector").Sector_GDP_Contribution_Pct.mean().sort_values(ascending=False)
    with R:
        panel(6, "The sector mix of economies", "Where GDP comes from in each income group. The Overall_National total row is left out.", [("Chart", "Stacked bar"), ("Library", "Plotly"), ("Use", "Hover, click legend")])
        take(f"<b>{sv.index[0]}</b> dominates with about {sv.iloc[0]:.0f}% of GDP, then {sv.index[1]} ({sv.iloc[1]:.0f}%) and {sv.index[2]} ({sv.iloc[2]:.0f}%). The mix is similar across groups, a clue that this is a generated dataset.")


def s_drivers():
    feats = [c for c in NICE if c != "Life_Expectancy_Years"]
    cr = f[feats + ["Life_Expectancy_Years"]].corr()["Life_Expectancy_Years"].drop("Life_Expectancy_Years").sort_values()
    L, R = slide()
    with L, st.container(border=True):
        fig = go.Figure(go.Bar(x=cr.values, y=[NICE[i] for i in cr.index], orientation="h", marker_color=[TEAL if v > 0 else ROSE for v in cr.values], text=[f"{v:+.2f}" for v in cr.values], textposition="outside"))
        fig.update_xaxes(range=[-1.05, 1.05], title="Correlation with life expectancy")
        style(fig, 600, False)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    with R:
        panel(7, "Indicators linked to life expectancy", "How closely every other indicator moves with life expectancy (Pearson correlation).", [("Method", "Correlation"), ("Chart", "Horizontal bars"), ("Library", "Plotly")])
        take(f"The strongest positive link is <b>{NICE[cr.index[-1]]}</b> ({cr.iloc[-1]:+.2f}) and the strongest negative is <b>{NICE[cr.index[0]]}</b> ({cr.iloc[0]:+.2f}). Correlation is not causation: these indicators rise together, so the bars show where to look next.")


def s_co2():
    L, R = slide()
    with L, st.container(border=True):
        fig = px.scatter(fyr, x="GDP_Per_Capita_USD", y="CO2_Emissions_Per_Capita_Tons", size="Population_Total", color="Continent", hover_name="Country", log_x=True, size_max=44,
                         color_discrete_map=CONT_COL, labels={"GDP_Per_Capita_USD": "GDP per capita (USD, log scale)", "CO2_Emissions_Per_Capita_Tons": "CO2 per person (tons)"})
        style(fig, 560)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    t3 = fyr.nlargest(3, "CO2_Emissions_Per_Capita_Tons")
    with R:
        panel(8, "Emissions and income", "CO2 per person against GDP per capita in the focus year, sized by population.", [("Chart", "Bubble scatter"), ("Library", "Plotly"), ("Use", "Hover, log scale")])
        if len(t3):
            take(f"Emissions rise with income, but the very highest values belong to a few outliers: <b>{', '.join(t3.Country)}</b> ({t3.CO2_Emissions_Per_Capita_Tons.iloc[0]:.1f} t for the top one). Wealth alone does not decide emissions; the structure of the economy does.")


def s_rank():
    rm = st.session_state.get("rankm", NICE["Human_Development_Index"])
    rmode = st.session_state.get("rankmode", "Top 10")
    race = st.session_state.get("race", False)
    rcol = {v: k for k, v in NICE.items()}[rm]
    cont_of = f.drop_duplicates("Country").set_index("Country").Continent
    L, R = slide()
    if race:
        rows_ = []
        for y_ in range(yr[0], yr[1] + 1):
            d_ = f[f.Year == y_].dropna(subset=[rcol])
            d_ = d_.nsmallest(10, rcol) if rmode == "Bottom 10" else d_.nlargest(10, rcol)
            rows_.append(d_.assign(Rank=np.arange(1, len(d_) + 1))[["Country", "Continent", "Year", rcol, "Rank"]])
        rc = pd.concat(rows_)
        fig = px.bar(rc, x=rcol, y="Rank", orientation="h", color="Continent", text="Country", animation_frame="Year", color_discrete_map=CONT_COL, range_x=[0, float(rc[rcol].max()) * 1.08], labels={rcol: rm})
        fig.update_yaxes(autorange="reversed", showticklabels=False, title=None)
        fig.update_traces(textposition="inside", insidetextanchor="start")
        style(fig, 560)
        rk_last = rc[rc.Year == rc.Year.max()].sort_values("Rank")
        msg = f"Press Play: the list re-sorts every year. In {int(rc.Year.max())} <b>{rk_last.Country.iloc[0]}</b> is number one on {rm.lower()}. Watch who enters and leaves the top ten."
    else:
        if rmode == "Biggest improvement":
            a_ = f[f.Year == yr[0]].set_index("Country")[rcol]
            b_ = f[f.Year == yr[1]].set_index("Country")[rcol]
            rk = (b_ - a_).dropna().sort_values(ascending=False).head(10).sort_values()
            ttl = f"Gain in {rm}, {yr[0]} \u2192 {yr[1]}"
        else:
            s_ = fyr.set_index("Country")[rcol].dropna()
            rk = (s_.nlargest(10) if rmode == "Top 10" else s_.nsmallest(10)).sort_values(ascending=(rmode == "Top 10"))
            ttl = f"{rmode}: {rm}, {int(fyr.Year.iloc[0]) if len(fyr) else fy}"
        fig = go.Figure(go.Bar(x=rk.values, y=rk.index, orientation="h", marker_color=[CONT_COL.get(cont_of.get(c, ""), SLATE) for c in rk.index], text=[f"{v:,.2f}" for v in rk.values], textposition="outside"))
        fig.update_layout(title=ttl)
        style(fig, 560, False)
        msg = f"<b>{rk.index[-1]}</b> sits at the {'top' if rmode != 'Bottom 10' else 'very bottom'} of this list ({rk.iloc[-1]:,.2f}). Bar colours show the continent, so you can see whether leaders or strugglers cluster in one region."
    with L, st.container(border=True):
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    with R:
        panel(9, "Leaders and biggest improvers", "Top or bottom countries on a measure you pick, or the fastest improvers. Switch on the race to animate it.", [("Chart", "Ranked bars, bar race"), ("Library", "Plotly"), ("Use", "Pick measure, Play")])
        st.selectbox("Measure", [NICE[k] for k in ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Happiness_Index", "Internet_Penetration_Pct", "Literacy_Rate_Pct"]], key="rankm")
        st.radio("Show", ["Top 10", "Bottom 10", "Biggest improvement"], key="rankmode", horizontal=True)
        st.toggle("\U0001F3C1 Race through the years", key="race")
        take(msg)


def s_tree():
    tm = fyr.dropna(subset=["Population_Total", "GDP_Per_Capita_USD"])
    L, R = slide((2.4, 1))
    with L, st.container(border=True):
        fig = px.treemap(tm, path=[px.Constant("World"), "Continent", "Income_Group", "Country"], values="Population_Total", color="GDP_Per_Capita_USD", color_continuous_scale="Viridis", labels={"GDP_Per_Capita_USD": "GDP per capita"})
        style(fig, 580, False)
        fig.update_traces(marker=dict(cornerradius=5))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    pop = tm.groupby("Continent").Population_Total.sum().sort_values(ascending=False)
    with R:
        panel(10, "Population and wealth", "Box size is population and colour is GDP per capita. Click a box to drill down from continent to country.", [("Chart", "Treemap"), ("Library", "Plotly"), ("Use", "Click to drill down")])
        if len(pop):
            take(f"<b>{pop.index[0]}</b> holds the largest share of people ({pop.iloc[0] / pop.sum():.0%} of the selection). Big, dark boxes are populous countries with low income, which is where development gains matter most.")


def s_explorer():
    all_c = sorted(cy.Country.unique())
    dflt = [c for c in ["India", "United States", "Brazil", "Nigeria"] if c in all_c]
    L, R = slide((2, 1))
    with R:
        st.markdown('<div class="kick">Compare</div><div class="h">Country Explorer</div><div class="d">Pick up to four countries. The radar scales every indicator from 0 (lowest in the world) to 1 (highest) and compares them with the world average.</div>'
                    + pills([("Chart", "Radar and line"), ("Library", "Plotly"), ("Use", "Pick countries and trend")]), unsafe_allow_html=True)
        pick = st.multiselect("Countries to compare", all_c, default=dflt, max_selections=4, key="explorer")
        em = st.selectbox("Trend of", [NICE[k] for k in NICE], index=0, key="expm")
    if not pick:
        with L:
            st.info("Choose at least one country on the right.")
        return
    lastY = int(cy.Year.max())
    now = cy[cy.Year == lastY].copy()
    now["GDP_Per_Capita_USD"] = np.log(now["GDP_Per_Capita_USD"])
    rad = ["Life_Expectancy_Years", "Literacy_Rate_Pct", "Internet_Penetration_Pct", "Human_Development_Index", "Happiness_Index", "GDP_Per_Capita_USD", "Urban_Population_Pct", "Health_Expenditure_Pct_GDP"]
    mn, mx = now[rad].min(), now[rad].max()
    norm = (now.set_index("Country")[rad] - mn) / (mx - mn)
    world = norm.mean()
    labels = [NICE[c].replace("GDP per capita (USD)", "GDP per capita (log)") for c in rad]
    pal = [GOLD, TEAL, BLUE, ROSE]
    with L, st.container(border=True):
        fig = go.Figure()
        for i, c_ in enumerate(pick):
            v_ = norm.loc[c_].values
            fig.add_trace(go.Scatterpolar(r=list(v_) + [v_[0]], theta=labels + [labels[0]], name=c_, fill="toself", opacity=.5, line=dict(color=pal[i], width=3)))
        fig.add_trace(go.Scatterpolar(r=list(world.values) + [world.values[0]], theta=labels + [labels[0]], name="World average", line=dict(color="#cbd5e1", dash="dash", width=2)))
        fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="rgba(148,163,184,.2)", tickfont=dict(size=9)), angularaxis=dict(gridcolor="rgba(148,163,184,.2)")))
        style(fig, 470)
        fig.update_layout(margin=dict(l=70, r=70, t=20, b=50))
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    ecol = {v: k for k, v in NICE.items()}[em]
    with R, st.container(border=True):
        tr = cy[cy.Country.isin(pick)][["Country", "Year", ecol]]
        wd = cy.groupby("Year", as_index=False)[ecol].mean().assign(Country="World average")
        fig = px.line(pd.concat([tr, wd]), x="Year", y=ecol, color="Country", labels={ecol: em}, color_discrete_map={**{c_: pal[i] for i, c_ in enumerate(pick)}, "World average": "#cbd5e1"})
        fig.update_traces(line=dict(width=3))
        for trc in fig.data:
            if trc.name == "World average":
                trc.line.dash = "dash"
        style(fig, 300)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    with st.container(border=True):
        st.markdown('<div class="kick">Scorecard</div><div class="d" style="margin-bottom:6px">Value (and world rank, 1 = best of all countries; for CO2 lower is better).</div>', unsafe_allow_html=True)
        sc_cols = ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Happiness_Index", "Literacy_Rate_Pct", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons"]
        raw_ = cy[cy.Year == lastY].set_index("Country")
        tb = []
        for c_ in pick:
            r_ = {"Country": c_}
            for col_ in sc_cols:
                rk_ = raw_[col_].rank(ascending=(col_ == "CO2_Emissions_Per_Capita_Tons"), method="min")[c_]
                r_[NICE[col_]] = f"{raw_.loc[c_, col_]:,.2f}  (#{int(rk_)})"
            tb.append(r_)
        st.dataframe(pd.DataFrame(tb).set_index("Country"), width="stretch")
        diff = (norm.loc[pick[0]] - world).sort_values()
        take(f"<b>{pick[0]}</b> is strongest relative to the world on <b>{labels[rad.index(diff.index[-1])]}</b> and <b>{labels[rad.index(diff.index[-2])]}</b>, and weakest on <b>{labels[rad.index(diff.index[0])]}</b>. A big gap to the dashed line shows where a country could improve most.")


def s_ml1():
    res, oof, coef, icpt, xm, xs = train_models(cy[["Country"] + FEATS + [TARGET]], "v1")
    best = res.iloc[0]
    r2r = float(res[res.Model == "Ridge regression"]["R-squared"].iloc[0])
    st.markdown('<div class="kick">Machine learning \u00b7 1 of 3</div><div class="h">Predict life expectancy</div>', unsafe_allow_html=True)
    spec([("Model used", "Ridge regression (final)"), ("Task", "Regression: predict life expectancy"), ("Inputs", "6 indicators: GDP, literacy, urban %, internet %, health spend, fertility"),
          ("Tested how", "5-fold, whole countries held out"), ("Compared with", "k-NN, Random Forest, Gradient Boosting"), ("Result", f"R\u00b2 {r2r:.2f}, typical error \u00b1{best['MAE (years)']:.1f} years")])
    A, B = st.columns([1, 1.15], gap="large")
    with A, st.container(border=True):
        st.markdown('<div class="kick">Model comparison</div>', unsafe_allow_html=True)
        st.dataframe(res.style.format({"R-squared": "{:.3f}", "MAE (years)": "{:.2f}"}).background_gradient(subset=["R-squared"], cmap="Greens"), width="stretch", hide_index=True)
        fig = px.scatter(x=cy[TARGET], y=oof, opacity=.35, labels={"x": "Actual life expectancy", "y": "Predicted (Ridge)"})
        fig.add_shape(type="line", x0=35, y0=35, x1=90, y1=90, line=dict(color=GOLD, dash="dash"))
        fig.update_traces(marker=dict(color=BLUE, size=5))
        style(fig, 300, False)
        fig.update_layout(title=f"Predicted vs actual on unseen countries (R\u00b2 = {r2r:.2f})")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        take("Ridge is as accurate as the complex models on this data, trains instantly and every coefficient is readable, so it is our final model.")
    with B, st.container(border=True):
        st.markdown('<div class="kick">What-if simulator</div>', unsafe_allow_html=True)
        cl = sorted(cy.Country.unique())
        cty = st.selectbox("Start from a country (latest year)", cl, index=cl.index("India") if "India" in cl else 0)
        row = cy[(cy.Country == cty) & (cy.Year == cy.Year.max())].iloc[0]
        vals = {}
        sl = st.columns(2)
        ranges = {"Literacy rate (%)": (0, 100), "Urban population (%)": (0, 100), "Internet users (%)": (0, 100), "Health spending (% GDP)": (0, 20), "Fertility rate": (1.0, 8.0)}
        for i, n in enumerate(FNAMES):
            with sl[i % 2]:
                if n == "ln(GDP per capita)":
                    v = st.slider("GDP per capita (USD)", 200, 120000, int(row.GDP_Per_Capita_USD), step=100, key=f"w{cty}{n}")
                    vals[n] = float(np.log(v))
                else:
                    lo, hi = ranges[n]
                    vals[n] = st.slider(n, float(lo), float(hi), float(np.clip(row[FEATS[i]], lo, hi)), key=f"w{cty}{n}")
        pred = icpt + sum(coef[n] * vals[n] for n in FNAMES)
        actual = float(row[TARGET])
        m = st.columns(3)
        kpi(m[0], "Predicted", f"{pred:.1f} yrs", color=TEAL)
        kpi(m[1], f"{cty} actual ({int(row.Year)})", f"{actual:.1f} yrs", pred - actual, color=BLUE)
        kpi(m[2], "Typical error", f"\u00b1{best['MAE (years)']:.1f} yrs", color=GOLD)
        contrib = pd.Series({n: coef[n] * (vals[n] - xm[n]) for n in FNAMES}).sort_values()
        fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=[TEAL if v > 0 else ROSE for v in contrib.values], text=[f"{v:+.1f}" for v in contrib.values], textposition="outside"))
        fig.update_layout(title="Why this prediction? (years added or removed vs the average country)")
        style(fig, 290, False)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        strongest = contrib.abs().idxmax()
        take(f"The biggest driver here is <b>{strongest}</b> ({contrib[strongest]:+.1f} years). Move the sliders to see what each factor is worth. This is a pattern in the data, not proof that changing one factor would cause the change.")


def s_ml2():
    k = int(st.session_state.get("kclusters", 4))
    last, prof, sil, ev = build_clusters(cy, k, "v1")
    st.markdown('<div class="kick">Machine learning \u00b7 2 of 3</div><div class="h">Group countries into tiers</div>', unsafe_allow_html=True)
    spec([("Model used", "K-Means clustering + PCA"), ("Task", "Unsupervised grouping into development tiers"), ("Inputs", "8 indicators per country, 2022\u201324 average, standardised"),
          ("Output", f"{k} tiers named by average HDI"), ("Quality check", f"Silhouette score {sil:.2f}"), ("Map", f"2-D view keeps {ev:.0%} of the information")])
    A, B = st.columns([1.5, 1], gap="large")
    with A, st.container(border=True):
        fig = px.scatter(last.sort_values("Cluster"), x="PC1", y="PC2", color="Cluster", hover_name="Country", hover_data={"PC1": False, "PC2": False, "Continent": True},
                         color_discrete_sequence=[ROSE, GOLD, TEAL, BLUE, VIOLET, SLATE], labels={"PC1": "Overall development \u2192", "PC2": "Second pattern"})
        fig.update_traces(marker=dict(size=11, line=dict(width=.6, color="#070B14")))
        style(fig, 500)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    with B, st.container(border=True):
        st.slider("Number of tiers", 3, 6, 4, key="kclusters")
        show = prof.rename(columns=lambda c: NICE.get(c, c))
        fmt = {c_: "{:,.2f}" for c_ in show.columns}
        fmt["Countries"] = "{:,.0f}"
        st.dataframe(show.style.format(fmt).background_gradient(cmap="Blues", axis=0), width="stretch", height=250)
        cl_sel = st.selectbox("Countries in a tier", sorted(last.Cluster.unique()))
        st.caption(", ".join(sorted(last[last.Cluster == cl_sel].Country)))
        top_c, low_c = prof.index[-1], prof.index[0]
        take(f"Nobody told the algorithm about continents or income, yet it finds {k} tiers. <b>{low_c}</b> countries ({int(prof.loc[low_c, 'Countries'])}) average ${prof.loc[low_c, 'GDP_Per_Capita_USD']:,.0f} and {prof.loc[low_c, 'Life_Expectancy_Years']:.0f} years of life, against ${prof.loc[top_c, 'GDP_Per_Capita_USD']:,.0f} and {prof.loc[top_c, 'Life_Expectancy_Years']:.0f} for <b>{top_c}</b>.")


def s_ml3():
    fm = st.session_state.get("fm", "Life_Expectancy_Years")
    hz = int(st.session_state.get("hz", 2030))
    shown = st.session_state.get("fshown", sorted(cy.Continent.unique()))
    fc, bt = forecast_table(cy, fm, hz, "v1")
    st.markdown('<div class="kick">Machine learning \u00b7 3 of 3</div><div class="h">Forecast to 2030</div>', unsafe_allow_html=True)
    spec([("Model used", "Linear trend (ordinary least squares)"), ("Task", "Time-series forecast per continent"), ("Inputs", "Yearly continent averages, last 15 years"),
          ("Output", "Forecast with an 80% prediction band"), ("Back-test", "Trained to 2018, tested on 2019\u201324"), ("Beat 'no change'", f"{int((bt['Trend-model error'] < bt[chr(39) + 'Same as last year' + chr(39) + ' error']).sum())} of {len(bt)} continents")])
    A, B = st.columns([2.2, 1], gap="large")
    hist = cy.groupby(["Continent", "Year"], as_index=False)[fm].mean()
    with A, st.container(border=True):
        fig = go.Figure()
        for c_ in shown:
            col = CONT_COL[c_]
            h_, f_ = hist[hist.Continent == c_], fc[fc.Continent == c_]
            rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
            fig.add_trace(go.Scatter(x=h_.Year, y=h_[fm], name=c_, line=dict(color=col, width=3.5)))
            fig.add_trace(go.Scatter(x=list(f_.Year) + list(f_.Year[::-1]), y=list(f_.Upper) + list(f_.Lower[::-1]), mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[h_.Year.iloc[-1]] + list(f_.Year), y=[h_[fm].iloc[-1]] + list(f_.Value), mode="lines", line=dict(color=col, dash="dash", width=3.5), showlegend=False, name=f"{c_} forecast"))
        fig.add_vline(x=int(cy.Year.max()) + 0.5, line_dash="dot", line_color="#667", annotation_text="forecast \u2192")
        fig.update_yaxes(title=NICE[fm])
        style(fig, 470)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
    with B, st.container(border=True):
        st.selectbox("Measure to forecast", ["Life_Expectancy_Years", "GDP_Per_Capita_USD", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons"], format_func=lambda k_: NICE[k_], key="fm")
        st.slider("Forecast until", 2026, 2035, 2030, key="hz")
        st.multiselect("Continents", sorted(cy.Continent.unique()), default=sorted(cy.Continent.unique()), key="fshown")
        last_y = fc[fc.Year == hz].set_index("Continent").Value
        now = hist[hist.Year == hist.Year.max()].set_index("Continent")[fm]
        chg = (last_y - now).sort_values(ascending=False)
        take(f"If recent trends continue, <b>{chg.index[0]}</b> changes the most by {hz} ({chg.iloc[0]:+,.1f}) and <b>{chg.index[-1]}</b> the least ({chg.iloc[-1]:+,.1f}). It is a straight-line extrapolation, so a shock like the 2020 dip would not be foreseen.")
    C, D = st.columns(2, gap="large")
    tbl = pd.DataFrame({f"{int(cy.Year.max())}": now, f"{hz} forecast": last_y, "Change": last_y - now}).loc[chg.index]
    with C, st.container(border=True):
        st.markdown('<div class="kick">Forecast summary</div>', unsafe_allow_html=True)
        st.dataframe(tbl.style.format("{:,.1f}"), width="stretch")
    with D, st.container(border=True):
        st.markdown('<div class="kick">Back-test error (trained to 2018, tested 2019\u201324)</div>', unsafe_allow_html=True)
        st.dataframe(bt.set_index("Continent").style.format("{:,.2f}"), width="stretch")


def s_concl():
    res, *_ = train_models(cy[["Country"] + FEATS + [TARGET]], "v1")
    lf = cy.groupby(["Year", "Continent"]).Life_Expectancy_Years.mean().unstack()
    nt = cy.groupby(["Year", "Continent"]).Internet_Penetration_Pct.mean().unstack()
    hd = cy.groupby("Income_Group").Human_Development_Index.mean().reindex(INCOME_ORDER)
    r_ = np.corrcoef(np.log(cy.GDP_Per_Capita_USD), cy.Life_Expectancy_Years)[0, 1]
    r2 = float(res[res.Model == "Ridge regression"]["R-squared"].iloc[0])
    mae = float(res[res.Model == "Ridge regression"]["MAE (years)"].iloc[0])
    st.markdown('<div class="kick">The story</div><div class="h">What Insight Atlas tells us</div>', unsafe_allow_html=True)
    A, B, C = st.columns(3, gap="large")
    with A, st.container(border=True):
        st.markdown('<div class="kick">From the 10 graphs</div>', unsafe_allow_html=True)
        st.markdown(f"- **Life expectancy rose everywhere**, from {lf.iloc[0].mean():.1f} to {lf.iloc[-1].mean():.1f} years. Africa gained most ({lf.Africa.iloc[-1] - lf.Africa.iloc[0]:+.1f}) but is still about {lf.Europe.iloc[-1] - lf.Africa.iloc[-1]:.0f} years behind Europe.\n"
                    f"- **Income helps, with diminishing returns**: log GDP and life expectancy correlate at {r_:.2f}.\n"
                    f"- **The digital divide narrowed, not closed**: Africa {nt.Africa.iloc[-1]:.0f}% online against Europe {nt.Europe.iloc[-1]:.0f}%.\n"
                    f"- **Development is a staircase**: HDI {hd.iloc[0]:.2f} \u2192 {hd.iloc[1]:.2f} \u2192 {hd.iloc[2]:.2f} \u2192 {hd.iloc[3]:.2f} from low to high income.\n"
                    f"- **Emissions depend on structure**: a few high-income oil economies lead CO2 per person.")
    with B, st.container(border=True):
        st.markdown('<div class="kick">From machine learning</div>', unsafe_allow_html=True)
        st.markdown(f"- **Predict**: Ridge regression estimates life expectancy for unseen countries with R\u00b2 \u2248 {r2:.2f} (about \u00b1{mae:.1f} years). Complex models did not beat it.\n"
                    "- **Group**: K-Means finds clear development tiers without being told continents or income.\n"
                    "- **Forecast**: every continent keeps improving to 2030, but the gaps shrink only slowly.")
    with C, st.container(border=True):
        st.markdown('<div class="kick">Limits to remember</div>', unsafe_allow_html=True)
        st.markdown("- This is a **practice dataset**, so results describe the data, not the real world.\n- Averages treat a small country like a large one.\n- Correlation is not causation, and the forecast is a straight-line extrapolation.")
    st.success("Take-away: development moves together, with income, literacy, health and technology rising as a bundle, and the gaps between regions are closing, but slowly.")


SLIDES = [s_welcome, s_map, s_bubble, s_life, s_internet, s_hdi, s_sector, s_drivers, s_co2, s_rank, s_tree, s_explorer, s_ml1, s_ml2, s_ml3, s_concl]

if f.empty and 1 <= step <= 10:
    st.warning("No data matches these filters. Open Filters above and widen them, or press Reset.")
else:
    SLIDES[step]()

# ---- bottom navigation
st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
b1, b2, b3 = st.columns([1.2, 3, 1.2], vertical_alignment="center")
b1.button(f"\u2190 {STEPS[step - 1][0]}" if step > 0 else "\u2190 Back", key="back_bot", on_click=goto, args=(step - 1,), disabled=step == 0, width="stretch")
b3.button(f"{STEPS[step + 1][0]} \u2192" if step < N - 1 else "The end", key="next_bot", on_click=goto, args=(step + 1,), disabled=step == N - 1, width="stretch")
st.markdown('<div class="foot">Insight Atlas \u00b7 Data Visualization Lab (BISL504-C) \u00b7 The National Institute of Engineering, Mysuru \u00b7 Built with Python, Plotly, scikit-learn and Streamlit</div>', unsafe_allow_html=True)
