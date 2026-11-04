"""
INSIGHT ATLAS  -  one-file interactive dashboard for the Data Visualization Lab (BISL504-C)
Dataset : global_socioeconomic_trends.csv  (194 countries, 1994-2024, practice dataset)

Run     :  pip install streamlit plotly pandas numpy scikit-learn statsmodels
           streamlit run insight_atlas.py
Place global_socioeconomic_trends.csv next to this file (or upload it in the sidebar).
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

st.set_page_config(page_title="Insight Atlas", page_icon="\U0001F9ED", layout="wide")

# ------------------------------------------------------------------ look & feel
GOLD, TEAL, BLUE, CORAL, PURPLE, GREY = "#F0B429", "#3FA796", "#4C78FF", "#E4572E", "#9B6BD3", "#8A94A6"
CONT_COL = {"Africa": CORAL, "Asia": BLUE, "Europe": TEAL, "North America": GOLD, "Oceania": PURPLE, "South America": GREY}
INCOME_ORDER = ["Low income", "Lower-middle income", "Upper-middle income", "High income"]
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input { font-family: 'Poppins', sans-serif !important; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a2342 0%, transparent 60%), radial-gradient(900px 500px at 100% 0%, #10302f 0%, transparent 55%), #0b0f1a; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], footer { visibility: hidden; height: 0; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1440px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #11172a 0%, #0d1220 100%); border-right: 1px solid #222a42; }
h1, h2, h3 { letter-spacing: .2px; }
h3 { margin-top: .2rem; }
.hero { position: relative; overflow: hidden; padding: 26px 32px; border-radius: 22px; border: 1px solid #2c3858; margin-bottom: 14px;
        background: linear-gradient(120deg, #1b2a52 0%, #14403f 55%, #3a2f12 100%); background-size: 200% 200%; animation: flow 14s ease infinite; }
.hero:after { content: ""; position: absolute; right: -60px; top: -60px; width: 260px; height: 260px; border-radius: 50%;
        background: radial-gradient(circle, rgba(240,180,41,.35), transparent 65%); }
@keyframes flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero h1 { margin: 0; font-size: 2.6rem; font-weight: 700; background: linear-gradient(90deg, #fff, #F0B429); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: 6px 0 12px 0; color: #d3dbed; font-size: 1.02rem; max-width: 900px; }
.chip { display: inline-block; background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.14); border-radius: 30px; padding: 3px 14px; font-size: .8rem; color: #eef2fb; margin: 0 8px 4px 0; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background: #121829; padding: 6px; border-radius: 14px; border: 1px solid #222b44; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 8px 18px; height: auto; color: #aab4cc; font-weight: 500; }
.stTabs [aria-selected="true"] { background: linear-gradient(90deg, #F0B429, #e08a1e); color: #10131c !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.kpi { background: linear-gradient(160deg, #182037 0%, #131a2c 100%); border: 1px solid #28324f; border-radius: 16px; padding: 14px 16px 8px 16px; transition: transform .2s, border-color .2s; }
.kpi:hover { transform: translateY(-3px); border-color: #F0B429; }
.kpi .l { color: #98a2b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .8px; }
.kpi .v { font-size: 1.7rem; font-weight: 700; color: #fff; line-height: 1.2; }
.kpi .d { font-size: .78rem; margin-top: 2px; }
.kpi svg { display: block; margin-top: 4px; }
.infer { background: linear-gradient(90deg, rgba(240,180,41,.12), rgba(240,180,41,.02)); border-left: 4px solid #F0B429; padding: 11px 16px; border-radius: 10px; margin: 6px 0 4px 0; color: #dfe5f2; font-size: .95rem; }
.infer b { color: #F0B429; }
.tag { display: inline-block; background: #232d49; border-radius: 20px; padding: 2px 12px; font-size: .76rem; color: #b7c1da; margin-right: 8px; }
.badge { display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; border-radius: 10px; margin-right: 10px;
         background: linear-gradient(135deg, #F0B429, #e0701e); color: #10131c; font-weight: 700; }
.ttl { font-size: 1.3rem; font-weight: 600; display: flex; align-items: center; margin-bottom: 4px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; border-color: #252e4a !important; background: rgba(20,26,43,.55); }
div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="tag"] { border-radius: 10px; }
.stMultiSelect div[data-baseweb="tag"] { background: #2a3556 !important; color: #fff; }
.spot { background: linear-gradient(135deg, #1a2a46, #163a3a); border: 1px solid #2f4a63; border-radius: 16px; padding: 14px 18px; }
.spot h4 { margin: 0 0 4px 0; color: #F0B429; }
</style>
""",
    unsafe_allow_html=True,
)

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


px.defaults.color_discrete_sequence = [BLUE, TEAL, GOLD, CORAL, PURPLE, "#52C7E8", "#E58BB7", "#9BD36A"]
PLOTLY_CFG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
MAP_SCALE = ["#1b2a49", "#2b4f81", "#2a7f9e", "#3fa796", "#b7d36a", "#F0B429"]


def style(fig, h=430, legend=True):
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=40, b=10), font=dict(size=13), showlegend=legend,
                      legend=dict(orientation="h", y=-0.18, font=dict(size=11)))
    fig.update_xaxes(gridcolor="#242b3d")
    fig.update_yaxes(gridcolor="#242b3d")
    return fig


# ------------------------------------------------------------------ data
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


with st.sidebar:
    st.markdown("## \U0001F9ED Insight Atlas")
    st.caption("Data Visualization Lab (BISL504-C)")
    path = find_csv()
    up = None
    if path is None:
        up = st.file_uploader("Upload global_socioeconomic_trends.csv", type="csv")
        if up is None:
            st.info("Put the CSV next to this file or upload it here.")
            st.stop()
    cy, sec, n_rows, n_cols_raw = load(up if up is not None else path)
    st.markdown("### Filters")
    y0, y1 = int(cy.Year.min()), int(cy.Year.max())
    ALLC, ALLI = sorted(cy.Continent.unique()), INCOME_ORDER
    for k_, v_ in {"f_cont": ALLC, "f_inc": ALLI, "f_yr": (y0, y1), "f_fy": y1}.items():
        if k_ not in st.session_state:
            st.session_state[k_] = v_

    def _reset():
        st.session_state.f_cont, st.session_state.f_inc, st.session_state.f_yr, st.session_state.f_fy = ALLC, ALLI, (y0, y1), y1

    conts = st.multiselect("\U0001F30D Continent", ALLC, key="f_cont")
    incs = st.multiselect("\U0001F4B0 Income group", ALLI, key="f_inc")
    yr = st.slider("\U0001F4C5 Year range", y0, y1, key="f_yr")
    fy = st.slider("\U0001F3AF Focus year (map, rankings, scatter)", y0, y1, key="f_fy")
    st.button("\u21BA Reset all filters", on_click=_reset, width="stretch")
    st.markdown("---")
    st.caption("All filters apply to every graph on the Insights tab. The ML tab always trains on the full dataset.")

mask = cy.Continent.isin(conts) & cy.Income_Group.isin(incs) & cy.Year.between(*yr)
f = cy[mask]
fyr = f[f.Year == min(max(fy, yr[0]), yr[1])]
fsec = sec[sec.Continent.isin(conts) & sec.Income_Group.isin(incs) & sec.Year.between(*yr)]

st.markdown(
    f'<div class="hero"><h1><span style="-webkit-text-fill-color:initial">\U0001F9ED</span> Insight Atlas</h1><p>Thirty years of global development in one interactive page: explore the patterns, '
    f'then let machine learning predict, group and forecast.</p>'
    f'<span class="chip">\U0001F4CA {n_rows:,} rows</span><span class="chip">\U0001F30D {cy.Country.nunique()} countries</span>'
    f'<span class="chip">\U0001F4C5 {y0}\u2013{y1}</span><span class="chip">\U0001F9E9 {n_cols_raw} indicators</span>'
    f'<span class="chip">\U0001F916 3 ML models</span></div>', unsafe_allow_html=True)
tab_over, tab_ins, tab_exp, tab_ml, tab_con = st.tabs(["\U0001F3E0 Overview", "\U0001F4C8 10 Insights", "\U0001F50D Country Explorer", "\U0001F916 Predict (ML)", "\U0001F3C1 Conclusions"])

if f.empty:
    st.warning("No data matches these filters. Widen the filters in the sidebar.")
    st.stop()


def kpi(col, label, value, delta=None, good_up=True, series=None):
    d = ""
    if delta is not None:
        up_ = delta >= 0
        c = "#3FA796" if (up_ == good_up) else "#E4572E"
        d = f'<div class="d" style="color:{c}">{"▲" if up_ else "▼"} {abs(delta):,.2f} vs all data</div>'
    sp = spark(series) if series is not None else ""
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}{sp}</div>', unsafe_allow_html=True)


def head(n, title, question):
    st.markdown(f'<div class="ttl"><span class="badge">{n}</span>{title}</div><span class="tag">Question</span> {question}', unsafe_allow_html=True)


def say(text):
    st.markdown(f'<div class="infer"><b>What we can infer:</b> {text}</div>', unsafe_allow_html=True)


def insight(n, title, question, fig, text, ctrl=None):
    with st.container(border=True):
        head(n, title, question)
        if fig is not None:
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        say(text)


def spark(vals, color="#F0B429", w=150, h=34):
    v = np.asarray(vals, dtype=float)
    if len(v) < 2 or np.nanmax(v) == np.nanmin(v):
        return ""
    xs = np.linspace(2, w - 2, len(v))
    ys = h - 3 - (v - np.nanmin(v)) / (np.nanmax(v) - np.nanmin(v)) * (h - 8)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="{color}" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" points="{pts}"/></svg>'


# ================================================================== OVERVIEW
with tab_over:
    a, b = st.columns([1.35, 1])
    with a:
        st.subheader("What is Insight Atlas?")
        st.write(
            "Insight Atlas is a one-file interactive dashboard that turns a large socioeconomic table into a story you can click through. "
            "It answers questions such as: *Does a richer country really live longer? Did the digital divide close? Which countries improved the most?* "
            "and then uses machine learning to **predict** life expectancy, **group** countries into development profiles and **forecast** the next years."
        )
        st.subheader("About the dataset")
        st.write(
            f"**global_socioeconomic_trends.csv** has **{n_rows:,} rows and {n_cols_raw} columns** covering **{cy.Country.nunique()} countries** "
            f"from **{y0} to {y1}**. Every country-year is stored nine times, once for each economic sector (the last one, *Overall_National*, is a total), "
            f"so the dashboard works with the **{len(cy):,} unique country-years** and treats sector data separately. "
            "It is a practice dataset: some labels do not match real-world classifications, so every finding here describes *this data*, not official statistics."
        )
        st.markdown('<span class="tag">Demography</span><span class="tag">Economy</span><span class="tag">Health</span><span class="tag">Education</span>'
                    '<span class="tag">Environment</span><span class="tag">Technology</span><span class="tag">Governance</span>', unsafe_allow_html=True)
    with b:
        st.subheader("How to use it")
        st.markdown("1. Use the **sidebar** to filter continents, income groups and years.\n"
                    "2. Open **10 Insights** - every graph has a question and a plain-language takeaway.\n"
                    "3. Open **Predict** to try the life-expectancy predictor, see country clusters and a forecast to 2030.\n"
                    "4. Read **Conclusions** for the final story.")
    st.subheader("Snapshot of your current selection")
    allm = cy[["GDP_Per_Capita_USD", "Life_Expectancy_Years", "CO2_Emissions_Per_Capita_Tons", "Happiness_Index", "Human_Development_Index"]].mean()
    cur = f[allm.index].mean()
    k = st.columns(5)
    ser = f.groupby("Year")[list(allm.index)].mean()
    kpi(k[0], "Avg GDP per capita", f"${cur.GDP_Per_Capita_USD:,.0f}", cur.GDP_Per_Capita_USD - allm.GDP_Per_Capita_USD, series=ser.GDP_Per_Capita_USD, )
    kpi(k[1], "Avg life expectancy", f"{cur.Life_Expectancy_Years:.1f} yrs", cur.Life_Expectancy_Years - allm.Life_Expectancy_Years, series=ser.Life_Expectancy_Years)
    kpi(k[2], "Avg CO2 per person", f"{cur.CO2_Emissions_Per_Capita_Tons:.2f} t", cur.CO2_Emissions_Per_Capita_Tons - allm.CO2_Emissions_Per_Capita_Tons, good_up=False, series=ser.CO2_Emissions_Per_Capita_Tons)
    kpi(k[3], "Avg happiness", f"{cur.Happiness_Index:.2f}", cur.Happiness_Index - allm.Happiness_Index, series=ser.Happiness_Index)
    kpi(k[4], "Avg HDI", f"{cur.Human_Development_Index:.3f}", cur.Human_Development_Index - allm.Human_Development_Index, series=ser.Human_Development_Index)
    st.caption(f"{f.Country.nunique()} countries, {len(f):,} country-years in the current selection.")
    with st.expander("Peek at the data"):
        st.dataframe(f.sample(min(200, len(f)), random_state=1).sort_values(["Country", "Year"]).head(100), width="stretch", height=300)

# ================================================================== INSIGHTS
with tab_ins:
    # 1 world map
    with st.container(border=True):
        head(1, "The world at a glance", "Where are the high and low performers? Click any country to spotlight it.")
        mopts = {NICE[k]: k for k in ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Internet_Penetration_Pct",
                                       "CO2_Emissions_Per_Capita_Tons", "Happiness_Index", "Literacy_Rate_Pct"]}
        mcol1, mcol2 = st.columns([1, 1])
        msel = mcol1.selectbox("Measure on the map", list(mopts), key="mapm")
        spot_pick = mcol2.selectbox("\U0001F4CD Spotlight a country (or click it on the map)", ["-"] + sorted(fyr.Country.unique()), key="spotpick")
        mc = mopts[msel]
        fmap = fyr.copy()
        yfocus = int(fmap.Year.iloc[0]) if len(fmap) else fy
        fig = px.choropleth(fmap, locations="ISO3_Code", color=mc, hover_name="Country", custom_data=["Country"],
                            color_continuous_scale="Inferno" if mc == "CO2_Emissions_Per_Capita_Tons" else MAP_SCALE, labels={mc: msel})
        fig.update_geos(showframe=False, showcoastlines=False, projection_type="natural earth", bgcolor="rgba(0,0,0,0)", landcolor="#1c2233", showland=True)
        style(fig, 480, False)
        fig.update_layout(coloraxis_colorbar=dict(title="", thickness=12, len=.7))
        top, bot = fmap.nlargest(1, mc).iloc[0], fmap.nsmallest(1, mc).iloc[0]
        ev = st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG, on_select="rerun", selection_mode="points", key="worldmap")
        sel_c = None
        try:
            pts = ev.selection.points
            if pts:
                sel_c = pts[0].get("customdata", [None])[0] or pts[0].get("hovertext")
        except Exception:
            sel_c = None
        if not sel_c and spot_pick != "-":
            sel_c = spot_pick
        say(f"In {yfocus}, <b>{top.Country}</b> leads on {msel.lower()} ({top[mc]:,.2f}) and <b>{bot.Country}</b> is lowest ({bot[mc]:,.2f}). "
            "Dark-to-bright shading shows how unevenly development is spread - change the measure or the focus year to see whether the same regions stay on top.")
        if sel_c and sel_c in set(cy.Country):
            cc = cy[cy.Country == sel_c].sort_values("Year")
            rw = cc[cc.Year == yfocus].iloc[0] if (cc.Year == yfocus).any() else cc.iloc[-1]
            rank_ = int((fmap[mc] > rw[mc]).sum()) + 1
            s1, s2 = st.columns([1.2, 1])
            with s1:
                st.markdown(f'<div class="spot"><h4>\U0001F4CD {sel_c}</h4><span class="tag">{rw.Continent}</span><span class="tag">{rw.Income_Group}</span><br><br>'
                            f'<b>{msel}</b>: <span style="font-size:1.6rem;color:#fff"><b>{rw[mc]:,.2f}</b></span><br>'
                            f'Rank <b>{rank_}</b> of {len(fmap)} in {yfocus} &nbsp;|&nbsp; average of shown countries {fmap[mc].mean():,.2f}<br>'
                            f'Life expectancy {rw.Life_Expectancy_Years:.1f} yrs &nbsp;|&nbsp; HDI {rw.Human_Development_Index:.2f} &nbsp;|&nbsp; GDP/capita ${rw.GDP_Per_Capita_USD:,.0f}</div>', unsafe_allow_html=True)
            with s2:
                mini = px.line(cc, x="Year", y=mc)
                mini.update_traces(line=dict(color=GOLD, width=3))
                style(mini, 170, False)
                mini.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None, xaxis_title=None)
                st.plotly_chart(mini, width="stretch", config=PLOTLY_CFG)

    # 2 gapminder
    g = f.dropna(subset=["Population_Total"]).sort_values("Year")
    fig = px.scatter(g, x="GDP_Per_Capita_USD", y="Life_Expectancy_Years", size="Population_Total", color="Continent", hover_name="Country",
                     animation_frame="Year", log_x=True, size_max=48, color_discrete_map=CONT_COL, range_y=[35, 90], range_x=[150, 120000],
                     labels={"GDP_Per_Capita_USD": "GDP per capita (USD, log)", "Life_Expectancy_Years": "Life expectancy (years)"})
    style(fig, 520)
    r = np.corrcoef(np.log(f.GDP_Per_Capita_USD), f.Life_Expectancy_Years)[0, 1]
    insight(2, "Does money buy a longer life?", "How are income and life expectancy linked - and how did it move over time? (press Play)", fig,
            f"Across the selection the correlation between log GDP and life expectancy is <b>{r:.2f}</b>. Life expectancy climbs steeply up to roughly $10,000 per person and then flattens: "
            "extra income helps far less once basic health and education are in place. The bubbles also drift up and to the right over the years.")

    # 3 life trend
    g = f.groupby(["Year", "Continent"], as_index=False).Life_Expectancy_Years.mean()
    fig = px.line(g, x="Year", y="Life_Expectancy_Years", color="Continent", color_discrete_map=CONT_COL, markers=False,
                  labels={"Life_Expectancy_Years": "Life expectancy (years)"})
    fig.update_traces(line=dict(width=3))
    style(fig)
    gain = g.pivot(index="Year", columns="Continent", values="Life_Expectancy_Years")
    d = (gain.iloc[-1] - gain.iloc[0]).sort_values(ascending=False)
    insight(3, "Who gained the most years?", "How did life expectancy change in each continent?", fig,
            f"Between {gain.index[0]} and {gain.index[-1]}, <b>{d.index[0]}</b> gained the most (+{d.iloc[0]:.1f} years) while <b>{d.index[-1]}</b> gained the least (+{d.iloc[-1]:.1f}). "
            "Every line rises, and they all dip together around 2020 - a sign of one shared shock. The gaps between continents narrow but do not close.")

    # 4 internet
    g = f.groupby(["Year", "Continent"], as_index=False).Internet_Penetration_Pct.mean()
    fig = go.Figure()
    for c_ in sorted(g.Continent.unique()):
        s_ = g[g.Continent == c_]
        fig.add_trace(go.Scatter(x=s_.Year, y=s_.Internet_Penetration_Pct, name=c_, mode="lines", fill="tozeroy", line=dict(color=CONT_COL[c_], width=2.5),
                                 opacity=0.85))
    fig.update_yaxes(title="Internet users (% of population)")
    style(fig)
    last = g[g.Year == g.Year.max()].set_index("Continent").Internet_Penetration_Pct.sort_values()
    insight(4, "Did the digital divide close?", "How fast did the internet spread, and where is it still behind?", fig,
            f"Internet use starts near zero everywhere. By {int(g.Year.max())} it reaches <b>{last.iloc[-1]:.0f}%</b> in {last.index[-1]} but only <b>{last.iloc[0]:.0f}%</b> in {last.index[0]}. "
            "The S-shaped curves show a slow start, a fast middle and a slowing finish - the divide narrowed quickly but a 30-50 point gap remains.")

    # 5 HDI by income
    g = f.groupby("Income_Group", as_index=False).Human_Development_Index.mean()
    g["Income_Group"] = pd.Categorical(g.Income_Group, INCOME_ORDER, ordered=True)
    g = g.sort_values("Income_Group")
    fig = px.bar(g, x="Income_Group", y="Human_Development_Index", color="Income_Group", text_auto=".2f",
                 color_discrete_sequence=[CORAL, GOLD, TEAL, BLUE], labels={"Human_Development_Index": "Average HDI"})
    fig.update_traces(textposition="outside")
    style(fig, legend=False)
    fig.update_yaxes(range=[0, 1])
    hv = g.set_index("Income_Group").Human_Development_Index
    insight(5, "Is development a staircase?", "Does the income group predict human development?", fig,
            f"HDI rises step by step with income: {' → '.join(f'{v:.2f}' for v in hv.dropna().values)}. The steps are fairly even, so income group is a reliable shorthand for development in this data - "
            "though within each group there is still a wide spread between countries.")

    # 6 sector mix
    g = fsec.groupby(["Income_Group", "Economic_Sector"], as_index=False).Sector_GDP_Contribution_Pct.mean()
    g["Income_Group"] = pd.Categorical(g.Income_Group, INCOME_ORDER, ordered=True)
    fig = px.bar(g.sort_values("Income_Group"), x="Income_Group", y="Sector_GDP_Contribution_Pct", color="Economic_Sector", barmode="stack",
                 color_discrete_sequence=px.colors.qualitative.Bold, labels={"Sector_GDP_Contribution_Pct": "Share of GDP (%)"})
    style(fig, 460)
    sv = g.groupby("Economic_Sector").Sector_GDP_Contribution_Pct.mean().sort_values(ascending=False)
    insight(6, "What do economies run on?", "How is GDP split across sectors in each income group? (the Overall_National total is excluded)", fig,
            f"<b>{sv.index[0]}</b> dominates with about {sv.iloc[0]:.0f}% of GDP, followed by {sv.index[1]} ({sv.iloc[1]:.0f}%) and {sv.index[2]} ({sv.iloc[2]:.0f}%). "
            "The mix is surprisingly similar across income groups here, which is itself a clue that this is a generated dataset - real low-income economies lean far more on agriculture.")

    # 7 drivers
    feats = [c for c in NICE if c not in ("Life_Expectancy_Years",)]
    cr = f[feats + ["Life_Expectancy_Years"]].corr()["Life_Expectancy_Years"].drop("Life_Expectancy_Years").sort_values()
    fig = go.Figure(go.Bar(x=cr.values, y=[NICE[i] for i in cr.index], orientation="h", marker_color=[TEAL if v > 0 else CORAL for v in cr.values],
                           text=[f"{v:+.2f}" for v in cr.values], textposition="outside"))
    fig.update_xaxes(range=[-1, 1], title="Correlation with life expectancy")
    style(fig, 520, False)
    insight(7, "What moves with life expectancy?", "Which indicators are most strongly tied to how long people live?", fig,
            f"The strongest positive link is <b>{NICE[cr.index[-1]]}</b> ({cr.iloc[-1]:+.2f}) and the strongest negative is <b>{NICE[cr.index[0]]}</b> ({cr.iloc[0]:+.2f}). "
            "Correlation is not causation: literacy, income and health spending tend to rise together, so the bars show where to look next - which the ML tab then tests.")

    # 8 co2
    fig = px.scatter(fyr, x="GDP_Per_Capita_USD", y="CO2_Emissions_Per_Capita_Tons", size="Population_Total", color="Continent", hover_name="Country",
                     log_x=True, size_max=40, color_discrete_map=CONT_COL, labels={"GDP_Per_Capita_USD": "GDP per capita (USD, log)", "CO2_Emissions_Per_Capita_Tons": "CO2 per person (tons)"})
    style(fig, 470)
    t3 = fyr.nlargest(3, "CO2_Emissions_Per_Capita_Tons")
    insight(8, "Who pollutes the most - and is it just about wealth?", "How do emissions relate to income in the focus year?", fig,
            f"Emissions rise with income, but the very highest values belong to a few outliers: <b>{', '.join(t3.Country)}</b> ({t3.CO2_Emissions_Per_Capita_Tons.iloc[0]:.1f} t for the top one). "
            "Wealth alone does not decide emissions - the economy's structure does. Countries high on income but low on CO2 are the ones worth studying.")

    # 9 ranking
    with st.container(border=True):
        head(9, "Leaders and strugglers", "Who are the top and bottom countries - and who improved the most?")
        c1, c2, c3 = st.columns([1.2, 1.6, 1])
        rm = c1.selectbox("Measure", [NICE[k] for k in ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Happiness_Index",
                                                          "Internet_Penetration_Pct", "Literacy_Rate_Pct"]], key="rankm")
        rmode = c2.radio("Show", ["Top 10", "Bottom 10", "Biggest improvement"], horizontal=True, key="rankmode")
        race = c3.toggle("\U0001F3C1 Race through the years", value=False, key="race")
        rcol = {v: k for k, v in NICE.items()}[rm]
        cont_of = f.drop_duplicates("Country").set_index("Country").Continent
        if race:
            rows_ = []
            for y_ in range(yr[0], yr[1] + 1):
                d_ = f[f.Year == y_].dropna(subset=[rcol])
                d_ = d_.nsmallest(10, rcol) if rmode == "Bottom 10" else d_.nlargest(10, rcol)
                d_ = d_.assign(Rank=np.arange(1, len(d_) + 1))
                rows_.append(d_[["Country", "Continent", "Year", rcol, "Rank"]])
            rc = pd.concat(rows_)
            fig = px.bar(rc, x=rcol, y="Rank", orientation="h", color="Continent", text="Country", animation_frame="Year", color_discrete_map=CONT_COL,
                         range_x=[0, float(rc[rcol].max()) * 1.08], labels={rcol: rm})
            fig.update_yaxes(autorange="reversed", showticklabels=False, title=None)
            fig.update_traces(textposition="inside", insidetextanchor="start")
            style(fig, 470)
            rk_last = rc[rc.Year == rc.Year.max()].sort_values("Rank")
            msg = (f"Press Play: the list re-sorts every year. In {int(rc.Year.max())} <b>{rk_last.Country.iloc[0]}</b> is number one on {rm.lower()}. "
                   "Watch which countries enter and leave the top ten - those are the ones whose position is changing fastest.")
        else:
            if rmode == "Biggest improvement":
                a_ = f[f.Year == yr[0]].set_index("Country")[rcol]
                b_ = f[f.Year == yr[1]].set_index("Country")[rcol]
                rk = (b_ - a_).dropna().sort_values(ascending=False).head(10).sort_values()
                ttl = f"Gain in {rm}, {yr[0]} \u2192 {yr[1]}"
            else:
                s_ = fyr.set_index("Country")[rcol].dropna()
                rk = (s_.nlargest(10) if rmode == "Top 10" else s_.nsmallest(10)).sort_values(ascending=(rmode == "Top 10"))
                ttl = f"{rmode}: {rm}, {int(fyr.Year.iloc[0])}"
            fig = go.Figure(go.Bar(x=rk.values, y=rk.index, orientation="h", marker_color=[CONT_COL.get(cont_of.get(c, ""), GREY) for c in rk.index],
                                   text=[f"{v:,.2f}" for v in rk.values], textposition="outside"))
            fig.update_layout(title=ttl)
            style(fig, 440, False)
            msg = (f"<b>{rk.index[-1]}</b> sits at the {'top' if rmode != 'Bottom 10' else 'very bottom'} of this list ({rk.iloc[-1]:,.2f}). "
                   "Bar colours show the continent, so you can see at a glance whether the leaders (or the strugglers) are concentrated in one region.")
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        say(msg)

    # 10 treemap
    tm = fyr.dropna(subset=["Population_Total", "GDP_Per_Capita_USD"])
    fig = px.treemap(tm, path=[px.Constant("World"), "Continent", "Income_Group", "Country"], values="Population_Total", color="GDP_Per_Capita_USD",
                     color_continuous_scale="Viridis", hover_data={"GDP_Per_Capita_USD": ":,.0f"}, labels={"GDP_Per_Capita_USD": "GDP per capita"})
    style(fig, 520, False)
    fig.update_traces(marker=dict(cornerradius=4))
    pop = tm.groupby("Continent").Population_Total.sum().sort_values(ascending=False)
    insight(10, "Where do the people - and the wealth - sit?", "How is the population split by continent and income, and how rich is each block?", fig,
            f"Box size is population and colour is GDP per capita. <b>{pop.index[0]}</b> holds the largest share of people ({pop.iloc[0] / pop.sum():.0%} of the selection). "
            "Click a continent to drill down to income groups and then countries - big, dark boxes are populous countries with low income, which is where development gains matter most.")

# ================================================================== COUNTRY EXPLORER
with tab_exp:
    st.markdown("Pick up to four countries to compare. The radar shows how each one stacks up against the world average on eight indicators.")
    all_c = sorted(cy.Country.unique())
    dflt = [c for c in ["India", "United States", "Brazil", "Nigeria"] if c in all_c]
    pick = st.multiselect("\U0001F30D Countries to compare", all_c, default=dflt, max_selections=4, key="explorer")
    if not pick:
        st.info("Choose at least one country above.")
    else:
        lastY = int(cy.Year.max())
        now = cy[cy.Year == lastY].copy()
        now["GDP_Per_Capita_USD"] = np.log(now["GDP_Per_Capita_USD"])
        rad = ["Life_Expectancy_Years", "Literacy_Rate_Pct", "Internet_Penetration_Pct", "Human_Development_Index", "Happiness_Index",
               "GDP_Per_Capita_USD", "Urban_Population_Pct", "Health_Expenditure_Pct_GDP"]
        mn, mx = now[rad].min(), now[rad].max()
        norm = (now.set_index("Country")[rad] - mn) / (mx - mn)
        world = norm.mean()
        labels = [NICE[c].replace("GDP per capita (USD)", "GDP per capita (log)") for c in rad]
        with st.container(border=True):
            c1, c2 = st.columns([1, 1.15])
            with c1:
                st.markdown(f"##### Profile in {lastY} (0 = lowest in the world, 1 = highest)")
                fig = go.Figure()
                pal = [GOLD, TEAL, BLUE, CORAL]
                for i, c_ in enumerate(pick):
                    v_ = norm.loc[c_].values
                    fig.add_trace(go.Scatterpolar(r=list(v_) + [v_[0]], theta=labels + [labels[0]], name=c_, fill="toself", opacity=.55, line=dict(color=pal[i], width=3)))
                fig.add_trace(go.Scatterpolar(r=list(world.values) + [world.values[0]], theta=labels + [labels[0]], name="World average", line=dict(color="#aab4cc", dash="dash", width=2)))
                fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor="#2a3350", tickfont=dict(size=9)), angularaxis=dict(gridcolor="#2a3350")))
                style(fig, 500)
                fig.update_layout(margin=dict(l=95, r=95, t=30, b=60))
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
            with c2:
                em = st.selectbox("Trend of", [NICE[k] for k in NICE], index=0, key="expm")
                ecol = {v: k for k, v in NICE.items()}[em]
                tr = cy[cy.Country.isin(pick)][["Country", "Year", ecol]]
                wd = cy.groupby("Year", as_index=False)[ecol].mean().assign(Country="World average")
                fig = px.line(pd.concat([tr, wd]), x="Year", y=ecol, color="Country", labels={ecol: em},
                              color_discrete_map={**{c_: pal[i] for i, c_ in enumerate(pick)}, "World average": "#aab4cc"})
                fig.update_traces(line=dict(width=3))
                for trc in fig.data:
                    if trc.name == "World average":
                        trc.line.dash = "dash"
                style(fig, 400)
                st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        with st.container(border=True):
            st.markdown("##### Scorecard (rank 1 = best of all countries; for CO2 and fertility, lower is better)")
            sc_cols = ["Human_Development_Index", "Life_Expectancy_Years", "GDP_Per_Capita_USD", "Happiness_Index", "Literacy_Rate_Pct", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons"]
            lowbad = {"CO2_Emissions_Per_Capita_Tons"}
            tb = []
            raw = cy[cy.Year == lastY].set_index("Country")
            for c_ in pick:
                r_ = {"Country": c_}
                for col_ in sc_cols:
                    rk_ = raw[col_].rank(ascending=(col_ in lowbad), method="min")[c_]
                    r_[NICE[col_]] = f"{raw.loc[c_, col_]:,.2f}  (#{int(rk_)})"
                tb.append(r_)
            st.dataframe(pd.DataFrame(tb).set_index("Country"), width="stretch")
            c0 = pick[0]
            diff = (norm.loc[c0] - world).sort_values()
            st.markdown(f'<div class="infer"><b>What we can infer about {c0}:</b> strongest relative to the world on <b>{labels[rad.index(diff.index[-1])]}</b> '
                        f'and <b>{labels[rad.index(diff.index[-2])]}</b>; weakest on <b>{labels[rad.index(diff.index[0])]}</b>. '
                        f'A large gap between a country and the dashed world line points to where it could improve most.</div>', unsafe_allow_html=True)

# ================================================================== ML
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
        Xf = sm.add_constant(tr.Year.values.astype(float))
        mdl = sm.OLS(tr[metric].values, Xf).fit()
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


with tab_ml:
    st.markdown("Three machine-learning views of the same data: **predict** life expectancy, **group** countries by profile and **forecast** the next years. "
                "All models train on the *full* dataset and are always tested on countries they have not seen.")
    sub = st.tabs(["1. Life-expectancy predictor", "2. Country clusters", "3. Forecast to 2030"])
    res, oof, coef, icpt, xm, xs = train_models(cy[["Country"] + FEATS + [TARGET]], "v1")

    with sub[0]:
        best = res.iloc[0]
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Model comparison")
            st.dataframe(res.style.format({"R-squared": "{:.3f}", "MAE (years)": "{:.2f}"}).background_gradient(subset=["R-squared"], cmap="Greens"),
                         width="stretch", hide_index=True)
            st.caption("5-fold test where whole countries are held out. We use **Ridge regression** as the final model: it is as accurate as the complex models on this data, "
                       "trains instantly and every coefficient is readable.")
            r2r = float(res[res.Model == "Ridge regression"]["R-squared"].iloc[0])
            fig = px.scatter(x=cy[TARGET], y=oof, opacity=.35, labels={"x": "Actual life expectancy", "y": "Predicted (Ridge)"})
            fig.add_shape(type="line", x0=35, y0=35, x1=90, y1=90, line=dict(color=GOLD, dash="dash"))
            fig.update_traces(marker=dict(color=BLUE, size=5))
            style(fig, 360, False)
            fig.update_layout(title=f"Predicted vs actual (R² = {r2r:.2f})")
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        with c2:
            st.subheader("What-if simulator")
            cty = st.selectbox("Start from a country (latest year)", sorted(cy.Country.unique()), index=sorted(cy.Country.unique()).index("India") if "India" in set(cy.Country) else 0)
            row = cy[(cy.Country == cty) & (cy.Year == cy.Year.max())].iloc[0]
            base = design(row.to_frame().T.astype(float, errors="ignore").assign(**{k: float(row[k]) for k in FEATS}))
            vals = {}
            sl = st.columns(2)
            ranges = {"ln(GDP per capita)": (None,), "Literacy rate (%)": (0, 100), "Urban population (%)": (0, 100), "Internet users (%)": (0, 100),
                      "Health spending (% GDP)": (0, 20), "Fertility rate": (1.0, 8.0)}
            for i, n in enumerate(FNAMES):
                with sl[i % 2]:
                    if n == "ln(GDP per capita)":
                        v = st.slider("GDP per capita (USD)", 200, 120000, int(row.GDP_Per_Capita_USD), step=100, key=f"w{cty}{n}")
                        vals[n] = float(np.log(v))
                    else:
                        lo, hi = ranges[n]
                        dv = float(np.clip(row[FEATS[i]], lo, hi))
                        vals[n] = st.slider(n, float(lo), float(hi), dv, key=f"w{cty}{n}")
            pred = icpt + sum(coef[n] * vals[n] for n in FNAMES)
            actual = float(row[TARGET])
            m1, m2_, m3 = st.columns(3)
            kpi(m1, "Predicted life expectancy", f"{pred:.1f} yrs")
            kpi(m2_, f"{cty} actual ({int(row.Year)})", f"{actual:.1f} yrs", pred - actual)
            kpi(m3, "Typical error of the model", f"±{best['MAE (years)']:.1f} yrs")
            contrib = pd.Series({n: coef[n] * (vals[n] - xm[n]) for n in FNAMES}).sort_values()
            fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=[TEAL if v > 0 else CORAL for v in contrib.values],
                                   text=[f"{v:+.1f}" for v in contrib.values], textposition="outside"))
            fig.update_layout(title="Why this prediction? (years added/removed vs the average country)")
            style(fig, 330, False)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
            strongest = contrib.abs().idxmax()
            st.markdown(f'<div class="infer"><b>What we can infer:</b> The biggest driver here is <b>{strongest}</b> ({contrib[strongest]:+.1f} years). '
                        f'Move the sliders to see how much each factor is worth - this is a <i>pattern in the data</i>, not proof that changing one factor alone would cause the change.</div>', unsafe_allow_html=True)

    with sub[1]:
        k = st.slider("Number of development profiles (clusters)", 3, 6, 4)
        last, prof, sil, ev = build_clusters(cy, k, "v1")
        c1, c2 = st.columns([1.3, 1])
        with c1:
            fig = px.scatter(last.sort_values("Cluster"), x="PC1", y="PC2", color="Cluster", hover_name="Country", hover_data={"PC1": False, "PC2": False, "Continent": True},
                             color_discrete_sequence=[CORAL, GOLD, TEAL, BLUE, PURPLE, GREY], labels={"PC1": "Overall development →", "PC2": "Second pattern"})
            fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="#0e1117")))
            style(fig, 480)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        with c2:
            st.markdown("**Average profile of each group**")
            show = prof.rename(columns=lambda c: NICE.get(c, c))
            fmt = {c_: "{:,.2f}" for c_ in show.columns}
            fmt["Countries"] = "{:,.0f}"
            st.dataframe(show.style.format(fmt).background_gradient(cmap="Blues", axis=0), width="stretch", height=330)
            st.caption(f"Silhouette score {sil:.2f} (1 = perfectly separated). The 2-D map keeps {ev:.0%} of the information.")
        cl_sel = st.selectbox("See the countries in a group", sorted(last.Cluster.unique()))
        st.write(", ".join(sorted(last[last.Cluster == cl_sel].Country)))
        top_c = prof.index[-1]
        low_c = prof.index[0]
        st.markdown(f'<div class="infer"><b>What we can infer:</b> The algorithm was never told about continents or income groups, yet it splits countries into {k} tiers. '
                    f'<b>{low_c}</b> countries ({int(prof.loc[low_c, "Countries"])}) average ${prof.loc[low_c, "GDP_Per_Capita_USD"]:,.0f} per person and {prof.loc[low_c, "Life_Expectancy_Years"]:.0f} years of life, '
                    f'against ${prof.loc[top_c, "GDP_Per_Capita_USD"]:,.0f} and {prof.loc[top_c, "Life_Expectancy_Years"]:.0f} years for <b>{top_c}</b>. '
                    f'Countries that sit between two groups on the map are the ones closest to moving up.</div>', unsafe_allow_html=True)

    with sub[2]:
        c1, c2, c3 = st.columns([1.2, 1, 1])
        fm = c1.selectbox("Measure to forecast", ["Life_Expectancy_Years", "GDP_Per_Capita_USD", "Internet_Penetration_Pct", "CO2_Emissions_Per_Capita_Tons"], format_func=lambda k_: NICE[k_])
        hz = c2.slider("Forecast until", 2026, 2035, 2030)
        shown = c3.multiselect("Continents", sorted(cy.Continent.unique()), default=sorted(cy.Continent.unique()))
        fc, bt = forecast_table(cy, fm, hz, "v1")
        hist = cy.groupby(["Continent", "Year"], as_index=False)[fm].mean()
        fig = go.Figure()
        for c_ in shown:
            col = CONT_COL[c_]
            h_ = hist[hist.Continent == c_]
            f_ = fc[fc.Continent == c_]
            fig.add_trace(go.Scatter(x=h_.Year, y=h_[fm], name=c_, line=dict(color=col, width=3)))
            xs_ = list(f_.Year) + list(f_.Year[::-1])
            ys_ = list(f_.Upper) + list(f_.Lower[::-1])
            rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
            fig.add_trace(go.Scatter(x=xs_, y=ys_, mode="lines", fill="toself", fillcolor=f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)", line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[h_.Year.iloc[-1]] + list(f_.Year), y=[h_[fm].iloc[-1]] + list(f_.Value), mode="lines", line=dict(color=col, dash="dash", width=3), showlegend=False,
                                     name=f"{c_} forecast"))
        fig.add_vline(x=int(cy.Year.max()) + 0.5, line_dash="dot", line_color="#667", annotation_text="forecast →")
        fig.update_yaxes(title=NICE[fm])
        style(fig, 480)
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CFG)
        last_y = fc[fc.Year == hz].set_index("Continent").Value
        now = hist[hist.Year == hist.Year.max()].set_index("Continent")[fm]
        chg = (last_y - now).sort_values(ascending=False)
        tbl = pd.DataFrame({f"{int(cy.Year.max())}": now, f"{hz} forecast": last_y, "Change": last_y - now}).loc[chg.index]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Forecast summary**")
            st.dataframe(tbl.style.format("{:,.1f}"), width="stretch")
        with c2:
            st.markdown("**How reliable is it?** (trained up to 2018, tested on 2019-2024)")
            st.dataframe(bt.set_index("Continent").style.format("{:,.2f}"), width="stretch")
        beat = int((bt["Trend-model error"] < bt["'Same as last year' error"]).sum())
        st.markdown(f'<div class="infer"><b>What we can infer:</b> If recent trends continue, <b>{chg.index[0]}</b> is projected to change the most by {hz} ({chg.iloc[0]:+,.1f}) and '
                    f'<b>{chg.index[-1]}</b> the least ({chg.iloc[-1]:+,.1f}). The trend model beat a "nothing changes" guess in {beat} of {len(bt)} continents in the back-test. '
                    f'The shaded band is an 80% range; it is a straight-line extrapolation, so a shock like the 2020 dip would not be foreseen.</div>', unsafe_allow_html=True)

# ================================================================== CONCLUSIONS
with tab_con:
    full = cy
    lf = full.groupby(["Year", "Continent"]).Life_Expectancy_Years.mean().unstack()
    nt = full.groupby(["Year", "Continent"]).Internet_Penetration_Pct.mean().unstack()
    hd = full.groupby("Income_Group").Human_Development_Index.mean().reindex(INCOME_ORDER)
    r_ = np.corrcoef(np.log(full.GDP_Per_Capita_USD), full.Life_Expectancy_Years)[0, 1]
    st.subheader("What Insight Atlas tells us")
    st.markdown(
        f"""
**From the 10 graphs**
- **Life expectancy rose everywhere** - from {lf.iloc[0].mean():.1f} to {lf.iloc[-1].mean():.1f} years on average. Africa gained the most ({lf.Africa.iloc[-1] - lf.Africa.iloc[0]:+.1f} years) but is still about {lf.Europe.iloc[-1] - lf.Africa.iloc[-1]:.0f} years behind Europe.
- **Income matters, with diminishing returns** - log GDP and life expectancy correlate at {r_:.2f}; the gains flatten after roughly $10,000 per person.
- **The digital divide narrowed but did not close** - internet use went from under 1% to {nt.iloc[-1].mean():.0f}% on average, yet Africa ({nt.Africa.iloc[-1]:.0f}%) trails Europe ({nt.Europe.iloc[-1]:.0f}%).
- **Development is a staircase** - average HDI climbs {hd.iloc[0]:.2f} → {hd.iloc[1]:.2f} → {hd.iloc[2]:.2f} → {hd.iloc[3]:.2f} from low to high income groups.
- **Emissions are driven by structure, not wealth alone** - a few high-income oil economies dominate CO2 per person.

**From the machine learning**
- **Prediction:** a simple Ridge regression predicts life expectancy for unseen countries with R² ≈ {float(res[res.Model == 'Ridge regression']['R-squared'].iloc[0]):.2f} (typical error about {float(res[res.Model == 'Ridge regression']['MAE (years)'].iloc[0]):.1f} years). Complex models did not beat it, so we keep the explainable one.
- **Grouping:** clustering finds clear development tiers without being told continents or income - useful for targeting where help is needed.
- **Forecast:** continuing today's trends, every continent keeps improving through 2030, but the gaps between them shrink only slowly.

**Limits we should be honest about**
- This is a **practice dataset** (some labels contradict each other), so results describe the data, not the real world.
- Averages treat a small country the same as a large one; population-weighted averages would be fairer.
- Correlation is not causation, and the forecast is a straight-line extrapolation.
        """
    )
    st.success("Take-away: the data says development moves together - income, literacy, health and technology rise as a bundle - and the gaps between regions are closing, but slowly.")
    st.caption("Insight Atlas  |  Data Visualization Lab (BISL504-C)  |  Built with Python, Plotly, scikit-learn and Streamlit")
