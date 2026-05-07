"""
Optimo Menu Analytics — Streamlit Dashboard
============================================
All metrics are calculated directly from the uploaded data files.
No mock data, no hardcoded numbers.

DATA SOURCES
------------
MENU_CSV   : GAMP_Menu_Comparison_raw.csv
             936 rows × 85 cols — 4 weeks × 2 GAMP versions × 117 recipes
             Pre-computed weighted cols: weightedRIS, weightedScorescm,
             weightedScorewoscm, weightedCost2p (metric × weight, sums to 1)
AOR_CSV    : L1_2w_AOR.csv — L1 RAR, 2W AOR, 2W AOR expected per recipe
LT_CSV     : L3/LT vol share, uptake, score, 1-star share per recipe
TREND_CSV  : 12-week menu-level weekly aggregates (W06–W17)
POOL_CSV   : recipe_pool_master_enriched — full plannable pool (3,967 recipes)
PT_CSV     : Weekly protein demand trend by primaryprotein (W09–W17)
NR_CSV     : New/repeat recipe counts per week (n_new = 0 — known query issue)
DEMAND_CSV : 52-week avg customer demand % by attribute (protein/cuisine/dish_type)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

st.set_page_config(
    page_title="Optimo — Menu Analytics",
    page_icon="🍽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=DM+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:1400px!important}
.top-bar{display:flex;align-items:center;justify-content:space-between;padding:0 0 18px;border-bottom:1px solid #E8E6E0;margin-bottom:20px}
.app-name{font-size:17px;font-weight:600;letter-spacing:-.3px;color:#1A1A18}
.app-sub{font-size:13px;color:#888780}
.badge{font-family:'DM Mono',monospace;font-size:10px;font-weight:500;background:#F0EDE5;color:#5A5750;padding:3px 8px;border-radius:4px}
.status-banner{background:#FFFDF5;border:1px solid #EDE8D8;border-radius:10px;padding:12px 16px;margin-bottom:20px;display:flex;align-items:center;gap:10px;font-size:13px;color:#7A5A00}
.status-dot{width:8px;height:8px;border-radius:50%;background:#F5A623;flex-shrink:0}
.flags-box{background:#FFFDF5;border:1px solid #EDE8D8;border-radius:10px;padding:14px 16px;margin-bottom:20px}
.flags-title{font-size:10px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:#888780;margin-bottom:10px}
.flag-item{display:flex;align-items:flex-start;gap:8px;font-size:12px;color:#2A2A28;margin-bottom:7px;line-height:1.45}
.flag-item:last-child{margin-bottom:0}
.flag-dot-red{width:6px;height:6px;border-radius:50%;background:#E84040;flex-shrink:0;margin-top:4px}
.flag-dot-amber{width:6px;height:6px;border-radius:50%;background:#F5A623;flex-shrink:0;margin-top:4px}
.flag-dot-green{width:6px;height:6px;border-radius:50%;background:#27AE60;flex-shrink:0;margin-top:4px}
.kpi-row{display:flex;gap:10px;margin-bottom:22px;flex-wrap:wrap}
.kpi{flex:1;min-width:105px;background:#F8F6F0;border-radius:10px;padding:13px 15px}
.kpi-label{font-size:10px;color:#888780;font-weight:500;letter-spacing:.04em;text-transform:uppercase;margin-bottom:5px}
.kpi-value{font-size:22px;font-weight:300;color:#1A1A18;letter-spacing:-1px;line-height:1}
.kpi-delta{font-size:11px;margin-top:4px;color:#888780}
.kpi-up{color:#27AE60;font-size:11px;margin-top:4px}
.kpi-down{color:#E84040;font-size:11px;margin-top:4px}
.kpi-warn{color:#F5A623;font-size:11px;margin-top:4px}
.cc{background:#FFF;border:1px solid #ECEAE4;border-radius:12px;padding:16px 18px 12px;margin-bottom:14px}
.cc-title{font-size:13px;font-weight:600;color:#1A1A18;margin-bottom:2px}
.cc-sub{font-size:12px;color:#888780;margin-bottom:12px}
.lt{width:100%;border-collapse:collapse;font-size:12px}
.lt th{font-size:10px;font-weight:600;color:#888780;letter-spacing:.05em;text-transform:uppercase;padding:7px 10px;border-bottom:1px solid #ECEAE4;text-align:left}
.lt td{padding:8px 10px;border-bottom:1px solid #F4F2EC;color:#2A2A28;vertical-align:middle}
.lt tr:last-child td{border-bottom:none}
.lt tr:hover td{background:#FAFAF7}
.pill{display:inline-block;padding:2px 8px;border-radius:10px;font-size:10px;font-weight:600}
.pill-green{background:#D4F4E2;color:#0F6B30}
.pill-red{background:#FCDCDC;color:#8A1A1A}
.pill-amber{background:#FEF0D0;color:#8A5000}
.pill-blue{background:#DCEDFb;color:#1A5FA8}
.pill-gray{background:#EEECE6;color:#5A5750}
.stTabs [data-baseweb="tab-list"]{gap:0;border-bottom:1px solid #E8E6E0;background:transparent}
.stTabs [data-baseweb="tab"]{font-family:'DM Sans',sans-serif!important;font-size:13px!important;font-weight:400!important;color:#888780!important;padding:10px 16px!important;border-bottom:2px solid transparent!important;background:transparent!important}
.stTabs [aria-selected="true"]{color:#1A1A18!important;font-weight:600!important;border-bottom:2px solid #1A1A18!important}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none!important}
.stSelectbox label{font-size:11px!important;color:#888780!important;font-weight:500!important}
div[data-baseweb="select"]>div{background:#F8F6F0!important;border:1px solid #E0DDD5!important;border-radius:8px!important;font-size:13px!important}
hr{border:none;border-top:1px solid #ECEAE4;margin:6px 0 18px}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────
C = dict(
    blue="#2E86E8", blue_lt="#B5D4F4", teal="#1D9E75", teal_lt="#9FE1CB",
    amber="#F5A623", coral="#E84040", pink="#D4537E", purple="#7F77DD",
    gray="#888780", gray_lt="#D3D1C7", green="#27AE60", ink="#1A1A18",
)

# ─────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────
BASE = dict(
    font_family="DM Sans",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#ECEAE4"),
)
GX = dict(gridcolor="#ECEAE4", linecolor="#ECEAE4", tickfont_size=11)
GY = dict(gridcolor="#ECEAE4", linecolor="#ECEAE4", tickfont_size=11)
M  = dict(l=8, r=8, t=8,  b=8)
ML = dict(l=8, r=8, t=36, b=8)

# ─────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────
def cc(title, sub=""):
    st.markdown(
        f"<div class='cc'><div class='cc-title'>{title}</div>"
        f"<div class='cc-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

def chart(fig, key, height=None):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=key)

def kpis(*items):
    """Render a row of KPI tiles. Each item: (label, value, delta, kind)"""
    html = "<div class='kpi-row'>"
    for label, val, delta, kind in items:
        cls = {"up":"kpi-up","down":"kpi-down","warn":"kpi-warn"}.get(kind,"kpi-delta")
        html += (f"<div class='kpi'>"
                 f"<div class='kpi-label'>{label}</div>"
                 f"<div class='kpi-value'>{val}</div>"
                 f"<div class='{cls}'>{delta}</div></div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def flags_box(items):
    """items = list of (severity, text) — severity: red | amber | green"""
    rows = "".join(
        f"<div class='flag-item'>"
        f"<div class='flag-dot-{sev}'></div>"
        f"<span>{text}</span></div>"
        for sev, text in items
    )
    st.markdown(
        f"<div class='flags-box'>"
        f"<div class='flags-title'>Pre-sign-off flags</div>"
        f"{rows}</div>",
        unsafe_allow_html=True,
    )

def pill(text, kind):
    return f"<span class='pill pill-{kind}'>{text}</span>"

def _fmt(val, fmt, na="N/A"):
    """Safe format — returns na string on NaN/None."""
    try:
        return format(val, fmt) if pd.notna(val) else na
    except (TypeError, ValueError):
        return na

def _pill_rar(v):
    if pd.isna(v): return pill("N/A", "gray")
    if v > 0.01:   return pill(f"{v:+.3f}", "green")
    if v > -0.01:  return pill(f"{v:+.3f}", "gray")
    return pill(f"{v:+.3f}", "red")

def _pill_star(v):
    if pd.isna(v): return pill("N/A", "gray")
    if v < 0.08:   return pill(f"{v:.0%}", "green")
    if v < 0.15:   return pill(f"{v:.0%}", "amber")
    return pill(f"{v:.0%}", "red")

def _pill_score(v):
    if pd.isna(v): return pill("N/A", "gray")
    if v >= 4.3:   return pill(f"{v:.2f}", "green")
    if v >= 4.0:   return pill(f"{v:.2f}", "amber")
    return pill(f"{v:.2f}", "red")

def _pill_ris(v):
    if pd.isna(v): return pill("N/A", "gray")
    return pill(f"{v:.3f}", "blue" if v >= 0.65 else "amber")

# ─────────────────────────────────────────────────────────────────────
# PROTEIN GROUPING (for pool and demand alignment)
# ─────────────────────────────────────────────────────────────────────
def _grp_protein(p):
    """Group detailed primaryprotein values into main categories."""
    if pd.isna(p): return None
    p = str(p).lower()
    if "chicken" in p:                                    return "Chicken"
    if "beef" in p or "bison" in p:                       return "Beef"
    if "pork" in p:                                       return "Pork"
    if any(x in p for x in ["fish","salmon","tilapia","basa","sea bass"]): return "Fish"
    if any(x in p for x in ["shellfish","shrimp","scallop"]):              return "Shellfish"
    if "turkey" in p:                                     return "Turkey"
    if any(x in p for x in ["veggie","plantbased","egg"]): return "Veggie"
    return "Other"

def _sku_category(n):
    """Categorise a SKU name string."""
    n = n.lower()
    if any(x in n for x in ["chicken","beef","pork","salmon","shrimp",
                              "turkey","lamb","tuna","bison","duck"]):  return "Protein"
    if any(x in n for x in ["pasta","rice","potato","spaghetti",
                              "noodle","bread","flour","tortilla"]):    return "Starch"
    if any(x in n for x in ["cheese","cream","yogurt","butter",
                              "milk","sour cream","crème"]):            return "Dairy"
    if any(x in n for x in ["garlic","onion","pepper","tomato","spinach",
                              "cabbage","zucchini","lettuce","kale",
                              "broccoli","carrot","corn","mushroom",
                              "celery","beans","lentil","chickpea",
                              "lemon","lime","apple","pear"]):          return "Produce"
    if any(x in n for x in ["oil","fat"]):                             return "Oil/Fat"
    if any(x in n for x in ["sauce","paste","dressing","marinade",
                              "mayo","vinegar","soy","mustard",
                              "salsa","pesto"]):                        return "Sauce"
    if any(x in n for x in ["spice","blend","powder","seasoning",
                              "herb","thyme","basil","cumin","paprika",
                              "chili","oregano","salt","pepper",
                              "cilantro","parsley"]):                   return "Herb/Spice"
    return "Pantry"

# ─────────────────────────────────────────────────────────────────────
# FILE PATHS — update to your local folder
# ─────────────────────────────────────────────────────────────────────
MENU_CSV   = "GAMP_Menu_Comparison_raw.csv"
AOR_CSV    = "L1_2w_AOR.csv"
LT_CSV     = "__L3___LT_vol_share_ore__share_of_1_stars.csv"
TREND_CSV  = "__Trend_data.csv"
POOL_CSV   = "recipe_pool.csv"
PT_CSV     = "Protein_mix_trend.csv"
NR_CSV     = "__n_new_recipes_in_trend.csv"
DEMAND_CSV = "Averaged_demand.csv"

# ─────────────────────────────────────────────────────────────────────
# CACHED DATA LOADERS
# ─────────────────────────────────────────────────────────────────────

@st.cache_data
def load_all_menus():
    """
    Load the full combined menu CSV.
    Returns all 936 rows (4 weeks × 2 versions × 117 recipes).
    Used to populate the week/version selectors dynamically.
    """
    return pd.read_csv(MENU_CSV)


@st.cache_data
def load_menu(week: str, version: str) -> pd.DataFrame:
    """
    Load and enrich the planned menu for a specific week + GAMP version.

    Joins:
      - AOR file    → l1_rar, l1_2w_aor, l1_2w_aor_expected  (LEFT JOIN on code)
      - LT file     → l3/lt vol share, score, 1-star windows  (LEFT JOIN on code)

    Adds:
      - is_ineligible : True where mainprotein/cuisine/dishtype == "Gap"
      - is_new        : from isnewrecipe col
      - data_issues   : True where any key attribute is null/Gap
      - no_l1_history : non-new recipes with no AOR match
    """
    raw = pd.read_csv(MENU_CSV)
    # Filter to selected week and version
    df = raw[(raw["hellofreshweek"] == week) &
             (raw["gampversion"]    == version)].copy()

    # ── Rename to internal names ──────────────────────────────────
    df = df.rename(columns={
        "code":                            "recipe_code_main",
        "title":                           "name",
        "mainprotein":                     "protein",
        # cuisine, dishtype kept as-is — raw values used directly
        "preference":                      "preference",
        "cost2p":                          "cost_2p",
        "scorescm":                        "l1_score",
        "scorewoscm":                      "l1_score_wo_scm",
        "volumesharelast":                 "l1_vol",
        "volumeshare2last":                "l2_vol",
        "uptake_ratio":                    "l1_uptake",
        "retention_performance_normalized":"ris",
        "avg_swap_ratio":                  "l1_sir",
        "seasonalityrisk_w1":              "skew_risk",
        "nb_of_appearances":               "nb_appearances",
        "isnewrecipe":                     "is_new",
        "slotnumber":                      "slot",
        "skucount":                        "sku_count",
        "skuname":                         "sku_names_raw",
        "skucode":                         "sku_codes_raw",
        "handsontime":                     "hands_on_time",
        "totaltime":                       "total_time",
        "calories":                        "calories",
        "absolutelastused":                "last_used",
        # Weighted cols — pre-computed in CSV (metric × weight, weight sums to 1)
        "weightedRIS":                     "weightedRIS",
        "weightedScorescm":                "weightedScorescm",
        "weightedScorewoscm":              "weightedScorewoscm",
        "weightedCost2p":                  "weightedCost2p",
        "weight":                          "weight",
    })

    # ── Join L1 AOR + RAR ─────────────────────────────────────────
    # Source: AOR_CSV — computed from facts_recipes_ordered_enriched
    # Covers recipes planned in last ~104 weeks. 29 recipes have no match
    # (17 new recipes + 12 with outing >104 weeks ago).
    aor = pd.read_csv(AOR_CSV)[
        ["recipe_code_main","l1_2w_aor","l1_2w_aor_expected","l1_rar"]
    ]
    df = df.merge(aor, on="recipe_code_main", how="left")

    # ── Join L3/LT performance windows ───────────────────────────
    # Source: LT_CSV — computed from facts_recipes_ordered_enriched (104w)
    lt = pd.read_csv(LT_CSV)
    lt = lt[~lt["recipe_code_main"].str.contains(r"-SP-|-FR-", na=False)]
    df = df.merge(lt, on="recipe_code_main", how="left")

    # ── Derived flags ─────────────────────────────────────────────
    df["is_new"]     = df["is_new"].astype(bool)
    df["skew_risk"]  = df["skew_risk"].fillna(1).clip(1, 4).astype(int)

    # Ineligible: GAMP assigned "Gap" to a key attribute — no real recipe
    df["is_ineligible"] = (
        (df["protein"]  == "Gap") |
        (df["cuisine"]  == "Gap") |
        (df["dishtype"] == "Gap")
    )
    df["inelig_reason"] = df.apply(lambda r: (
        "Gap protein"   if r["protein"]  == "Gap" else
        "Gap cuisine"   if r["cuisine"]  == "Gap" else
        "Gap dish type" if r["dishtype"] == "Gap" else ""
    ), axis=1)

    # Data quality: key attribute null or Gap
    df["data_issues"] = (
        df["protein"].isna()  | (df["protein"]  == "Gap") |
        df["cuisine"].isna()  | (df["cuisine"]  == "Gap") |
        df["preference"].isna() |
        df["ris"].isna()
    )
    df["missing_field"] = df.apply(lambda r: (
        "protein"     if (pd.isna(r.get("protein"))    or r.get("protein")    == "Gap") else
        "cuisine"     if (pd.isna(r.get("cuisine"))    or r.get("cuisine")    == "Gap") else
        "preference"  if pd.isna(r.get("preference"))  else
        "RIS"         if pd.isna(r.get("ris"))         else ""
    ), axis=1)

    # No L1 history: non-new recipes that didn't match the AOR file
    df["no_l1_history"] = df["l1_rar"].isna() & ~df["is_new"]

    # Convenience delta cols — only where both values exist
    df["score_delta"] = df["l1_score"] - df["lt_avg_score_wo_scm"]
    df["vol_delta"]   = df["l1_vol"]   - df["lt_avg_vol_share"]
    df["outing"]      = df["nb_appearances"].fillna(0).astype(int)

    return df


@st.cache_data
def load_pool(planned_codes: frozenset):
    """
    Load recipe pool and split into planned vs unplanned.
    Source: POOL_CSV (recipe_pool_master_enriched, HF only)
    """
    pool = pd.read_csv(POOL_CSV)
    hf   = pool[pool["brand"] == "HelloFresh"].copy()
    hf["prot_group"] = hf["primaryprotein"].apply(_grp_protein)
    unplanned = hf[~hf["code"].isin(planned_codes)]
    return hf, unplanned


@st.cache_data
def load_trend():
    """
    Load 12-week trend data (W06–W17) and protein mix trend.
    Source: TREND_CSV + PT_CSV + NR_CSV
    """
    t  = pd.read_csv(TREND_CSV)
    nr = pd.read_csv(NR_CSV)
    pt = pd.read_csv(PT_CSV)

    # Merge new/repeat counts into trend
    t = t.merge(
        nr[["hellofresh_week","n_new_recipes","n_repeat_recipes","n_total_recipes"]],
        on="hellofresh_week", how="left", suffixes=("","_nr"),
    )
    t["wk"] = t["hellofresh_week"].str.replace("2026-","W", regex=False)

    # Group detailed protein into main categories and sum pct per week×group
    pt["protein_group"] = pt["primary_protein"].apply(_grp_protein)
    pt_grp = (pt.dropna(subset=["protein_group"])
               .groupby(["hf_week","protein_group"])["pct"]
               .sum().reset_index())
    pt_grp["wk"] = pt_grp["hf_week"].str.replace("2026-","W", regex=False)

    return t, pt_grp


@st.cache_data
def load_demand():
    """
    Load 52-week averaged customer demand by attribute.
    Source: DEMAND_CSV — computed from facts_recipes_ordered_enriched (W33 2025–W17 2026)
    Returns separate dataframes for protein, cuisine, dish_type.
    """
    dem = pd.read_csv(DEMAND_CSV)

    # Protein: group detailed primaryprotein into main categories, re-normalise
    dp = dem[dem["attribute_type"] == "protein"].copy()
    dp["grp"] = dp["attribute_value"].apply(_grp_protein)
    dp_grp = (dp.dropna(subset=["grp"])
               .groupby("grp")["avg_demand_pct"].sum().reset_index())
    total = dp_grp["avg_demand_pct"].sum()
    dp_grp["avg_demand_pct"] = (dp_grp["avg_demand_pct"] / total * 100).round(2)
    dp_grp.columns = ["protein","demand_pct"]

    dc = dem[dem["attribute_type"] == "cuisine"][
        ["attribute_value","avg_demand_pct"]].copy()
    dc.columns = ["cuisine","demand_pct"]

    dd = dem[dem["attribute_type"] == "dish_type"][
        ["attribute_value","avg_demand_pct"]].copy()
    dd.columns = ["dish_type","demand_pct"]

    return dp_grp, dc, dd


def build_svd(df_menu, prot_dem, cuis_dem, dish_dem):
    """
    Build supply-vs-demand tables for protein, cuisine, and dish type.
    Supply = % of planned recipes per value (from menu CSV).
    Demand = 52w historical customer selection % (from demand CSV).
    Gap = supply_pct − demand_pct.
    Status = Surplus (gap>2pp) | Gap (gap<-2pp) | On target.
    """
    def _build(sup_ser, dem_df, key):
        sup = sup_ser.reset_index()
        sup.columns = [key,"supply_pct"]
        sup["supply_pct"] = sup["supply_pct"].round(2)
        merged = sup.merge(dem_df, on=key, how="left")
        merged["demand_pct"] = merged["demand_pct"].round(2)
        merged["gap_pp"] = (merged["supply_pct"] - merged["demand_pct"]).round(2)
        merged["status"] = merged["gap_pp"].apply(
            lambda g: "Surplus"     if pd.notna(g) and g >  2 else
                      "Gap"         if pd.notna(g) and g < -2 else
                      "On target"   if pd.notna(g) else "No demand data"
        )
        return merged

    # Exclude Gap rows from supply calculation
    valid = df_menu[~df_menu["is_ineligible"]]
    prot_sup = valid["protein"].value_counts(normalize=True).mul(100)
    cuis_sup = valid["cuisine"].dropna().value_counts(normalize=True).mul(100)
    dish_sup = valid["dishtype"].dropna().value_counts(normalize=True).mul(100)

    return (
        _build(prot_sup, prot_dem, "protein"),
        _build(cuis_sup, cuis_dem, "cuisine"),
        _build(dish_sup, dish_dem, "dish_type"),
    )


def build_sku_data(df_menu):
    """
    Parse SKU names from pipe-delimited skuname column.
    Returns: top SKUs list, category list, frequency distribution.
    Source: skuname column in menu CSV.
    """
    all_skus = []
    for raw in df_menu["sku_names_raw"].dropna():
        for s in str(raw).split("|"):
            s = s.strip()
            if s:
                all_skus.append(s)
    sku_counts = Counter(all_skus)
    top        = sku_counts.most_common(25)
    names      = [s[0][:40] for s in top]
    cnts       = [s[1]      for s in top]
    cats       = [_sku_category(s[0]) for s in top]
    total      = len(sku_counts)
    single     = sum(1 for v in sku_counts.values() if v == 1)
    most_used  = top[0][0] if top else "N/A"
    most_cnt   = top[0][1] if top else 0

    # Frequency distribution buckets
    freq = Counter(sku_counts.values())
    freq_labels = ["1 recipe","2","3–5","6–10","11–15","16+"]
    freq_counts = [
        sum(v for k,v in freq.items() if k == 1),
        sum(v for k,v in freq.items() if k == 2),
        sum(v for k,v in freq.items() if 3 <= k <= 5),
        sum(v for k,v in freq.items() if 6 <= k <= 10),
        sum(v for k,v in freq.items() if 11 <= k <= 15),
        sum(v for k,v in freq.items() if k >= 16),
    ]
    return names, cnts, cats, total, single, most_used, most_cnt, freq_labels, freq_counts


# ─────────────────────────────────────────────────────────────────────
# STEP 1: LOAD FULL MENU TO POPULATE SELECTORS
# Must happen before any widget or data-dependent code.
# ─────────────────────────────────────────────────────────────────────
_all_menus   = load_all_menus()
_avail_weeks = sorted(_all_menus["hellofreshweek"].unique())
_avail_vers  = sorted(_all_menus["gampversion"].unique())

# ─────────────────────────────────────────────────────────────────────
# TOP BAR + SELECTORS
# ─────────────────────────────────────────────────────────────────────
st.markdown("""<div class='top-bar'>
  <div style='display:flex;align-items:baseline;gap:12px'>
    <span class='app-name'>Optimo</span>
    <span class='app-sub'>Menu Analytics</span>
  </div>
  <span class='badge'>v3.0 · pre-sign-off</span>
</div>""", unsafe_allow_html=True)

# Selectors are dynamically populated from the combined CSV.
# Changing any selector triggers a full Streamlit re-run,
# reloading df via load_menu(hf_week, gamp_v).
c1, c2, c3, c4, _ = st.columns([1.3, 1.2, 1.4, 1.1, 2])
with c1: market  = st.selectbox("Market",       ["HelloFresh CA","Chef's Plate CA"])
with c2: hf_week = st.selectbox("HF Week",      _avail_weeks)
with c3: gamp_v  = st.selectbox("GAMP version", _avail_vers)
with c4: pref_f  = st.selectbox("Preference",   ["All","Classic","Family","Veggie","Quick & Easy"])

# ─────────────────────────────────────────────────────────────────────
# STEP 2: LOAD DATA FOR SELECTED WEEK + VERSION
# All downstream metrics derive from df.
# ─────────────────────────────────────────────────────────────────────
df = load_menu(hf_week, gamp_v)

# Apply preference filter if selected
if pref_f != "All":
    df_view = df[df["preference"].str.contains(pref_f, case=False, na=False)]
else:
    df_view = df

# Load supporting datasets
pool_df, unplanned_df = load_pool(frozenset(df["recipe_code_main"]))
trend_df, pt_grp      = load_trend()
prot_dem, cuis_dem, dish_dem = load_demand()
prot_tbl, cuis_tbl, dish_tbl = build_svd(df_view, prot_dem, cuis_dem, dish_dem)
(sku_names, sku_cnt, sku_cat,
 total_unique_skus, single_use_skus,
 most_used_sku, most_used_count,
 sku_freq_labels, sku_freq_counts) = build_sku_data(df_view)

# ─────────────────────────────────────────────────────────────────────
# STEP 3: COMPUTE ALL PROJECTED MENU-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────

# --- Weighted sums (pre-computed in CSV: metric × normalised weight) ---
proj_ris        = round(df_view["weightedRIS"].sum(),        3)
proj_cost       = round(df_view["weightedCost2p"].sum(),     2)
proj_score      = round(df_view["weightedScorescm"].sum(),   3)
proj_score_wo   = round(df_view["weightedScorewoscm"].sum(), 3)

# --- Weighted means for metrics not pre-computed in CSV ---
def _wmean(series, weights, ndigits):
    mask = series.notna()
    if not mask.any(): return None
    w = weights[mask]
    return round((series[mask] * w).sum() / w.sum(), ndigits)

proj_rar        = _wmean(df_view["l1_rar"],            df_view["weight"], 3)
proj_aor        = _wmean(df_view["l1_2w_aor"],         df_view["weight"], 3)
proj_uptake     = _wmean(df_view["l1_uptake"],          df_view["weight"], 4)
proj_1star      = _wmean(df_view["l3_avg_share_1star"], df_view["weight"], 4)

# --- Counts ---
n_recipes       = len(df_view)
n_new           = int(df_view["is_new"].sum())
n_ineligible    = int(df_view["is_ineligible"].sum())
n_high_skew     = int((df_view["skew_risk"] >= 3).sum())
n_data_issues   = int(df_view["data_issues"].sum())
n_no_l1         = int(df_view["no_l1_history"].sum())

# --- Pool stats ---
pool_total          = len(pd.read_csv(POOL_CSV))
pool_hf_count       = len(pool_df)
pool_ready          = int((pool_df["status"] == "READY FOR MENU PLANNING").sum())
pool_avg_ris        = round(pool_df["2w_RIS"].mean(), 3)
pool_median_ris     = round(pool_df["2w_RIS"].median(), 3)
pool_high_ris_unplan= int((unplanned_df["2w_RIS"] > 0.75).sum())

# --- Pre-compute flags ---
flags_neg_rar    = df_view[df_view["l1_rar"].fillna(0) < -0.02]
flags_high_1star = df_view[df_view["l3_avg_share_1star"].fillna(0) > 0.15]
flags_below_med  = df_view[df_view["ris"].notna() & (df_view["ris"] < pool_median_ris)]
flags_high_skew  = df_view[df_view["skew_risk"] >= 3]
flags_ineligible = df_view[df_view["is_ineligible"]]
flags_data       = df_view[df_view["data_issues"]]

# --- Trend KPIs (last 4 weeks of trend file) ---
_last4         = trend_df.tail(4)
trend_ris_4w   = round(_last4["avg_ris"].mean(), 3)
trend_score_4w = round(_last4["avg_score_wo_scm"].mean(), 3)
trend_aor_4w   = round(_last4["avg_2w_aor"].mean(), 3)

# --- Pool RIS by protein ---
pool_ris_prot = (pool_df.groupby("prot_group")["2w_RIS"]
                 .mean().round(3).dropna())
menu_ris_prot = (df_view[~df_view["is_ineligible"]]
                 .groupby("protein")["ris"]
                 .mean().round(3).dropna())
ris_prot_shared = sorted(set(pool_ris_prot.index) & set(menu_ris_prot.index))

# STATUS BANNER
st.markdown(f"""<div class='status-banner'>
  <div class='status-dot'></div>
  <strong>{hf_week} · {gamp_v} · {market}</strong>&nbsp;&nbsp;
  Status: <strong>Pending sign-off</strong>&nbsp;·&nbsp;
  Data through W17&nbsp;·&nbsp;
  <strong>{n_recipes}</strong> recipes &nbsp;·&nbsp;
  <strong>{n_new}</strong> new &nbsp;·&nbsp;
  Weighted RIS: <strong>{proj_ris:.3f}</strong>&nbsp;·&nbsp;
  Weighted cost: <strong>${proj_cost:.2f}</strong>
</div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "1 — Composition",
    "2 — Projected performance",
    "3 — Trends",
    "4 — Supply vs demand",
    "5 — Pool health",
    "6 — Ingredients & SKUs",
    "7 — Menu comparison",
])

# ═════════════════════════════════════════════════════════════════════
# TAB 1 — COMPOSITION
# What does the planned menu look like?
# Source: menu CSV (direct attribute counts)
# ═════════════════════════════════════════════════════════════════════
with t1:
    # Dynamic flags — computed from real data
    _t1_flags = []
    for _, row in prot_tbl.dropna(subset=["gap_pp"]).sort_values("gap_pp",key=abs,ascending=False).head(2).iterrows():
        if abs(row.gap_pp) > 3:
            _d = "over-indexed" if row.gap_pp > 0 else "under-supplied"
            _t1_flags.append(("amber", f"{row.protein}: {row.supply_pct:.1f}% menu vs {row.demand_pct:.1f}% demand — {_d}"))
    if n_ineligible > 0:
        _t1_flags.append(("amber", f"{n_ineligible} Gap slots — GAMP could not assign a recipe"))
    if proj_ris >= pool_median_ris:
        _t1_flags.append(("green", f"Weighted RIS {proj_ris:.3f} ≥ pool median {pool_median_ris:.3f}"))
    else:
        _t1_flags.append(("amber", f"Weighted RIS {proj_ris:.3f} < pool median {pool_median_ris:.3f}"))
    if not _t1_flags:
        _t1_flags = [("green","No major composition flags")]
    flags_box(_t1_flags)

    kpis(
        ("Total recipes",    str(n_recipes),        f"{hf_week} · {gamp_v}", ""),
        ("Weighted RIS",     f"{proj_ris:.3f}",     f"pool median {pool_median_ris:.3f}", "up" if proj_ris >= pool_median_ris else "down"),
        ("Weighted score",   f"{proj_score_wo:.3f}","w/o SCM", ""),
        ("Weighted 2P cost", f"${proj_cost:.2f}",   "vs $12.00 target", "warn" if proj_cost > 12.0 else "up"),
        ("Gap slots",        str(n_ineligible),     "no recipe assigned", "down" if n_ineligible > 0 else "up"),
        ("High skew risk",   str(n_high_skew),      "risk ≥3", "warn" if n_high_skew > 0 else "up"),
        ("New recipes",      str(n_new),            "no L1/L3 history", "warn" if n_new > 3 else ""),
    )

    ca, cb = st.columns(2)

    with ca:
        # Protein donut — counts of valid (non-Gap) recipes by mainprotein
        # Source: protein col from menu CSV
        _prot_valid = df_view[df_view["protein"] != "Gap"]["protein"].value_counts()
        _prot_colors_map = {
            "Chicken":C["blue"],"Beef":C["coral"],"Pork":C["teal"],
            "Fish":C["amber"],"Veggie":C["gray"],"Turkey":C["purple"],
            "Shellfish":C["pink"],"Plantbased":C["teal_lt"],
            "Beef and Pork":C["coral"],"Other":C["gray_lt"],
        }
        fig = go.Figure(go.Pie(
            labels=_prot_valid.index.tolist(),
            values=_prot_valid.values.tolist(),
            hole=0.62, textinfo="none",
            marker_colors=[_prot_colors_map.get(p, C["gray_lt"]) for p in _prot_valid.index],
            hovertemplate="<b>%{label}</b><br>%{value} recipes (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            **BASE, margin=dict(l=8,r=8,t=8,b=8),
            legend=dict(font_size=10,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=-0.1,x=0.5,xanchor="center"),
            height=270, showlegend=True,
        )
        cc("Protein mix", f"Recipe count by mainprotein · {n_recipes} planned · {n_ineligible} Gap slots excluded")
        chart(fig, "donut_prot")

    with cb:
        # Cuisine × protein heatmap — count recipes per cell
        # Source: cuisine + protein cols from menu CSV, no normalisation
        _valid = df_view[~df_view["is_ineligible"]].copy()
        _prots = _valid["protein"].value_counts().head(7).index.tolist()
        _cuis  = _valid["cuisine"].dropna().value_counts().head(8).index.tolist()
        _z = []
        for c_ in _cuis:
            row_ = []
            for p_ in _prots:
                row_.append(int((((_valid["protein"]==p_) & (_valid["cuisine"]==c_)).sum())))
            _z.append(row_)
        fig = go.Figure(go.Heatmap(
            z=_z, x=_prots, y=_cuis,
            colorscale=[[0,"#EBF3FC"],[0.5,"#85B7EB"],[1,"#185FA5"]],
            showscale=True, text=_z, texttemplate="%{text}",
            hovertemplate="<b>%{y} × %{x}</b><br>%{z} recipes<extra></extra>",
            colorbar=dict(thickness=10,len=0.8,tickfont_size=10),
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(tickfont_size=10, side="bottom", tickangle=-20),
            yaxis=dict(tickfont_size=10),
            height=270, margin=dict(l=80,r=30,t=8,b=60),
        )
        _total_shown = sum(sum(r) for r in _z)
        cc("Cuisine × protein heatmap",
           f"Top 8 cuisines × top 7 proteins · {_total_shown} of {len(_valid)} valid recipes shown · Gap slots excluded")
        chart(fig, "heatmap_cp")

    cc2, cd = st.columns(2)

    with cc2:
        # Dish type distribution — raw dishtype values from CSV, all shown
        # Source: dishtype col in menu CSV
        _dish_valid = df_view[df_view["dishtype"] != "Gap"]["dishtype"].dropna().value_counts().sort_values()
        fig = go.Figure(go.Bar(
            y=_dish_valid.index.tolist(),
            x=_dish_valid.values.tolist(),
            orientation="h",
            marker_color=C["blue"],
            marker_cornerradius=3,
            hovertemplate="<b>%{y}</b>: %{x} recipes<extra></extra>",
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX, dtick=1),
            yaxis=dict(**GY),
            margin=M, height=max(200, len(_dish_valid)*20+40),
        )
        cc("Dish type distribution",
           "All dishtype values from menu CSV · raw, no grouping · Gap excluded")
        chart(fig, "dish_type")

    with cd:
        # RIS distribution: pool vs planned menu
        # Pool: pool_df["2w_RIS"] · Menu: df_view["ris"]
        bins_l = [f"{i/10:.1f}" for i in range(10)]
        pool_h = [0]*10
        for r in pool_df["2w_RIS"].dropna():
            pool_h[min(int(r*10), 9)] += 1
        menu_h = [0]*10
        for r in df_view["ris"].dropna():
            menu_h[min(int(r*10), 9)] += 1
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bins_l, y=pool_h, name="Full pool",
            marker_color=C["blue_lt"], marker_cornerradius=2))
        fig.add_trace(go.Bar(x=bins_l, y=menu_h, name="Planned menu",
            marker_color=C["teal"], marker_cornerradius=2))
        fig.update_layout(
            **BASE, barmode="overlay",
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.06,x=0),
            xaxis=dict(**GX,title="RIS score",title_font_size=11),
            yaxis=dict(**GY,title="Count",title_font_size=11),
            margin=ML, height=270,
        )
        cc("RIS distribution — pool vs planned",
           "Are we selecting from the top of the pool? · retention_performance_normalized from menu CSV")
        chart(fig, "ris_dist")

    # Attribute radar — variety/entropy scores computed from real menu data
    def _radar_scores(sub):
        if len(sub) == 0: return [0]*5
        def _entropy(ser):
            p = ser.value_counts(normalize=True)
            return float(-(p * np.log(p + 1e-9)).sum() / np.log(max(len(p),2)))
        pv = _entropy(sub["protein"].dropna())
        cv = _entropy(sub["cuisine"].dropna())
        dv = _entropy(sub["dishtype"].dropna())
        pf = _entropy(sub["preference"].dropna())
        veg= float(sub["protein"].str.lower().str.contains("veggie|plantbased",na=False).mean())*3
        return [round(min(v,1),2) for v in [pv,cv,dv,pf,veg]]

    _full = _radar_scores(df_view[~df_view["is_ineligible"]])
    _fam  = _radar_scores(df_view[df_view["preference"].str.contains("Family",na=False,case=False)])
    _veg2 = _radar_scores(df_view[df_view["preference"].str.contains("Veggie",na=False,case=False)])
    if df_view["preference"].str.contains("Family",na=False,case=False).sum() < 3:  _fam  = _full
    if df_view["preference"].str.contains("Veggie",na=False,case=False).sum() < 3:  _veg2 = _full

    _attrs_r = ["Protein variety","Cuisine variety","Dish type mix","Pref variety","Veg share"]
    _ac = _attrs_r + [_attrs_r[0]]
    fig = go.Figure()
    for vals, name, color in [(_full,"Full menu",C["blue"]),(_fam,"Family",C["teal"]),(_veg2,"Veggie",C["coral"])]:
        vc = vals + [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vc, theta=_ac, fill="toself", name=name,
            line_color=color, line_width=2,
            hovertemplate="<b>%{theta}</b>: %{r:.2f}<extra></extra>",
        ))
    fig.update_layout(
        **BASE,
        polar=dict(bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True,range=[0,1],gridcolor="#ECEAE4",tickfont_size=9),
            angularaxis=dict(gridcolor="#ECEAE4",tickfont_size=11)),
        legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                    orientation="h",y=-0.08,x=0.5,xanchor="center"),
        height=320, margin=dict(l=30,r=30,t=20,b=40),
    )
    cc("Attribute variety radar",
       "Normalised entropy score per attribute (0=no variety, 1=max variety) · computed from menu CSV")
    chart(fig, "radar")


# ═════════════════════════════════════════════════════════════════════
# TAB 2 — PROJECTED PERFORMANCE
# Based on historical L1 metrics of planned recipes
# Sources: AOR_CSV (L1 RAR, AOR) · LT_CSV (L3/LT) · menu CSV (RIS, score, vol)
# ═════════════════════════════════════════════════════════════════════
with t2:
    sc_col, _ = st.columns([1, 4])
    with sc_col:
        score_threshold = st.number_input(
            "Low score threshold", min_value=1.0, max_value=5.0,
            value=3.5, step=0.1, format="%.1f",
            help="Flag recipes with L1 score below this value",
        )
    flags_low_score = df_view[df_view["l1_score"] < score_threshold]

    # Dynamic flags
    _t2 = []
    if len(flags_neg_rar) > 0:
        _n = ", ".join(flags_neg_rar["name"].tolist()[:3])
        _m = f"… +{len(flags_neg_rar)-3}" if len(flags_neg_rar) > 3 else ""
        _t2.append(("red", f"{len(flags_neg_rar)} recipes negative L1 RAR: {_n}{_m}"))
    if len(flags_high_1star) > 0:
        _n = ", ".join(flags_high_1star["name"].tolist()[:3])
        _m = f"… +{len(flags_high_1star)-3}" if len(flags_high_1star) > 3 else ""
        _t2.append(("red", f"{len(flags_high_1star)} recipes 1-star share >15%: {_n}{_m}"))
    if len(flags_low_score) > 0:
        _n = ", ".join(flags_low_score["name"].tolist()[:3])
        _m = f"… +{len(flags_low_score)-3}" if len(flags_low_score) > 3 else ""
        _t2.append(("red", f"{len(flags_low_score)} recipes below {score_threshold:.1f}: {_n}{_m}"))
    if n_high_skew > 0:
        _n = ", ".join(flags_high_skew["name"].tolist()[:3])
        _m = f"… +{n_high_skew-3}" if n_high_skew > 3 else ""
        _t2.append(("amber", f"{n_high_skew} recipes skew risk ≥3: {_n}{_m}"))
    if len(flags_below_med) > 0:
        _t2.append(("amber", f"{len(flags_below_med)} recipes below pool median RIS ({pool_median_ris:.3f})"))
    if n_no_l1 > 0:
        _t2.append(("amber", f"{n_no_l1} non-new recipes have no L1 history (last planned >104w ago)"))
    if proj_rar is not None:
        _t2.append(("green" if proj_rar > 0 else "amber", f"Avg L1 RAR {proj_rar:+.3f} vs retention expectation"))
    flags_box(_t2 if _t2 else [("green","No major performance flags")])

    kpis(
        ("Weighted RIS",     f"{proj_ris:.3f}",                              "vol-share weighted", ""),
        ("Weighted score wo",f"{proj_score_wo:.3f}",                         "vol-share weighted", ""),
        ("Avg L1 vol share", _fmt(df_view["l1_vol"].mean(), ".1%"),           "from menu CSV", ""),
        ("Weighted uptake",  _fmt(proj_uptake, ".1%") if proj_uptake else "N/A","vol-share weighted", ""),
        ("Weighted L1 RAR",  _fmt(proj_rar, "+.3f") if proj_rar else "N/A",  "vol-share weighted",
                             "up" if proj_rar and proj_rar>0 else "down"),
        ("Weighted 2W AOR",  _fmt(proj_aor, ".3f") if proj_aor else "N/A",   "vol-share weighted", ""),
        ("Low score",        str(len(flags_low_score)),                       f"< {score_threshold:.1f}",
                             "down" if len(flags_low_score)>0 else "up"),
        ("High skew",        str(n_high_skew),                                "risk ≥3",
                             "warn" if n_high_skew>0 else "up"),
    )

    # Vol share × RAR scatter
    # x = l1_vol (volumesharelast from menu CSV)
    # y = l1_rar (from AOR_CSV)
    # bubble size = ris (retention_performance_normalized from menu CSV)
    def _fmt_hover(r):
        return (
            f"<b>{r.name}</b><br>"
            f"Protein: {r.protein}<br>"
            f"Cuisine: {r.cuisine}<br>"
            f"L1 vol share: {_fmt(r.l1_vol,'.1%')}<br>"
            f"L1 RAR: {_fmt(r.l1_rar,'+.3f')}<br>"
            f"RIS: {_fmt(r.ris,'.3f')}<br>"
            f"L1 score wo SCM: {_fmt(r.l1_score_wo_scm,'.2f')}<br>"
            f"Skew risk: {r.skew_risk}"
        )
    _prot_colors_map2 = {
        "Chicken":C["blue"],"Beef":C["coral"],"Pork":C["teal"],
        "Fish":C["amber"],"Veggie":C["gray"],"Turkey":C["purple"],
        "Shellfish":C["pink"],"Plantbased":C["teal_lt"],
        "Beef and Pork":C["coral"],"Gap":C["gray_lt"],
    }
    fig = go.Figure()
    for prot in df_view[~df_view["is_ineligible"]]["protein"].dropna().unique():
        sub = df_view[df_view["protein"] == prot].copy()
        sub = sub.dropna(subset=["l1_vol"])
        if len(sub) == 0: continue
        fig.add_trace(go.Scatter(
            x=sub["l1_vol"], y=sub["l1_rar"].fillna(0),
            mode="markers", name=prot,
            marker=dict(
                size=[(r.ris*26+7) if pd.notna(r.ris) else 7 for r in sub.itertuples()],
                color=_prot_colors_map2.get(prot, C["gray_lt"]),
                opacity=0.82,
                line=dict(width=1, color="white"),
            ),
            text=[_fmt_hover(r) for r in sub.itertuples()],
            hovertemplate="%{text}<extra></extra>",
        ))
    # Quadrant lines at medians
    _med_vol = df_view["l1_vol"].median()
    fig.add_vline(x=_med_vol, line_dash="dash", line_color=C["gray_lt"], line_width=1)
    fig.add_hline(y=0,         line_dash="dash", line_color=C["gray_lt"], line_width=1)
    fig.update_layout(
        **BASE,
        xaxis=dict(**GX,title="L1 vol share (volumesharelast)",title_font_size=11,tickformat=".1%"),
        yaxis=dict(**GY,title="L1 RAR (from AOR file)",title_font_size=11,tickformat="+.3f",zeroline=False),
        legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                    orientation="h",y=1.06,x=0),
        height=420, margin=dict(l=8,r=8,t=36,b=8), hovermode="closest",
    )
    cc("Projected performance — vol share × RAR",
       "Bubble size = RIS · x = L1 vol share (menu CSV) · y = L1 RAR (AOR file) · RAR=0 for recipes with no history")
    chart(fig, "vol_rar_scatter")

    ca2, cb2 = st.columns(2)

    with ca2:
        # Cost × RIS scatter
        # x = cost_2p (from menu CSV) · y = ris (from menu CSV)
        fig = go.Figure()
        for prot in df_view[~df_view["is_ineligible"]]["protein"].dropna().unique():
            sub = df_view[df_view["protein"] == prot].dropna(subset=["cost_2p","ris"])
            if len(sub) == 0: continue
            fig.add_trace(go.Scatter(
                x=sub["cost_2p"], y=sub["ris"],
                mode="markers", name=prot,
                marker=dict(size=9, color=_prot_colors_map2.get(prot,C["gray_lt"]),
                            opacity=0.85, line=dict(width=1,color="white")),
                text=[f"<b>{r.name}</b><br>Cost: ${_fmt(r.cost_2p,'.2f')}<br>"
                      f"RIS: {_fmt(r.ris,'.3f')}<br>RAR: {_fmt(r.l1_rar,'+.3f')}"
                      for r in sub.itertuples()],
                hovertemplate="%{text}<extra></extra>",
            ))
        fig.add_vline(x=df_view["cost_2p"].mean(), line_dash="dash",
                      line_color=C["gray_lt"], line_width=1,
                      annotation_text="avg cost",annotation_font_size=9)
        fig.add_hline(y=df_view["ris"].mean(),    line_dash="dash",
                      line_color=C["gray_lt"], line_width=1,
                      annotation_text="avg RIS",annotation_font_size=9)
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,title="2P cost (menu CSV)",title_font_size=11),
            yaxis=dict(**GY,title="RIS (menu CSV)",title_font_size=11),
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.06,x=0),
            height=300, margin=ML, hovermode="closest",
        )
        cc("Cost efficiency — 2P cost × RIS",
           "Top-left = low cost + high RIS (efficient) · Source: menu CSV")
        chart(fig, "cost_ris")

    with cb2:
        # 2W AOR vs expected — top 10 by vol share with AOR data
        # Source: AOR_CSV joined to menu CSV
        _aor_df = df_view[df_view["l1_2w_aor"].notna() & df_view["l1_2w_aor_expected"].notna()]
        if len(_aor_df) == 0:
            st.info("⚠ No L1 2W AOR data for this menu.")
        else:
            _aor_top = _aor_df.nlargest(10, "l1_vol")
            _names   = [n[:18]+"…" if len(n)>18 else n for n in _aor_top["name"]]
            _actual  = _aor_top["l1_2w_aor"].tolist()
            _exp     = _aor_top["l1_2w_aor_expected"].tolist()
            _colors  = [C["teal"] if a>=e else C["coral"] for a,e in zip(_actual,_exp)]
            _xrange  = [min(_actual+_exp)*0.97, max(_actual+_exp)*1.03]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=_names, x=_actual, orientation="h",
                name="L1 2W AOR", marker_color=_colors, marker_cornerradius=3,
                hovertemplate="<b>%{y}</b><br>Actual: %{x:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                y=_names, x=_exp, mode="markers", name="Expected",
                marker=dict(symbol="line-ew",size=12,color=C["ink"],
                            line=dict(width=2,color=C["ink"])),
                hovertemplate="<b>%{y}</b><br>Expected: %{x:.3f}<extra></extra>",
            ))
            fig.update_layout(
                **BASE,
                xaxis=dict(**GX,title="2W AOR",title_font_size=11,range=_xrange),
                yaxis=dict(**GY,autorange="reversed"),
                legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                            orientation="h",y=1.06,x=0),
                height=300, margin=dict(l=8,r=8,t=36,b=8),
            )
            cc(f"2W AOR vs expected — top {len(_aor_top)} by vol share",
               f"Green = above expectation · Source: AOR file · {len(df_view)-len(_aor_df)} recipes have no history")
            chart(fig, "aor_bars")

    # Vol share vs SIR proxy
    # Vol share: l1_vol (volumesharelast) · SIR proxy: avg_swap_ratio
    # Both from menu CSV — true SIR (mealchoice/total_mealchoice) not available
    _top12 = df_view.nlargest(12,"l1_vol")[~df_view.nlargest(12,"l1_vol")["is_ineligible"]]
    _n12   = [n[:16]+"…" if len(n)>16 else n for n in _top12["name"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=_n12, y=_top12["l1_vol"].tolist(), name="Vol share (L1)",
        marker_color=C["blue_lt"], marker_cornerradius=3,
        hovertemplate="<b>%{x}</b><br>Vol share: %{y:.2%}<extra></extra>",
    ))
    if _top12["l1_sir"].notna().any():
        fig.add_trace(go.Bar(
            x=_n12, y=_top12["l1_sir"].fillna(0).tolist(), name="SIR proxy (avg_swap_ratio)",
            marker_color=C["blue"], marker_cornerradius=3,
            hovertemplate="<b>%{x}</b><br>SIR proxy: %{y:.2%}<extra></extra>",
        ))
    fig.update_layout(
        **BASE, barmode="group",
        xaxis=dict(**GX,tickangle=-35),
        yaxis=dict(**GY,title="Share",title_font_size=11,tickformat=".1%"),
        legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                    orientation="h",y=1.06,x=0),
        height=280, margin=dict(l=8,r=8,t=36,b=80),
    )
    _sir_note = ("⚠ True SIR (mealchoice / total_mealchoice) not available — "
                 "avg_swap_ratio used as proxy · Source: menu CSV")
    cc("Vol share vs SIR proxy — top 12 by vol share", _sir_note)
    chart(fig, "sir_vol")

    # Quality diagnostic table — flagged recipes
    # Flags: l1_score < threshold, l3_avg_share_1star > 12%, l1_rar < -0.01, skew_risk ≥ 3
    _qdf = df_view[
        (df_view["l1_score"] < score_threshold) |
        (df_view["l3_avg_share_1star"].fillna(0) > 0.12) |
        (df_view["l1_rar"].fillna(0) < -0.01) |
        (df_view["skew_risk"] >= 3)
    ].copy()
    _qdf = _qdf.merge(pool_df[["code","prot_group"]].rename(columns={"code":"recipe_code_main"}),
                      on="recipe_code_main", how="left")

    def _skew_pill(v):
        if pd.isna(v): return pill("N/A","gray")
        v = int(v)
        return pill(f"Risk {v}", "green" if v<=2 else "amber" if v==3 else "red")

    _qrows = ""
    for r in _qdf.sort_values("l3_avg_share_1star", ascending=False).itertuples():
        _sc  = (_fmt(r.score_delta,"+.2f") if pd.notna(r.score_delta) else "N/A")
        _scol= "color:#E84040" if pd.notna(r.score_delta) and r.score_delta<0 else "color:#27AE60"
        _qrows += (f"<tr>"
                   f"<td><strong>{r.name}</strong></td>"
                   f"<td>{r.protein}</td>"
                   f"<td>{_pill_score(r.l1_score)} <span style='font-size:10px;{_scol}'>{_sc} vs LT</span></td>"
                   f"<td>{_pill_star(r.l3_avg_share_1star)}</td>"
                   f"<td>{_pill_rar(r.l1_rar)}</td>"
                   f"<td>{_pill_ris(r.ris)}</td>"
                   f"<td>{_skew_pill(r.skew_risk)}</td>"
                   f"</tr>")
    if not _qrows:
        _qrows = "<tr><td colspan='7' style='color:#888;text-align:center;padding:16px'>No flagged recipes</td></tr>"
    st.markdown(f"""<div class='cc'>
      <div class='cc-title'>Quality diagnostic — flagged recipes</div>
      <div class='cc-sub'>Below {score_threshold:.1f} score · 1-star share &gt;12% · negative RAR · or skew risk ≥3
        · Source: score/RIS from menu CSV · RAR/1-star from AOR+LT files</div>
      <table class='lt'><thead><tr>
        <th>Recipe</th><th>Protein</th><th>L1 score (vs LT)</th>
        <th>L3 1-star</th><th>L1 RAR</th><th>RIS</th><th>Skew</th>
      </tr></thead><tbody>{_qrows}</tbody></table>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 3 — TRENDS
# Source: TREND_CSV (W06–W17) · PT_CSV (protein demand W09–W17) · NR_CSV
# ═════════════════════════════════════════════════════════════════════
with t3:
    _t3 = []
    if len(trend_df) >= 4:
        _re = trend_df["avg_ris"].iloc[:4].mean()
        _rl = trend_df["avg_ris"].iloc[-4:].mean()
        _t3.append(("green" if _rl>=_re else "amber",
            f"RIS: 4w avg {_rl:.3f} vs earlier {_re:.3f} ({'▲ improving' if _rl>=_re else '▼ declining'})"))
    _t3.append(("amber","⚠ n_new_recipes all zero — query returned no results, pending fix"))
    flags_box(_t3 if _t3 else [("green","No trend flags")])

    kpis(
        ("RIS 4w avg",    f"{trend_ris_4w:.3f}",  "W14–W17 actual", ""),
        ("Score 4w avg",  f"{trend_score_4w:.3f}", "w/o SCM",        ""),
        ("2W AOR 4w avg", f"{trend_aor_4w:.3f}",  "menu-level",     ""),
        ("Total SKUs",    str(total_unique_skus),  f"W{hf_week[-2:]} planned",""),
    )

    # Multi-metric trend line
    # avg_ris, avg_score_wo_scm, avg_2w_aor from TREND_CSV
    _wk = trend_df["wk"].tolist()
    fig = make_subplots(specs=[[{"secondary_y":True}]])
    fig.add_trace(go.Scatter(
        x=_wk, y=trend_df["avg_ris"].tolist(), name="Avg RIS",
        line=dict(color=C["blue"],width=2.5), mode="lines+markers", marker_size=5,
        hovertemplate="<b>%{x}</b><br>RIS: %{y:.4f}<extra></extra>",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=_wk, y=trend_df["avg_score_wo_scm"].tolist(), name="Avg score w/o SCM",
        line=dict(color=C["teal"],width=2,dash="dot"), mode="lines+markers", marker_size=4,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<extra></extra>",
    ), secondary_y=True)
    fig.add_trace(go.Scatter(
        x=_wk, y=trend_df["avg_2w_aor"].tolist(), name="Avg 2W AOR",
        line=dict(color=C["amber"],width=2,dash="dash"), mode="lines+markers", marker_size=4,
        hovertemplate="<b>%{x}</b><br>2W AOR: %{y:.3f}<extra></extra>",
    ), secondary_y=True)
    # Mark current planned week if in range
    _wk_label = f"W{hf_week[-2:]}"
    if _wk_label in _wk:
        fig.add_trace(go.Scatter(
            x=[_wk_label,_wk_label], y=[trend_df["avg_ris"].min()*0.98, trend_df["avg_ris"].max()*1.02],
            mode="lines", line=dict(color=C["coral"],width=1.5),
            showlegend=False, hoverinfo="skip",
        ), secondary_y=False)
    fig.update_layout(
        **BASE,
        xaxis=dict(**GX),
        yaxis=dict(**GY,title="RIS",title_font_size=11),
        yaxis2=dict(tickfont_size=11,title="Score / AOR",title_font_size=11,
                    overlaying="y",side="right"),
        legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                    orientation="h",y=1.08,x=0),
        hovermode="x unified", margin=ML, height=280,
    )
    cc("Key metrics trend — W06–W17 actual data",
       "Source: TREND_CSV · avg_ris, avg_score_wo_scm, avg_2w_aor (menu-level aggregates)")
    chart(fig, "trends_line")

    ca3, cb3 = st.columns(2)

    with ca3:
        # New vs repeat — from NR_CSV
        # Note: n_new_recipes all zero — known issue
        _nwk  = trend_df["wk"].tolist()
        _new  = trend_df["n_new_recipes"].fillna(0).astype(int).tolist()
        _rep  = trend_df["n_repeat_recipes"].fillna(0).astype(int).tolist()
        fig   = go.Figure()
        fig.add_trace(go.Bar(x=_nwk,y=_new,name="New",
            marker_color=C["blue"],marker_cornerradius=3,width=0.55))
        fig.add_trace(go.Bar(x=_nwk,y=_rep,name="Repeat",
            marker_color=C["blue_lt"],marker_cornerradius=3,width=0.55))
        fig.update_layout(
            **BASE, barmode="stack",
            xaxis=dict(**GX),
            yaxis=dict(**GY,title="Recipe count",title_font_size=11),
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.08,x=0),
            margin=ML, height=240,
        )
        cc("New vs repeat recipes — W06–W17",
           "⚠ n_new_recipes = 0 for all weeks (query issue) · Source: NR_CSV")
        chart(fig, "new_repeat")

    with cb3:
        # Protein demand trend — customer vol share by protein group W09–W17
        # Source: PT_CSV (from facts_recipes_ordered_enriched)
        _pt_cols = {"Chicken":C["blue"],"Beef":C["coral"],"Pork":C["teal"],
                    "Fish":C["amber"],"Veggie":C["gray"],"Turkey":C["purple"],
                    "Shellfish":C["pink"],"Plantbased":C["teal_lt"]}
        _pt_weeks = sorted(pt_grp["wk"].unique())
        fig = go.Figure()
        for prot in pt_grp["protein_group"].unique():
            sub = pt_grp[pt_grp["protein_group"]==prot].set_index("wk")["pct"]
            _vals = [round(sub.get(w,0),2) for w in _pt_weeks]
            if max(_vals) < 0.1: continue  # skip negligible
            fig.add_trace(go.Scatter(
                x=_pt_weeks, y=_vals, name=prot,
                line=dict(color=_pt_cols.get(prot,C["gray"]),width=2),
                mode="lines+markers", marker_size=4,
                hovertemplate=f"<b>{prot}</b> %{{y:.1f}}%<extra></extra>",
            ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX),
            yaxis=dict(**GY,title="% customer demand",title_font_size=11),
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.08,x=0),
            margin=ML, height=240,
        )
        cc("Protein demand trend — W09–W17 (customer vol share)",
           "Source: PT_CSV · grouped from primaryprotein · customer selection share")
        chart(fig, "prot_trend")

    # Quality degradation table
    # Recipes where L1 score or 1-star share is worse than LT average
    _qd = df_view[
        (df_view["score_delta"].notna() & (df_view["score_delta"] < -0.1)) |
        (df_view["l3_avg_share_1star"].notna() &
         df_view["lt_avg_share_1star"].notna() &
         ((df_view["l3_avg_share_1star"] - df_view["lt_avg_share_1star"]) > 0.03))
    ].sort_values("l1_vol", ascending=False)

    _drows = ""
    for r in _qd.itertuples():
        _1s_delta = (r.l3_avg_share_1star - r.lt_avg_share_1star
                     if pd.notna(r.l3_avg_share_1star) and pd.notna(r.lt_avg_share_1star) else None)
        _sc = (f"<span style='color:#E84040'>▼{abs(r.score_delta):.2f}</span>"
               if pd.notna(r.score_delta) and r.score_delta<0 else "stable")
        _st = (f"<span style='color:#E84040'>▲{abs(_1s_delta):.0%}</span>"
               if _1s_delta and _1s_delta>0 else "stable")
        _drows += (f"<tr>"
                   f"<td><strong>{r.name}</strong></td>"
                   f"<td>{r.protein}</td>"
                   f"<td>{_fmt(r.l1_vol,'.1%')}</td>"
                   f"<td>{_fmt(r.l1_score,'.2f')} {_sc}</td>"
                   f"<td>{_fmt(r.l3_avg_share_1star,'.0%')} {_st}</td>"
                   f"<td>{_fmt(r.lt_avg_score_wo_scm,'.2f')}</td>"
                   f"</tr>")
    if not _drows:
        _drows = "<tr><td colspan='6' style='color:#888;text-align:center;padding:16px'>No quality degradation detected</td></tr>"
    st.markdown(f"""<div class='cc'>
      <div class='cc-title'>Quality degradation — recipes declining vs LT average</div>
      <div class='cc-sub'>L1 score or L3 1-star share materially worse than lifetime avg · sorted by vol share
        · Source: LT_CSV + menu CSV</div>
      <table class='lt'><thead><tr>
        <th>Recipe</th><th>Protein</th><th>L1 vol</th>
        <th>L1 score (vs LT)</th><th>L3 1-star (vs LT)</th><th>LT score</th>
      </tr></thead><tbody>{_drows}</tbody></table>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 4 — SUPPLY VS DEMAND
# Supply: recipe count % from menu CSV
# Demand: 52w avg customer vol share from DEMAND_CSV
# ═════════════════════════════════════════════════════════════════════
with t4:
    _t4 = []
    for tbl, attr in [(prot_tbl,"protein"),(cuis_tbl,"cuisine")]:
        for _, row in tbl.dropna(subset=["gap_pp"]).sort_values("gap_pp",key=abs,ascending=False).head(2).iterrows():
            if abs(row["gap_pp"]) < 2: continue
            _d = "over-supplied" if row["gap_pp"]>0 else "under-supplied"
            _kind = "red" if abs(row["gap_pp"])>8 else "amber"
            _t4.append((_kind, f"{row[attr]}: {row['supply_pct']:.1f}% menu vs {row['demand_pct']:.1f}% demand — {_d}"))
    flags_box(_t4 if _t4 else [("green","No major supply vs demand gaps")])

    ca4, cb4 = st.columns(2)

    with ca4:
        # Cuisine gap bar — height = abs(supply-demand), colour = direction
        _ct = cuis_tbl.dropna(subset=["gap_pp"]).sort_values("gap_pp",key=abs,ascending=False).head(12)
        _gc = ["#185FA5" if g>6 else "#85B7EB" if g>2 else "#A32D2D" if g<-6 else "#E24B4A" if g<-2 else "#27AE60"
               for g in _ct["gap_pp"]]
        fig = go.Figure(go.Bar(
            x=_ct["cuisine"], y=_ct["gap_pp"].abs(),
            marker_color=_gc, marker_cornerradius=4,
            customdata=_ct[["cuisine","supply_pct","demand_pct","gap_pp"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>Supply: %{customdata[1]:.1f}%<br>"
                          "Demand: %{customdata[2]:.1f}%<br>Gap: %{customdata[3]:+.1f}pp<extra></extra>",
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,tickangle=-30),
            yaxis=dict(**GY,title="Gap (pp)",title_font_size=11),
            margin=dict(l=8,r=8,t=8,b=60), height=280,
        )
        cc("Cuisine — supply vs demand gap",
           "Blue=surplus · Red=gap · Supply from menu CSV · Demand from DEMAND_CSV (52w avg)")
        chart(fig, "cuis_gap")

    with cb4:
        # Protein supply vs demand bars + marker
        _pt2 = prot_tbl.dropna(subset=["supply_pct"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_pt2["protein"], y=_pt2["supply_pct"], name="Menu supply %",
            marker_color=C["blue"], marker_cornerradius=4,
            hovertemplate="<b>%{x}</b><br>Supply: %{y:.1f}%<extra></extra>",
        ))
        _pt2d = _pt2.dropna(subset=["demand_pct"])
        if len(_pt2d) > 0:
            fig.add_trace(go.Scatter(
                x=_pt2d["protein"], y=_pt2d["demand_pct"],
                name="Demand signal (52w avg)", mode="markers",
                marker=dict(symbol="line-ew",size=14,color=C["ink"],line=dict(width=2.5,color=C["ink"])),
                hovertemplate="<b>%{x}</b><br>Demand: %{y:.1f}%<extra></extra>",
            ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX),
            yaxis=dict(**GY,title="%",title_font_size=11),
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.08,x=0),
            margin=ML, height=260,
        )
        cc("Protein — supply vs demand",
           "Bar=menu supply (menu CSV) · Marker=52w customer avg (DEMAND_CSV)")
        chart(fig, "prot_svd")

    # Full SVD table
    def _spill(s):
        return {"Surplus":pill("Surplus","amber"),"Gap":pill("Gap","red"),
                "On target":pill("On target","green")}.get(s, pill(s,"gray"))
    _svd_rows = ""
    for tbl, attr_name, attr_col in [(prot_tbl,"Protein","protein"),
                                      (cuis_tbl,"Cuisine","cuisine"),
                                      (dish_tbl,"Dish type","dish_type")]:
        for _, row in tbl.iterrows():
            _sup = f"{row['supply_pct']:.1f}%" if pd.notna(row['supply_pct']) else "N/A"
            _dem = f"{row['demand_pct']:.1f}%" if pd.notna(row.get('demand_pct')) else "⚠ no data"
            _gap = f"{row['gap_pp']:+.1f}pp"    if pd.notna(row.get('gap_pp'))    else "N/A"
            _svd_rows += (f"<tr><td><strong>{attr_name}</strong></td>"
                          f"<td>{row[attr_col]}</td><td>{_sup}</td>"
                          f"<td>{_dem}</td><td style='font-weight:500'>{_gap}</td>"
                          f"<td>{_spill(row['status'])}</td></tr>")
    st.markdown(f"""<div class='cc'>
      <div class='cc-title'>Supply vs demand — full attribute breakdown</div>
      <div class='cc-sub'>Menu supply (menu CSV) vs 52w customer demand (DEMAND_CSV, W33 2025–W17 2026)</div>
      <table class='lt'><thead><tr>
        <th>Attribute</th><th>Value</th><th>Supply %</th>
        <th>Demand %</th><th>Gap</th><th>Status</th>
      </tr></thead><tbody>{_svd_rows}</tbody></table>
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# TAB 5 — POOL HEALTH
# Source: POOL_CSV (recipe_pool_master_enriched) + menu CSV
# ═════════════════════════════════════════════════════════════════════
with t5:
    _n_below = len(df_view[df_view["ris"].notna() & (df_view["ris"] < pool_median_ris)])
    flags_box([
        ("amber" if _n_below > 5 else "green",
         f"{_n_below} planned recipes below pool median RIS ({pool_median_ris:.3f})"),
        ("amber" if pool_high_ris_unplan > 20 else "green",
         f"{pool_high_ris_unplan} unplanned pool recipes with RIS > 0.75"),
        ("down" if n_data_issues > 0 else "green",
         f"{n_data_issues} planned recipes have missing/Gap data fields"),
        ("green",
         f"Pool: {pool_total:,} total · {pool_ready:,} HF ready for planning"),
    ])

    kpis(
        ("Total pool",         f"{pool_total:,}",         "all brands",                    ""),
        ("HF ready",           f"{pool_ready:,}",         f"{pool_ready/pool_hf_count*100:.0f}% of HF", ""),
        ("Planned this week",  str(n_recipes),             f"of {pool_ready:,} ready",      ""),
        ("Unplanned (ready)",  f"{pool_ready-n_recipes:,}","not selected this cycle",       ""),
        ("High RIS unplanned", str(pool_high_ris_unplan),  "RIS>0.75 not planned",          "warn" if pool_high_ris_unplan>20 else ""),
        ("Pool avg RIS",       f"{pool_avg_ris:.3f}",      f"planned avg {proj_ris:.3f}",   ""),
        ("Data issues",        str(n_data_issues),         "missing fields",                "down" if n_data_issues>0 else "up"),
    )

    ca5, cb5 = st.columns(2)

    with ca5:
        # RIS by protein — pool avg vs planned menu avg
        # Source: pool_df["2w_RIS"] · df_view["ris"]
        if ris_prot_shared:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=ris_prot_shared,
                y=[pool_ris_prot[p] for p in ris_prot_shared],
                name="Pool avg RIS", marker_color=C["blue_lt"], marker_cornerradius=4,
                hovertemplate="<b>%{x}</b><br>Pool avg: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Bar(
                x=ris_prot_shared,
                y=[menu_ris_prot[p] for p in ris_prot_shared],
                name="Planned avg RIS", marker_color=C["blue"], marker_cornerradius=4,
                hovertemplate="<b>%{x}</b><br>Planned avg: %{y:.3f}<extra></extra>",
            ))
            fig.update_layout(
                **BASE, barmode="group",
                xaxis=dict(**GX),
                yaxis=dict(**GY,title="Avg RIS",title_font_size=11),
                legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                            orientation="h",y=1.08,x=0),
                margin=ML, height=260,
            )
        else:
            fig = go.Figure()
        cc("RIS by protein — pool vs planned",
           "Source: POOL_CSV (2w_RIS) · menu CSV (retention_performance_normalized)")
        chart(fig, "ris_prot")

    with cb5:
        # Top unplanned recipes by RIS
        # Source: unplanned_df from POOL_CSV
        _top_u = unplanned_df.dropna(subset=["2w_RIS"]).nlargest(12,"2w_RIS")
        _ris_u = _top_u["2w_RIS"].tolist()
        _name_u= [t[:22]+"…" if len(t)>22 else t for t in _top_u["title"]]
        fig = go.Figure(go.Bar(
            y=_name_u, x=_ris_u, orientation="h",
            marker_color=[C["teal"] if r>0.75 else C["blue"] if r>0.65 else C["gray"]
                          for r in _ris_u],
            marker_cornerradius=3,
            hovertemplate="<b>%{y}</b><br>RIS: %{x:.3f}<extra></extra>",
        ))
        _xmin = min(_ris_u)*0.97 if _ris_u else 0.4
        _xmax = max(_ris_u)*1.02 if _ris_u else 1.0
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,range=[_xmin,_xmax],title="RIS",title_font_size=11),
            yaxis=dict(**GY,autorange="reversed"),
            margin=M, height=300,
        )
        cc(f"Top {len(_top_u)} unplanned recipes by RIS",
           "Source: POOL_CSV · green = RIS > 0.75")
        chart(fig, "unplanned_ris")

    # Pool status breakdown
    _status_counts = pool_df["status"].value_counts().reset_index()
    _status_counts.columns = ["status","count"]
    fig = go.Figure(go.Bar(
        x=_status_counts["status"],
        y=_status_counts["count"],
        marker_color=[C["teal"] if s=="READY FOR MENU PLANNING"
                      else C["blue"] if s=="IN DEVELOPMENT"
                      else C["amber"] if s=="ON HOLD"
                      else C["coral"] for s in _status_counts["status"]],
        marker_cornerradius=4,
        hovertemplate="<b>%{x}</b>: %{y}<extra></extra>",
    ))
    fig.update_layout(
        **BASE, xaxis=dict(**GX,tickangle=-15),
        yaxis=dict(**GY,title="Recipe count",title_font_size=11),
        margin=M, height=220,
    )
    cc("Pool status breakdown (HelloFresh)",
       "Source: POOL_CSV · status col from recipe_pool_master_enriched")
    chart(fig, "pool_status")

    # Data issues table — planned recipes with missing/Gap fields
    _di_df = df_view[df_view["data_issues"]][
        ["name","protein","cuisine","dishtype","missing_field","ris","l1_score"]].copy()
    if len(_di_df) > 0:
        _di_rows = ""
        for r in _di_df.itertuples():
            _di_rows += (f"<tr>"
                         f"<td><strong>{r.name}</strong></td>"
                         f"<td>{r.protein}</td><td>{r.cuisine}</td>"
                         f"<td>{pill(r.missing_field,'red')}</td>"
                         f"<td>{_pill_ris(r.ris)}</td>"
                         f"<td>{_fmt(r.l1_score,'.2f')}</td>"
                         f"</tr>")
        st.markdown(f"""<div class='cc'>
          <div class='cc-title'>Data issues — planned recipes with missing/Gap fields</div>
          <div class='cc-sub'>{len(_di_df)} recipes have incomplete data — metrics unreliable · Source: menu CSV</div>
          <table class='lt'><thead><tr>
            <th>Recipe</th><th>Protein</th><th>Cuisine</th>
            <th>Missing field</th><th>RIS</th><th>L1 score</th>
          </tr></thead><tbody>{_di_rows}</tbody></table>
        </div>""", unsafe_allow_html=True)

    # New recipe debut scatter — first outing vs pool median
    _new_df = df_view[df_view["is_new"]].copy()
    if len(_new_df) > 0:
        fig = go.Figure()
        _pool_med_score = pool_df["scorescm"].median()
        _pool_med_vol   = pool_df["volumesharelast"].median()
        fig.add_hline(y=_pool_med_score if pd.notna(_pool_med_score) else 0,
                      line_dash="dash",line_color=C["blue"],line_width=1,
                      annotation_text=f"Pool median score {_pool_med_score:.2f}" if pd.notna(_pool_med_score) else "",
                      annotation_font_size=9)
        fig.add_vline(x=_pool_med_vol if pd.notna(_pool_med_vol) else 0,
                      line_dash="dash",line_color=C["teal"],line_width=1,
                      annotation_text=f"Pool median vol {_pool_med_vol:.3f}" if pd.notna(_pool_med_vol) else "",
                      annotation_font_size=9)
        for r in _new_df.itertuples():
            if pd.isna(r.l1_vol): continue
            fig.add_trace(go.Scatter(
                x=[r.l1_vol], y=[r.l1_score] if pd.notna(r.l1_score) else [None],
                mode="markers+text",
                text=[r.name[:15]], textposition="top center", textfont=dict(size=9),
                marker=dict(size=12,color=C["purple"],opacity=0.85,
                            line=dict(width=1,color="white")),
                name=r.name, showlegend=False,
                hovertemplate=(f"<b>{r.name}</b><br>"
                               f"L1 vol: {_fmt(r.l1_vol,'.1%')}<br>"
                               f"L1 score: {_fmt(r.l1_score,'.2f')}<extra></extra>"),
            ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,title="L1 vol share",title_font_size=11,tickformat=".1%"),
            yaxis=dict(**GY,title="L1 score",title_font_size=11),
            margin=ML, height=250,
        )
        cc(f"New recipe first outing vs pool median — {len(_new_df)} new recipes",
           "Source: menu CSV (is_new flag) + pool CSV medians")
        chart(fig, "new_debut")


# ═════════════════════════════════════════════════════════════════════
# TAB 6 — INGREDIENTS & SKUs
# Source: skuname col in menu CSV (pipe-delimited)
# ═════════════════════════════════════════════════════════════════════
with t6:
    _avg_per = round(sum(sku_cnt)/total_unique_skus,1) if total_unique_skus>0 else 0
    _sgl_pct = round(single_use_skus/total_unique_skus*100,0) if total_unique_skus>0 else 0
    flags_box([
        ("amber" if single_use_skus>50 else "green",
         f"{single_use_skus} single-use SKUs ({_sgl_pct:.0f}%) — procured for 1 recipe only"),
        ("amber", f"Most used: '{most_used_sku[:40]}' in {most_used_count} recipes"),
        ("amber", "⚠ Weekly SKU trend not available — no weekly SKU history in data"),
        ("amber", "⚠ Ingredient overlap matrix not available — requires preference×SKU mapping"),
    ])
    kpis(
        ("Unique SKUs",       str(total_unique_skus),  f"{hf_week} planned menu",          ""),
        ("Avg SKUs/recipe",   f"{_avg_per:.1f}",        "utilization",                       ""),
        ("Single-use",        str(single_use_skus),     f"{_sgl_pct:.0f}% — cost risk",     "warn" if single_use_skus>50 else ""),
        ("Most used SKU",     str(most_used_count),     f"{most_used_sku[:25]}…",            ""),
        ("Net new SKUs",      "N/A",                    "⚠ no weekly history",               "warn"),
        ("SKU adoption",      "N/A",                    "⚠ no weekly history",               "warn"),
    )

    ca6, cb6 = st.columns(2)

    with ca6:
        # SKU weekly trend — no data
        st.markdown(
            "<div class='cc'><div class='cc-title'>SKU count over time</div>"
            "<div class='cc-sub'>⚠ No weekly SKU history available — "
            "requires menu_recipe_csku_ingredient_picklist query</div></div>",
            unsafe_allow_html=True,
        )
        st.info("SKU weekly trend not yet available.")

    with cb6:
        # SKU category breakdown from real parsed SKU names
        _cat_counts = Counter(sku_cat)
        _cats = list(_cat_counts.keys())
        _ccnt = [_cat_counts[c] for c in _cats]
        _cat_color_map = {
            "Protein":C["coral"],"Produce":C["teal"],"Starch":C["amber"],
            "Dairy":C["purple"],"Oil/Fat":C["amber"],"Herb/Spice":C["teal_lt"],
            "Sauce":C["pink"],"Pantry":C["gray"],"Bakery":C["gray_lt"],
        }
        fig = go.Figure(go.Bar(
            y=_cats, x=_ccnt, orientation="h",
            marker_color=[_cat_color_map.get(c,C["gray"]) for c in _cats],
            marker_cornerradius=3,
            hovertemplate="<b>%{y}</b>: %{x} SKUs<extra></extra>",
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,title="Unique SKU count",title_font_size=11),
            yaxis=dict(**GY,autorange="reversed"),
            margin=M, height=260,
        )
        cc("SKU category breakdown",
           "Source: skuname col from menu CSV (parsed + categorised)")
        chart(fig, "sku_cat")

    cc7, cd7 = st.columns(2)

    with cc7:
        # Top SKUs by recipe count
        fig = go.Figure(go.Bar(
            y=[n[:38]+"…" if len(n)>38 else n for n in sku_names[:15]],
            x=sku_cnt[:15], orientation="h",
            marker_color=[_cat_color_map.get(c,C["gray"]) for c in sku_cat[:15]],
            marker_cornerradius=3,
            hovertemplate="<b>%{y}</b>: %{x} recipes<extra></extra>",
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX,title="# recipes",title_font_size=11),
            yaxis=dict(**GY,autorange="reversed"),
            margin=M, height=340,
        )
        cc("Top 15 SKUs by recipe count",
           "Source: skuname col from menu CSV · coloured by category")
        chart(fig, "top_skus")

    with cd7:
        # SKU usage frequency distribution
        fig = go.Figure(go.Bar(
            x=sku_freq_labels, y=sku_freq_counts,
            marker_color=[C["coral"],C["amber"],C["blue_lt"],C["blue"],C["teal"],C["purple"]],
            marker_cornerradius=4,
            hovertemplate="<b>%{x}</b>: %{y} SKUs<extra></extra>",
        ))
        fig.update_layout(
            **BASE,
            xaxis=dict(**GX),
            yaxis=dict(**GY,title="# SKUs",title_font_size=11),
            margin=M, height=220,
        )
        cc("SKU usage frequency",
           "Source: skuname col from menu CSV · single-use = cost risk")
        chart(fig, "sku_freq")

        # Overlap placeholder
        st.markdown(
            "<div class='cc' style='margin-top:14px'>"
            "<div class='cc-title'>Ingredient overlap by recipe type</div>"
            "<div class='cc-sub'>⚠ Not available — requires SKU × preference mapping query</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════
# TAB 7 — MENU COMPARISON
# Source: combined menu CSV — all weeks × versions
# Weighted metrics pre-computed in CSV; simple means for others
# ═════════════════════════════════════════════════════════════════════
with t7:
    st.markdown(
        "<div class='cc' style='margin-bottom:0'>"
        "<div class='cc-sub'>Compare any week × version combination. "
        "Weighted metrics (RIS, score, cost) from pre-computed cols. "
        "RAR/AOR not available in menu CSV — shown as N/A.</div></div>",
        unsafe_allow_html=True,
    )

    n_cmp = st.selectbox("Weeks to compare", [2, 3, 4], index=0)

    _cmp_cols = st.columns(n_cmp)
    cmp_weeks    = []
    cmp_versions = []
    for i, col in enumerate(_cmp_cols):
        with col:
            _w = st.selectbox(f"Week {i+1}", _avail_weeks, index=min(i, len(_avail_weeks)-1),
                              key=f"cw_{i}")
            _v = st.selectbox(f"Version {i+1}", _avail_vers, index=0, key=f"cv_{i}")
            cmp_weeks.append(_w)
            cmp_versions.append(_v)

    # Slot preset filter for comparison
    _sl1, _sl2 = st.columns([1,2])
    with _sl1:
        _slot_preset = st.selectbox("Slot preset", ["None (all slots)","Core only","Surcharge only"])
    with _sl2:
        _slot_custom = st.text_input("Custom slot numbers (comma-separated)", placeholder="1001,1002,1003")

    st.markdown("<hr>", unsafe_allow_html=True)

    def _cmp_kpis(week, version, slot_preset=None, slot_custom=None):
        """
        Compute comparison KPIs for a given week × version.
        Weighted metrics: sum(pre-computed weighted col) — weights already sum to 1.
        Simple means: for metrics without a weight (uptake, SIR, skew count).
        Not available: RAR, 5w AOR — not in menu CSV.
        """
        sub = _all_menus[(_all_menus["hellofreshweek"]==week) &
                         (_all_menus["gampversion"]==version)].copy()
        if len(sub) == 0: return None

        # Apply slot filters
        if slot_custom:
            _slots = [s.strip() for s in slot_custom.split(",") if s.strip()]
            if _slots:
                sub = sub[sub["slotnumber"].astype(str).isin(_slots)]
        elif slot_preset == "Core only":
            sub = sub[sub["preference"].str.contains("Classic|Quick|Family",na=False,case=False)]
        elif slot_preset == "Surcharge only":
            sub = sub[sub["preference"].str.contains("Surcharge",na=False,case=False)]

        if len(sub) == 0: return None

        def _cmp_wmean(col, ndigits):
            if col not in sub.columns: return None
            mask = sub[col].notna()
            if not mask.any(): return None
            w = sub.loc[mask, "weight"]
            return round((sub.loc[mask, col] * w).sum() / w.sum(), ndigits)

        return {
            # Weighted — pre-computed in CSV (Gap rows included, weights already normalised)
            "Weighted RIS":           round(sub["weightedRIS"].sum(),         3),
            "Weighted score (w SCM)": round(sub["weightedScorescm"].sum(),    3),
            "Weighted score (wo SCM)":round(sub["weightedScorewoscm"].sum(),  3),
            "Weighted 2P cost":       round(sub["weightedCost2p"].sum(),      2),
            # Weighted means using `weight` column
            "Weighted uptake":        _cmp_wmean("uptake_ratio",  4),
            "Weighted SIR proxy":     _cmp_wmean("avg_swap_ratio",4),
            "# Recipes":              int(len(sub)),
            "# New recipes":          int(sub["isnewrecipe"].sum()) if "isnewrecipe" in sub.columns else None,
            "# Gap slots":            int((sub["mainprotein"]=="Gap").sum()),
            "# High skew (≥3)":       int((sub["seasonalityrisk_w1"]>=3).sum()) if "seasonalityrisk_w1" in sub.columns else None,
            # Not available in menu CSV
            "Avg RAR":                None,
            "Avg 5w AOR":             None,
        }

    # Build comparison data
    cmp_data = []
    for w, v in zip(cmp_weeks, cmp_versions):
        d = _cmp_kpis(w, v, _slot_preset, _slot_custom)
        cmp_data.append({"week":w,"version":v,"data":d})

    # ── KPI table ────────────────────────────────────────────────
    _metrics_order = [
        ("Weighted RIS",           "higher", ".3f"),
        ("Weighted score (wo SCM)","higher", ".3f"),
        ("Weighted score (w SCM)", "higher", ".3f"),
        ("Weighted 2P cost",       "lower",  ".2f"),
        ("Weighted uptake",        "higher", ".2%"),
        ("Weighted SIR proxy",     "higher", ".2%"),
        ("# Recipes",              "",       "d"),
        ("# New recipes",          "",       "d"),
        ("# Gap slots",            "lower",  "d"),
        ("# High skew (≥3)",       "lower",  "d"),
        ("Avg RAR",                "higher", "+.3f"),
        ("Avg 5w AOR",             "higher", ".3f"),
    ]

    _hdr = "<th>Metric</th>" + "".join(
        f"<th>{d['week']}<br><span style='font-weight:400;font-size:10px'>{d['version']}</span></th>"
        for d in cmp_data
    )
    _trows = ""
    for metric, direction, fmt in _metrics_order:
        _vals = [d["data"].get(metric) if d["data"] else None for d in cmp_data]
        # Skip if all N/A
        if all(v is None for v in _vals): continue
        # Find best
        _numeric = [v for v in _vals if v is not None]
        _best_val = (max(_numeric) if direction=="higher" else min(_numeric)) if len(_numeric)>1 else None

        _cells = f"<td style='color:#888;font-size:12px'>{metric}</td>"
        for v in _vals:
            if v is None:
                _cells += "<td style='color:#aaa;font-size:13px'>⚠ N/A</td>"
            else:
                try:
                    _str = (f"${v:{fmt}}" if metric.lower().endswith("cost") and fmt!=".2%" else
                            format(v, fmt))
                except:
                    _str = str(v)
                _is_best = (_best_val is not None and v == _best_val and len(_numeric) > 1)
                _style   = "font-weight:600;color:#1A1A18" if _is_best else "color:#2A2A28"
                _badge   = ("&nbsp;<span style='font-size:9px;background:#E8F7EF;color:#0F6B30;"
                             "padding:1px 5px;border-radius:8px'>best</span>") if _is_best else ""
                _cells  += f"<td style='font-size:13px;{_style}'>{_str}{_badge}</td>"
        _trows += f"<tr>{_cells}</tr>"

    st.markdown(f"""<div class='cc'>
      <div class='cc-title'>KPI comparison</div>
      <div class='cc-sub'>Weighted metrics = sum(weighted col) · RAR/AOR not in menu CSV · best value highlighted</div>
      <table class='lt'><thead><tr>{_hdr}</tr></thead>
      <tbody>{_trows}</tbody></table>
    </div>""", unsafe_allow_html=True)

    # Metric bar chart — select which metric to visualise
    _chartable = [(m,f) for m,d,f in _metrics_order
                  if any(cd["data"] and cd["data"].get(m) is not None for cd in cmp_data)]
    if _chartable:
        sel_m = st.selectbox("Chart metric", [m for m,_ in _chartable], index=0, key="cmp_chart_sel")
        sel_f = next(f for m,f in _chartable if m==sel_m)
        _cx   = [f"{d['week']}\n{d['version']}" for d in cmp_data if d["data"] and d["data"].get(sel_m) is not None]
        _cy   = [d["data"][sel_m] for d in cmp_data if d["data"] and d["data"].get(sel_m) is not None]
        if _cy:
            _best = max(_cy) if any(d=="higher" for m,d,_ in _metrics_order if m==sel_m) else min(_cy)
            fig = go.Figure(go.Bar(
                x=_cx, y=_cy,
                marker_color=[C["teal"] if v==_best else C["blue_lt"] for v in _cy],
                marker_cornerradius=5,
                text=[format(v,sel_f) for v in _cy],
                textposition="outside", textfont_size=11,
                hovertemplate="<b>%{x}</b><br>" + sel_m + ": %{y}<extra></extra>",
            ))
            fig.update_layout(
                **BASE,
                xaxis=dict(**GX),
                yaxis=dict(**GY,title=sel_m,title_font_size=11),
                margin=dict(l=8,r=8,t=36,b=8), height=260,
            )
            cc(f"{sel_m} — across selected weeks/versions","Teal = best value")
            chart(fig, "cmp_bar")

    # Protein mix comparison — real data
    fig = go.Figure()
    _any = False
    for d in cmp_data:
        _sub = _all_menus[
            (_all_menus["hellofreshweek"]==d["week"]) &
            (_all_menus["gampversion"]==d["version"]) &
            (_all_menus["mainprotein"]!="Gap")
        ]
        if len(_sub) == 0: continue
        _ps = _sub["mainprotein"].value_counts(normalize=True).mul(100).round(1)
        fig.add_trace(go.Bar(
            x=_ps.index.tolist(), y=_ps.values.tolist(),
            name=f"{d['week']} {d['version']}", marker_cornerradius=3,
            hovertemplate=f"<b>%{{x}}</b><br>{d['week']}: %{{y:.1f}}%<extra></extra>",
        ))
        _any = True
    if _any:
        fig.update_layout(
            **BASE, barmode="group",
            xaxis=dict(**GX),
            yaxis=dict(**GY,title="% of menu",title_font_size=11),
            legend=dict(font_size=11,bgcolor="rgba(0,0,0,0)",borderwidth=0,
                        orientation="h",y=1.08,x=0),
            margin=dict(l=8,r=8,t=36,b=8), height=280,
        )
        cc("Protein mix comparison — real menu data",
           "Source: mainprotein col from menu CSV · Gap slots excluded")
        chart(fig, "cmp_prot")
