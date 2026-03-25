"""
Smart Cricket Pod — Analytics Dashboard
Single-file Streamlit app. Upload: app.py + cricket_pod_survey_data.csv + requirements.txt
Models are trained on startup from the CSV — no .pkl files needed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import io
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, mean_squared_error, r2_score,
                              mean_absolute_error, silhouette_score)
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations

st.set_page_config(
    page_title="Smart Cricket Pod — Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
PRIMARY   = "#1D9E75"
SECONDARY = "#7F77DD"
ACCENT    = "#EF9F27"
DANGER    = "#D85A30"

st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{ background-color: #0f0f1a; }}
    [data-testid="stSidebar"] * {{ color: #e0e0e0 !important; }}
    .metric-card {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid {PRIMARY}44;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        color: white;
    }}
    .metric-card .val {{ font-size: 2rem; font-weight: 700; color: {PRIMARY}; }}
    .metric-card .lbl {{ font-size: 0.78rem; color: #aaa; margin-top: 4px; }}
    .section-header {{
        font-size: 1.1rem; font-weight: 600;
        border-left: 4px solid {PRIMARY};
        padding-left: 0.7rem; margin: 1.2rem 0 0.8rem;
        color: inherit;
    }}
    .insight-box {{
        background: #1a1a2e; border-left: 3px solid {ACCENT};
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.85rem; color: #ddd; margin: 0.5rem 0;
    }}
    .stDataFrame {{ border-radius: 10px; }}
    div[data-testid="metric-container"] {{
        background: #1a1a2e; border: 1px solid #333;
        border-radius: 10px; padding: 0.5rem 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

AGE_ORD   = {"Under 15":1,"15-18":2,"19-25":3,"26-35":4,"36-50":5,"50+":6}
CITY_ORD  = {"Rural":1,"Tier 3":2,"Tier 2":3,"Tier 1":4,"Metro":5}
INC_ORD   = {"Below 20K":1,"20K-40K":2,"40K-75K":3,"75K-150K":4,"Above 150K":5}
EDU_ORD   = {"Up to 10th":1,"12th/Diploma":2,"Bachelors":3,"Masters+":4}
PDAYS_ORD = {"0":0,"1-2":1,"3-4":2,"5-6":3,"Daily":4}
ROLE_ORD  = {"Not interested":0,"Fan only":1,"Occasional":2,"Regular":3,"Competitive":4,"Coach":3}
SPEND_ORD = {"0":0,"1-500":1,"501-1000":2,"1001-2500":3,"2501-5000":4,"Above 5000":5}
MEM_ORD   = {"Would not subscribe":0,"Up to 499":1,"500-999":2,"1000-1999":3,"2000-3000":4,"Above 3000":5}
DIG_ORD   = {"0":0,"1-200":1,"201-500":2,"501-1000":3,"Above 1000":4}
FD_ORD    = {"Rarely":1,"1-2/week":2,"3-4/week":3,"Daily":4}
TECH_ORD  = {"Tech avoider":0,"Laggard":1,"Late majority":2,"Early majority":3,"Early adopter":4}
DIST_ORD  = {"Within 1km":1,"Up to 3km":2,"Up to 5km":3,"Up to 10km":4,"Any distance":5}
FC_ORD    = {"Not interested":0,"Aware not using":1,"Occasional":2,"Active":3}
GENDER_ORD= {"Male":0,"Female":1,"Other/PNS":2}
PSM_TC_ORD= {"Below 50":1,"50-99":2,"100-149":3,"150-199":4,"200-249":5}
PSM_R_ORD = {"100-149":1,"150-199":2,"200-249":3,"250-299":4,"300-349":5}
PSM_E_ORD = {"200-299":1,"300-399":2,"400-499":3,"500-599":4,"600+":5}
PSM_TE_ORD= {"300-399":1,"400-499":2,"500-599":3,"600-799":4,"800+":5}

MULTI_SELECT_COLS = [
    "use_academy","use_boxcricket","use_bowlingmachine","use_homenet",
    "use_videoanalysis","use_mobilegame","use_gym",
    "disc_freetrial","disc_buy5get1","disc_student","disc_family",
    "disc_offpeak","disc_referral","disc_academy","disc_corporate",
    "feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork",
    "feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking",
    "brand_mrf","brand_sg","brand_ss","brand_kookaburra","brand_graynicolls",
    "brand_adidasnike","brand_decathlon","brand_nopref",
    "addon_smartbat","addon_wearables","addon_aicoaching",
    "addon_highlights","addon_fitness","addon_merch",
    "stream_hotstar","stream_jiocinema","stream_netflix",
    "stream_prime","stream_youtube","stream_sonyliv",
    "act_gym","act_yoga","act_othersport","act_swimming","act_videogaming","act_running",
    "past_boxcricket","past_trampoline","past_vr","past_bowling",
    "past_gokarting","past_fitclass","past_academy","past_golf",
    "frust_nodata","frust_coachattention","frust_timing","frust_crowded",
    "frust_distance","frust_cost","frust_equipment","frust_notracking",
    "bar_price","bar_location","bar_humancoach","bar_aidistrust",
    "bar_time","bar_notserious","bar_academy","bar_safety","bar_social",
    "hh_self","hh_child","hh_spouse","hh_sibling","hh_parent",
]

CLUSTERING_FEATURES = [
    "age_num","income_num","city_num","role_num","practice_num",
    "data_importance","pod_interest","spend_num","tech_num",
    "dist_num","nps_score","digital_num","fd_num",
    "addon_count","past_exp_count","barrier_count","feat_count",
]
CLASSIFICATION_FEATURES = [
    "age_num","gender_num","city_num","income_num","edu_num",
    "role_num","practice_num","data_importance","pod_interest",
    "spend_num","mem_num","digital_num","fd_num","tech_num",
    "dist_num","nps_score","addon_count","feat_count",
    "past_exp_count","barrier_count","frust_count",
    "past_boxcricket","past_trampoline","past_vr",
    "feat_ai","feat_bowlingmachine","feat_progressreport",
    "bar_aidistrust","bar_price","bar_location","bar_notserious",
    "use_academy","use_boxcricket",
]
REGRESSION_FEATURES = [
    "age_num","income_num","city_num","role_num","practice_num",
    "data_importance","pod_interest","spend_num","tech_num",
    "digital_num","nps_score","addon_count","feat_count",
    "past_exp_count","frust_count","barrier_count",
    "mem_num","dist_num",
]

def encode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_num"]      = d["age_group"].map(AGE_ORD).fillna(3)
    d["city_num"]     = d["city_tier"].map(CITY_ORD).fillna(3)
    d["income_num"]   = d["income_bracket"].map(INC_ORD).fillna(3)
    d["edu_num"]      = d["education"].map(EDU_ORD).fillna(2)
    d["role_num"]     = d["cricket_role"].map(ROLE_ORD).fillna(1)
    d["practice_num"] = d["practice_days"].map(PDAYS_ORD).fillna(1)
    d["spend_num"]    = d["monthly_rec_spend"].map(SPEND_ORD).fillna(2)
    d["mem_num"]      = d["membership_wtp"].map(MEM_ORD).fillna(1)
    d["digital_num"]  = d["digital_spend"].map(DIG_ORD).fillna(1)
    d["fd_num"]       = d["food_delivery_freq"].map(FD_ORD).fillna(2)
    d["tech_num"]     = d["tech_adoption"].map(TECH_ORD).fillna(2)
    d["dist_num"]     = d["distance_tolerance"].map(DIST_ORD).fillna(3)
    d["gender_num"]   = d["gender"].map(GENDER_ORD).fillna(0)
    d["fc_num"]       = d["fantasy_cricket"].map(FC_ORD).fillna(1)
    d["psm_tc_num"]   = d["psm_too_cheap"].map(PSM_TC_ORD).fillna(3)
    d["psm_r_num"]    = d["psm_reasonable"].map(PSM_R_ORD).fillna(3)
    d["psm_e_num"]    = d["psm_expensive"].map(PSM_E_ORD).fillna(3)
    d["psm_te_num"]   = d["psm_too_expensive"].map(PSM_TE_ORD).fillna(3)

    addon_cols   = ["addon_smartbat","addon_wearables","addon_aicoaching","addon_highlights","addon_fitness","addon_merch"]
    feat_cols    = ["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork","feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
    past_cols    = ["past_boxcricket","past_trampoline","past_vr","past_bowling","past_gokarting","past_fitclass","past_academy","past_golf"]
    barrier_cols = ["bar_price","bar_location","bar_humancoach","bar_aidistrust","bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    frust_cols   = ["frust_nodata","frust_coachattention","frust_timing","frust_crowded","frust_distance","frust_cost","frust_equipment","frust_notracking"]

    for clist, cname in [(addon_cols,"addon_count"),(feat_cols,"feat_count"),
                         (past_cols,"past_exp_count"),(barrier_cols,"barrier_count"),
                         (frust_cols,"frust_count")]:
        available = [c for c in clist if c in d.columns]
        d[cname] = d[available].fillna(0).sum(axis=1)
    return d

def get_cluster_features(df_enc):
    available = [c for c in CLUSTERING_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

def get_classification_features(df_enc):
    available = [c for c in CLASSIFICATION_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

def get_regression_features(df_enc):
    available = [c for c in REGRESSION_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

# ══════════════════════════════════════════════════════════════════════════════
# APRIORI (pure-pandas fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _apriori(df_basket, min_support=0.05, max_len=4):
    n = len(df_basket)
    cols = df_basket.columns.tolist()
    arr  = df_basket.values.astype(bool)
    freq = {}
    for i, col in enumerate(cols):
        sup = arr[:, i].sum() / n
        if sup >= min_support:
            freq[frozenset([col])] = sup
    result = list(freq.items())
    prev_level = list(freq.keys())
    for length in range(2, max_len + 1):
        if not prev_level:
            break
        prev_list = sorted([sorted(fs) for fs in prev_level])
        next_level = []
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                if a[:-1] == b[:-1]:
                    cand = frozenset(a) | frozenset(b)
                    idx = [cols.index(c) for c in cand if c in cols]
                    if len(idx) != len(cand):
                        continue
                    sup = (arr[:, idx].all(axis=1)).sum() / n
                    if sup >= min_support:
                        freq[cand] = sup
                        result.append((cand, sup))
                        next_level.append(cand)
        prev_level = next_level
    rows = [{"itemsets": fs, "support": sup} for fs, sup in result]
    return pd.DataFrame(rows)

def _association_rules(freq_itemsets, metric="lift", min_threshold=1.0):
    rows = []
    item_support = {frozenset(row["itemsets"]): row["support"] for _, row in freq_itemsets.iterrows()}
    for _, row in freq_itemsets.iterrows():
        itemset = frozenset(row["itemsets"])
        if len(itemset) < 2:
            continue
        sup_ab = row["support"]
        for i in range(1, len(itemset)):
            for ant in combinations(itemset, i):
                ant = frozenset(ant)
                con = itemset - ant
                sup_a = item_support.get(ant, np.nan)
                sup_b = item_support.get(con, np.nan)
                if np.isnan(sup_a) or np.isnan(sup_b) or sup_a == 0:
                    continue
                conf = sup_ab / sup_a
                lift = conf / sup_b if sup_b > 0 else 0
                lev  = sup_ab - sup_a * sup_b
                conv = (1 - sup_b) / (1 - conf) if conf < 1 else np.inf
                rows.append({"antecedents": ant, "consequents": con,
                              "support": sup_ab, "confidence": conf,
                              "lift": lift, "leverage": lev, "conviction": conv})
    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules
    if metric == "lift":
        rules = rules[rules["lift"] >= min_threshold]
    elif metric == "confidence":
        rules = rules[rules["confidence"] >= min_threshold]
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING (cached — runs once per session)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    df = pd.read_csv("cricket_pod_survey_data.csv")
    return df

@st.cache_resource
def train_all(_df):
    """Train all models from scratch. Cached so it runs only once."""
    df = _df.copy()
    results = {}
    df_enc = encode(df)

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    X_clust = get_cluster_features(df_enc)
    scaler_clust = StandardScaler()
    X_clust_s = scaler_clust.fit_transform(X_clust)

    inertias, silhouettes_list = [], []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_clust_s)
        inertias.append(km.inertia_)
        silhouettes_list.append(silhouette_score(X_clust_s, labs))

    best_k = 5
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_clust_s)
    df_enc["cluster"] = cluster_labels

    cluster_profiles = []
    persona_map = {}
    for c in range(best_k):
        mask = df_enc["cluster"] == c
        sub  = df_enc[mask]
        prof = {
            "cluster": c,
            "size": int(mask.sum()),
            "avg_income": float(sub["income_num"].mean()),
            "avg_role": float(sub["role_num"].mean()),
            "avg_pod_interest": float(sub["pod_interest"].mean()),
            "avg_spend": float(sub["realistic_monthly_spend"].mean()) if "realistic_monthly_spend" in sub else 0,
            "avg_age": float(sub["age_num"].mean()),
            "conversion_rate": float((sub["pod_conversion_binary"]==1).mean()) if "pod_conversion_binary" in sub else 0,
        }
        cluster_profiles.append(prof)
        if prof["avg_role"] >= 3.5 and prof["avg_income"] <= 2.5:
            name = "Rising Star"
        elif prof["avg_role"] >= 3.0 and prof["avg_income"] >= 3.5:
            name = "Elite Competitor"
        elif prof["avg_income"] >= 4.0 and prof["avg_role"] <= 2.0:
            name = "Corporate Cricket Fan"
        elif prof["avg_pod_interest"] <= 2.5:
            name = "Sceptic / Disengaged"
        else:
            name = "Recreational Player"
        persona_map[c] = name

    results["clustering"] = {
        "inertias": inertias, "silhouettes": silhouettes_list,
        "best_k": best_k, "labels": cluster_labels,
        "profiles": cluster_profiles, "persona_map": persona_map,
    }

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    valid_mask = df_enc["pod_conversion_binary"].notna()
    df_clf     = df_enc[valid_mask].copy()
    X_clf_raw  = get_classification_features(df_clf)
    y_clf      = df_clf["pod_conversion_binary"].astype(int)

    scaler_clf = StandardScaler()
    X_clf_s    = scaler_clf.fit_transform(X_clf_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(X_clf_s, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_prob = rf.predict_proba(X_te)[:,1]
    fpr, tpr, _ = roc_curve(y_te, rf_prob)

    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_clf.fit(X_tr, y_tr)
    lr_pred = lr_clf.predict(X_te)
    lr_prob = lr_clf.predict_proba(X_te)[:,1]
    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_prob)

    feat_imp = pd.Series(rf.feature_importances_, index=X_clf_raw.columns).sort_values(ascending=False)

    results["classification"] = {
        "rf": {"acc": accuracy_score(y_te, rf_pred),
               "prec": precision_score(y_te, rf_pred),
               "rec":  recall_score(y_te, rf_pred),
               "f1":   f1_score(y_te, rf_pred),
               "auc":  roc_auc_score(y_te, rf_prob),
               "cm":   confusion_matrix(y_te, rf_pred).tolist(),
               "fpr":  fpr.tolist(), "tpr": tpr.tolist()},
        "lr": {"acc": accuracy_score(y_te, lr_pred),
               "prec": precision_score(y_te, lr_pred),
               "rec":  recall_score(y_te, lr_pred),
               "f1":   f1_score(y_te, lr_pred),
               "auc":  roc_auc_score(y_te, lr_prob),
               "cm":   confusion_matrix(y_te, lr_pred).tolist(),
               "fpr":  fpr_lr.tolist(), "tpr": tpr_lr.tolist()},
        "feat_imp": feat_imp.head(20).to_dict(),
        "y_test": y_te.tolist(), "rf_prob": rf_prob.tolist(),
    }

    # ── ASSOCIATION RULES ─────────────────────────────────────────────────────
    avail_ms = [c for c in MULTI_SELECT_COLS if c in df.columns]
    basket = df[avail_ms].fillna(0).astype(bool)
    try:
        from mlxtend.frequent_patterns import apriori as mlx_apriori, association_rules as mlx_rules
        freq_items = mlx_apriori(basket, min_support=0.05, use_colnames=True, max_len=4)
        rules = mlx_rules(freq_items, metric="lift", min_threshold=1.2)
    except Exception:
        freq_items = _apriori(basket, min_support=0.05, max_len=4)
        rules = _association_rules(freq_items, metric="lift", min_threshold=1.2)

    rules = rules[rules["confidence"] >= 0.50].sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    results["association"] = {"rules": rules}

    # ── REGRESSION ────────────────────────────────────────────────────────────
    X_reg_raw = get_regression_features(df_enc)
    y_reg     = df_enc["realistic_monthly_spend"].fillna(df_enc["realistic_monthly_spend"].median())

    scaler_reg = StandardScaler()
    X_reg_s    = scaler_reg.fit_transform(X_reg_raw)

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_reg_s, y_reg, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xr_tr, yr_tr)
    ridge_pred = ridge.predict(Xr_te)

    lr_reg = LinearRegression()
    lr_reg.fit(Xr_tr, yr_tr)
    lr_reg_pred = lr_reg.predict(Xr_te)

    coef_imp = pd.Series(np.abs(ridge.coef_), index=X_reg_raw.columns).sort_values(ascending=False)

    results["regression"] = {
        "ridge": {"r2": r2_score(yr_te, ridge_pred),
                  "rmse": np.sqrt(mean_squared_error(yr_te, ridge_pred)),
                  "mae": mean_absolute_error(yr_te, ridge_pred),
                  "y_test": yr_te.tolist(), "y_pred": ridge_pred.tolist()},
        "lr":    {"r2": r2_score(yr_te, lr_reg_pred),
                  "rmse": np.sqrt(mean_squared_error(yr_te, lr_reg_pred)),
                  "mae": mean_absolute_error(yr_te, lr_reg_pred)},
        "coef_imp": coef_imp.head(15).to_dict(),
    }

    # Bundle all models
    models = {
        "kmeans": km_final,
        "scaler_clust": scaler_clust,
        "cluster_features": X_clust.columns.tolist(),
        "cluster_profiles": cluster_profiles,
        "persona_map": persona_map,
        "rf_classifier": rf,
        "lr_classifier": lr_clf,
        "scaler_clf": scaler_clf,
        "clf_features": X_clf_raw.columns.tolist(),
        "ridge_regressor": ridge,
        "lr_regressor": lr_reg,
        "scaler_reg": scaler_reg,
        "reg_features": X_reg_raw.columns.tolist(),
        "assoc_rules": rules,
        "all_results": results,
    }

    df_enc["persona"] = df_enc["cluster"].map(persona_map)
    return models, df_enc

# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA & TRAIN
# ══════════════════════════════════════════════════════════════════════════════

df = load_data()
with st.spinner("🏏 Training models on startup (this takes ~30 seconds the first time)..."):
    models, df_enc = train_all(df)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🏏 Smart Cricket Pod")
    st.markdown("**Data-Driven Analytics Platform**")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠  Home — Executive Summary",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🎯  Classification",
        "👥  Clustering — Personas",
        "🔗  Association Rule Mining",
        "📈  Regression — Spend Forecast",
        "🚀  New Customer Predictor",
    ])
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} respondents · {df.shape[1]} features")
    st.caption("v2.0 — Smart Cricket Pod")

CLUSTER_COLORS = [PRIMARY, SECONDARY, ACCENT, DANGER, "#5DCAA5", "#F0997B", "#85B7EB"]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

def page_home():
    st.title("🏏 Smart Cricket Pod — Analytics Dashboard")
    st.markdown("#### Data-Driven Decision Making for India's First AI-Powered Cricket Pod Network")
    st.markdown("---")

    total      = len(df)
    interested = int((df["pod_conversion_binary"] == 1).sum())
    not_int    = int((df["pod_conversion_binary"] == 0).sum())
    maybe      = int(df["pod_conversion_binary"].isna().sum())
    conv_rate  = interested / (interested + not_int) * 100
    avg_spend  = df["realistic_monthly_spend"].mean()
    avg_nps    = df["nps_score"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kpi(col, val, lbl, color=PRIMARY):
        col.markdown(f"""<div class="metric-card">
          <div class="val" style="color:{color}">{val}</div>
          <div class="lbl">{lbl}</div></div>""", unsafe_allow_html=True)

    kpi(c1, f"{total:,}",        "Total Respondents")
    kpi(c2, f"{interested:,}",   "Interested (Label=1)")
    kpi(c3, f"{conv_rate:.1f}%", "Conversion Rate",       ACCENT)
    kpi(c4, f"₹{avg_spend:,.0f}","Avg Monthly Spend",     SECONDARY)
    kpi(c5, f"{avg_nps:.1f}/10", "Avg NPS Score",         "#E74C3C" if avg_nps<6 else PRIMARY)
    kpi(c6, f"{maybe:,}",        "Maybe (Undecided)",     DANGER)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.2, 1, 1])
    with col1:
        st.markdown('<div class="section-header">Conversion Signal</div>', unsafe_allow_html=True)
        conv_counts = df["pod_conversion"].value_counts()
        color_map = {"Yes - definitely": PRIMARY, "Yes - if price right": "#5DCAA5",
                     "Maybe": ACCENT, "Unlikely": "#F0997B", "No": DANGER}
        fig = go.Figure(go.Bar(x=conv_counts.values, y=conv_counts.index, orientation="h",
                               marker_color=[color_map.get(l,"#888") for l in conv_counts.index],
                               text=[f"{v:,}" for v in conv_counts.values], textposition="outside"))
        fig.update_layout(height=280, margin=dict(l=0,r=40,t=10,b=10),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">True Segments</div>', unsafe_allow_html=True)
        seg_counts = df["true_segment"].value_counts()
        fig2 = go.Figure(go.Pie(labels=seg_counts.index, values=seg_counts.values, hole=0.45,
                                marker_colors=[PRIMARY,SECONDARY,ACCENT,DANGER,"#888"], textinfo="percent"))
        fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                           paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(font=dict(color="#ccc",size=10)), font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown('<div class="section-header">City Tier Distribution</div>', unsafe_allow_html=True)
        city_counts = df["city_tier"].value_counts()
        fig3 = go.Figure(go.Bar(x=city_counts.index, y=city_counts.values, marker_color=SECONDARY,
                                text=city_counts.values, textposition="outside"))
        fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=30),
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa"),
                           font=dict(color="#ccc"))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Dataset Health</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        null_counts = df.isnull().sum()
        null_pct    = (null_counts / len(df) * 100).round(2)
        null_df     = pd.DataFrame({"Missing Values": null_counts, "Missing %": null_pct})
        null_df     = null_df[null_df["Missing Values"] > 0].sort_values("Missing %", ascending=False)
        st.markdown("**Columns with missing values:**")
        st.dataframe(null_df.head(15), use_container_width=True, height=280)
    with col_b:
        st.markdown("**Quick dataset stats:**")
        stats_dict = {
            "Total Rows": len(df), "Total Columns": df.shape[1],
            "Numeric Columns": int(df.select_dtypes(include=np.number).shape[1]),
            "Categorical Cols": int(df.select_dtypes(include="object").shape[1]),
            "Binary (0/1) Cols": int((df.nunique() == 2).sum()),
            "Duplicate Rows": int(df.duplicated().sum()),
            "Complete Rows": int(df.dropna().shape[0]),
        }
        st.dataframe(pd.DataFrame.from_dict(stats_dict, orient="index", columns=["Value"]),
                     use_container_width=True, height=280)

    st.markdown("---")
    st.markdown("### Dashboard Navigation Guide")
    guide = [
        ("📊 Descriptive", "Who are my customers? PSM price curves, demographics, barrier analysis"),
        ("🔍 Diagnostic",  "Why are they interested? Correlation heatmap, chi-square tests, cross-tabs"),
        ("🎯 Classification","Will they convert? Random Forest + Logistic Regression, ROC-AUC"),
        ("👥 Clustering",  "What type are they? K-Means personas, radar charts"),
        ("🔗 Association",  "What goes together? Apriori rules, bundle discovery"),
        ("📈 Regression",  "How much will they spend? Ridge regression, revenue forecaster"),
        ("🚀 Predictor",   "Upload new leads → get conversion score + recommended offer"),
    ]
    cols = st.columns(4)
    for i, (title, desc) in enumerate(guide):
        cols[i % 4].markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #333;border-radius:10px;
                    padding:0.8rem;margin-bottom:8px;min-height:90px">
          <div style="font-weight:600;font-size:0.85rem;color:{PRIMARY};margin-bottom:4px">{title}</div>
          <div style="font-size:0.75rem;color:#aaa;line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════

PSM_TC_MID = {"Below 50":50,"50-99":75,"100-149":125,"150-199":175,"200-249":225}
PSM_R_MID  = {"100-149":125,"150-199":175,"200-249":225,"250-299":275,"300-349":325}
PSM_E_MID  = {"200-299":250,"300-399":350,"400-499":450,"500-599":550,"600+":650}
PSM_TE_MID = {"300-399":350,"400-499":450,"500-599":550,"600-799":700,"800+":900}

def page_descriptive():
    st.title("📊 Descriptive Analysis")
    st.markdown("Understanding the customer landscape — demographics, behaviours, and pricing signals.")
    st.markdown("---")

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        age_c = df["age_group"].value_counts()
        order = ["Under 15","15-18","19-25","26-35","36-50","50+"]
        age_c = age_c.reindex([o for o in order if o in age_c.index])
        fig = go.Figure(go.Bar(x=age_c.index, y=age_c.values, marker_color=PRIMARY,
                               text=age_c.values, textposition="outside"))
        fig.update_layout(title="Age distribution", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gen_c = df["gender"].value_counts()
        fig = go.Figure(go.Pie(labels=gen_c.index, values=gen_c.values, hole=0.4,
                               marker_colors=[SECONDARY,PRIMARY,ACCENT]))
        fig.update_layout(title="Gender split", height=280, paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(color="#ccc",size=11)),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        inc_c = df["income_bracket"].value_counts()
        order_i = ["Below 20K","20K-40K","40K-75K","75K-150K","Above 150K"]
        inc_c = inc_c.reindex([o for o in order_i if o in inc_c.index])
        fig = go.Figure(go.Bar(x=inc_c.values, y=inc_c.index, orientation="h",
                               marker_color=ACCENT, text=inc_c.values, textposition="outside"))
        fig.update_layout(title="Income bracket", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=40))
        st.plotly_chart(fig, use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        role_c = df["cricket_role"].value_counts()
        fig = go.Figure(go.Pie(labels=role_c.index, values=role_c.values, hole=0.4,
                               marker_colors=[PRIMARY,SECONDARY,ACCENT,DANGER,"#888","#5DCAA5"]))
        fig.update_layout(title="Cricket role", height=280, paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(color="#ccc",size=10)),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        occ_c = df["occupation"].value_counts().head(8)
        fig = go.Figure(go.Bar(x=occ_c.values, y=occ_c.index, orientation="h",
                               marker_color=SECONDARY, text=occ_c.values, textposition="outside"))
        fig.update_layout(title="Occupation", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=40))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        city_c = df["city_tier"].value_counts()
        fig = go.Figure(go.Bar(x=city_c.index, y=city_c.values, marker_color=DANGER,
                               text=city_c.values, textposition="outside"))
        fig.update_layout(title="City tier", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Cricket Behaviour</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        pd_c = df["practice_days"].value_counts()
        order_p = ["0","1-2","3-4","5-6","Daily"]
        pd_c = pd_c.reindex([o for o in order_p if o in pd_c.index])
        fig = go.Figure(go.Bar(x=pd_c.index, y=pd_c.values,
                               marker_color=[PRIMARY if v in ["3-4","5-6","Daily"] else "#888" for v in pd_c.index],
                               text=pd_c.values, textposition="outside"))
        fig.update_layout(title="Practice days per week", height=260,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        feat_cols = ["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork",
                     "feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
        feat_labels = ["AI Analysis","Bowling Machine","Bat Speed","Footwork",
                       "Video Replay","Leaderboard","Progress Report","App Booking"]
        avail = [c for c in feat_cols if c in df.columns]
        labels_avail = [feat_labels[feat_cols.index(c)] for c in avail]
        feat_pcts = df[avail].fillna(0).mean() * 100
        fig = go.Figure(go.Bar(x=feat_pcts.values, y=labels_avail, orientation="h",
                               marker_color=PRIMARY,
                               text=[f"{v:.1f}%" for v in feat_pcts.values], textposition="outside"))
        fig.update_layout(title="Feature interest (% respondents)", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0,100],showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=60))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Barriers to Adoption</div>', unsafe_allow_html=True)
    bar_cols   = ["bar_price","bar_location","bar_humancoach","bar_aidistrust",
                  "bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    bar_labels = ["Too expensive","No pod nearby","Prefer human coach","AI distrust",
                  "No time","Not serious","Already in academy","Safety concern","Want friends along"]
    avail_b = [c for c in bar_cols if c in df.columns]
    labels_b = [bar_labels[bar_cols.index(c)] for c in avail_b]
    bar_pcts = df[avail_b].fillna(0).mean() * 100
    bar_series = pd.Series(bar_pcts.values, index=labels_b).sort_values(ascending=True)
    fig_bar = go.Figure(go.Bar(x=bar_series.values, y=bar_series.index, orientation="h",
                               marker_color=[DANGER if v>30 else ACCENT if v>20 else "#888" for v in bar_series.values],
                               text=[f"{v:.1f}%" for v in bar_series.values], textposition="outside"))
    fig_bar.update_layout(height=320, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0,70],showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=10,b=10,l=0,r=60))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-header">Van Westendorp Price Sensitivity Meter</div>', unsafe_allow_html=True)
    df_psm = df.copy()
    df_psm["tc_mid"] = df_psm["psm_too_cheap"].map(PSM_TC_MID)
    df_psm["r_mid"]  = df_psm["psm_reasonable"].map(PSM_R_MID)
    df_psm["e_mid"]  = df_psm["psm_expensive"].map(PSM_E_MID)
    df_psm["te_mid"] = df_psm["psm_too_expensive"].map(PSM_TE_MID)
    df_psm = df_psm.dropna(subset=["tc_mid","r_mid","e_mid","te_mid"])
    prices = np.arange(50, 900, 10)
    n_valid = len(df_psm)
    pct_tc = [(df_psm["tc_mid"] >= p).sum() / n_valid * 100 for p in prices]
    pct_r  = [(df_psm["r_mid"]  <= p).sum() / n_valid * 100 for p in prices]
    pct_e  = [(df_psm["e_mid"]  <= p).sum() / n_valid * 100 for p in prices]
    pct_te = [(df_psm["te_mid"] <= p).sum() / n_valid * 100 for p in prices]
    r_arr, e_arr = np.array(pct_r), np.array(pct_e)
    tc_arr, te_arr = np.array(pct_tc), np.array(pct_te)
    opp_idx = np.argmin(np.abs(r_arr - e_arr)); opp_price = int(prices[opp_idx])
    apr_lo_idx = np.where(r_arr - tc_arr >= 0)[0]; apr_lo = int(prices[apr_lo_idx[0]]) if len(apr_lo_idx) else 100
    apr_hi_idx = np.where(te_arr - (100-r_arr) >= 0)[0]; apr_hi = int(prices[apr_hi_idx[0]]) if len(apr_hi_idx) else 500
    fig_psm = go.Figure()
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_tc, name="Too cheap",   line=dict(color="#5DCAA5",width=2,dash="dot")))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_r,  name="Reasonable", line=dict(color=PRIMARY,width=2.5)))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_e,  name="Expensive",  line=dict(color=ACCENT,width=2,dash="dash")))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_te, name="Too expensive", line=dict(color=DANGER,width=2.5)))
    fig_psm.add_vrect(x0=apr_lo, x1=apr_hi, fillcolor=PRIMARY, opacity=0.08, layer="below", line_width=0)
    fig_psm.add_vline(x=opp_price, line_color=PRIMARY, line_dash="dash", line_width=2,
                      annotation_text=f"Optimal: ₹{opp_price}", annotation_font_color=PRIMARY, annotation_position="top right")
    fig_psm.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(title="Price per 30-min session (₹)",color="#ccc",tickprefix="₹",showgrid=True,gridcolor="#333"),
                          yaxis=dict(title="% respondents",color="#aaa",showgrid=True,gridcolor="#333"),
                          legend=dict(font=dict(color="#ccc")), font=dict(color="#ccc"), margin=dict(t=20,b=40,l=0,r=0))
    st.plotly_chart(fig_psm, use_container_width=True)
    col_p1,col_p2,col_p3 = st.columns(3)
    col_p1.metric("Acceptable Price Range", f"₹{apr_lo} – ₹{apr_hi}")
    col_p2.metric("Optimal Price Point (OPP)", f"₹{opp_price}")
    col_p3.metric("Recommended Launch Price", f"₹{opp_price-10} – ₹{opp_price+15}")
    st.markdown(f"""<div class="insight-box">📌 <strong>Founder action:</strong> Launch price of ₹{opp_price} per 30-min session
    maximises conversion. Acceptable range ₹{apr_lo}–₹{apr_hi} gives flexibility for student discounts and premium weekend slots.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════

def page_diagnostic():
    st.title("🔍 Diagnostic Analysis")
    st.markdown("Uncovering *why* customers are or aren't interested — correlations, cross-tabs and statistical tests.")
    st.markdown("---")

    num_cols = ["age_num","income_num","city_num","role_num","practice_num",
                "data_importance","pod_interest","spend_num","tech_num",
                "dist_num","nps_score","digital_num","addon_count","feat_count",
                "past_exp_count","barrier_count","frust_count","pod_conversion_binary"]
    avail = [c for c in num_cols if c in df_enc.columns]
    corr_df = df_enc[avail].apply(pd.to_numeric, errors="coerce").corr().round(2)
    labels_map = {
        "age_num":"Age","income_num":"Income","city_num":"City Tier","role_num":"Cricket Role",
        "practice_num":"Practice Days","data_importance":"Data Importance","pod_interest":"Pod Interest",
        "spend_num":"Rec Spend","tech_num":"Tech Adoption","dist_num":"Distance Toler.",
        "nps_score":"NPS Score","digital_num":"Digital Spend","addon_count":"Addon Count",
        "feat_count":"Feature Count","past_exp_count":"Past Exp","barrier_count":"Barrier Count",
        "frust_count":"Frustration Ct","pod_conversion_binary":"Conversion",
    }
    tick_labels = [labels_map.get(c, c) for c in corr_df.columns]

    st.markdown('<div class="section-header">Correlation Heatmap — Numeric Features</div>', unsafe_allow_html=True)
    fig_hm = go.Figure(go.Heatmap(z=corr_df.values, x=tick_labels, y=tick_labels,
                                   colorscale=[[0,DANGER],[0.5,"#1a1a2e"],[1,PRIMARY]],
                                   zmid=0, zmin=-1, zmax=1,
                                   text=corr_df.values.round(2), texttemplate="%{text}",
                                   textfont=dict(size=8),
                                   colorbar=dict(tickfont=dict(color="#ccc"))))
    fig_hm.update_layout(height=500, margin=dict(t=10,b=10,l=0,r=0), paper_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(tickfont=dict(color="#ccc",size=9),tickangle=-45),
                         yaxis=dict(tickfont=dict(color="#ccc",size=9)), font=dict(color="#ccc"))
    st.plotly_chart(fig_hm, use_container_width=True)

    if "pod_conversion_binary" in corr_df.columns:
        conv_corr = corr_df["pod_conversion_binary"].drop("pod_conversion_binary").abs().sort_values(ascending=False)
        st.markdown("**Top features correlated with conversion:**")
        top5 = conv_corr.head(5)
        cols = st.columns(5)
        for i, (feat, val) in enumerate(top5.items()):
            cols[i].metric(labels_map.get(feat, feat), f"{val:.3f}")

    st.markdown("---")
    st.markdown('<div class="section-header">Income vs Willingness to Pay</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        scatter_df = df.copy()
        scatter_df["psm_r_val"]  = scatter_df["psm_reasonable"].map(PSM_R_MID)
        scatter_df["income_val"] = scatter_df["income_bracket"].map(INC_ORD)
        scatter_df = scatter_df.dropna(subset=["psm_r_val","income_val"])
        inc_labels = {1:"<20K",2:"20-40K",3:"40-75K",4:"75-150K",5:">150K"}
        scatter_df["income_label"] = scatter_df["income_val"].map(inc_labels)
        avg_wtp = scatter_df.groupby("income_label")["psm_r_val"].mean().reset_index()
        order_inc = ["<20K","20-40K","40-75K","75-150K",">150K"]
        avg_wtp["order"] = avg_wtp["income_label"].map({v:i for i,v in enumerate(order_inc)})
        avg_wtp = avg_wtp.sort_values("order")
        fig_sc = go.Figure(go.Bar(x=avg_wtp["income_label"], y=avg_wtp["psm_r_val"],
                                  marker_color=PRIMARY, text=[f"₹{v:.0f}" for v in avg_wtp["psm_r_val"]],
                                  textposition="outside"))
        fig_sc.update_layout(title="Avg reasonable price by income", height=300,
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa",title="₹"),
                             font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        if "city_tier" in df_enc.columns and "pod_conversion_binary" in df_enc.columns:
            conv_by_city = df_enc.groupby("city_tier")["pod_conversion_binary"].mean().dropna() * 100
            conv_by_city = conv_by_city.sort_values(ascending=False)
            fig_city = go.Figure(go.Bar(x=conv_by_city.index, y=conv_by_city.values,
                                        marker_color=[PRIMARY if v==conv_by_city.max() else SECONDARY for v in conv_by_city.values],
                                        text=[f"{v:.1f}%" for v in conv_by_city.values], textposition="outside"))
            fig_city.update_layout(title="Conversion rate by city tier", height=300,
                                   plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                   xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa",title="%"),
                                   font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
            st.plotly_chart(fig_city, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Statistical Tests</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Chi-Square: Past Box Cricket Experience vs Conversion**")
        if "past_boxcricket" in df_enc.columns and "pod_conversion_binary" in df_enc.columns:
            test_df = df_enc[["past_boxcricket","pod_conversion_binary"]].dropna()
            test_df = test_df.copy()
            test_df["past_boxcricket"] = test_df["past_boxcricket"].fillna(0).astype(int)
            ct = pd.crosstab(test_df["past_boxcricket"], test_df["pod_conversion_binary"].astype(int))
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            st.dataframe(ct, use_container_width=True)
            st.markdown(f"""<div class="insight-box">χ² = <strong>{chi2:.2f}</strong> | p-value = <strong>{p:.4f}</strong> | dof = {dof}<br>
            {'✅ Statistically significant (p < 0.05)' if p < 0.05 else '❌ Not significant at 0.05 level.'}</div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("**ANOVA: Tech Adoption Style vs Pod Interest Score**")
        if "tech_adoption" in df_enc.columns:
            groups = [df_enc[df_enc["tech_adoption"]==g]["pod_interest"].dropna().values
                      for g in df_enc["tech_adoption"].unique() if len(df_enc[df_enc["tech_adoption"]==g]) > 5]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                avg_by_tech = df_enc.groupby("tech_adoption")["pod_interest"].mean().sort_values(ascending=False)
                fig_an = go.Figure(go.Bar(x=avg_by_tech.index, y=avg_by_tech.values, marker_color=SECONDARY,
                                          text=[f"{v:.2f}" for v in avg_by_tech.values], textposition="outside"))
                fig_an.update_layout(height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                     xaxis=dict(color="#ccc",tickangle=-20),
                                     yaxis=dict(showgrid=False,color="#aaa",title="Avg pod interest"),
                                     font=dict(color="#ccc",size=10), margin=dict(t=10,b=60,l=0,r=0))
                st.plotly_chart(fig_an, use_container_width=True)
                st.markdown(f"""<div class="insight-box">F = <strong>{f_stat:.2f}</strong> | p = <strong>{p_val:.4f}</strong><br>
                {'✅ Significant — tech adoption affects pod interest.' if p_val < 0.05 else '❌ Not significant.'}</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Competitor Frustration → Conversion Heatmap</div>', unsafe_allow_html=True)
    frust_cols = ["frust_nodata","frust_coachattention","frust_timing","frust_crowded",
                  "frust_distance","frust_cost","frust_equipment","frust_notracking"]
    frust_labels = ["No data/feedback","Coach inattentive","Bad timing","Too crowded",
                    "Too far","High cost","Old equipment","No progress tracking"]
    avail_f = [c for c in frust_cols if c in df_enc.columns]
    labels_avail = [frust_labels[frust_cols.index(c)] for c in avail_f]
    if avail_f and "pod_conversion_binary" in df_enc.columns:
        rows = []
        for col, lbl in zip(avail_f, labels_avail):
            sub = df_enc[[col,"pod_conversion_binary"]].dropna().copy()
            sub[col] = sub[col].fillna(0)
            g0 = sub[sub[col]==0]["pod_conversion_binary"].mean()
            g1 = sub[sub[col]==1]["pod_conversion_binary"].mean()
            rows.append({"Frustration": lbl, "Has frustration": g1, "No frustration": g0, "Lift": g1-g0})
        frust_df = pd.DataFrame(rows).sort_values("Lift", ascending=False)
        fig_frust = go.Figure()
        fig_frust.add_trace(go.Bar(name="Has frustration", x=frust_df["Frustration"],
                                    y=(frust_df["Has frustration"]*100).round(1), marker_color=PRIMARY,
                                    text=[f"{v:.1f}%" for v in frust_df["Has frustration"]*100], textposition="outside"))
        fig_frust.add_trace(go.Bar(name="No frustration", x=frust_df["Frustration"],
                                    y=(frust_df["No frustration"]*100).round(1), marker_color="#444"))
        fig_frust.update_layout(barmode="group", height=340,
                                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(color="#ccc",tickangle=-25),
                                yaxis=dict(showgrid=False,color="#aaa",title="Conversion rate %"),
                                legend=dict(font=dict(color="#ccc")),
                                font=dict(color="#ccc"), margin=dict(t=10,b=80,l=0,r=0))
        st.plotly_chart(fig_frust, use_container_width=True)
        st.markdown("""<div class="insight-box">📌 <strong>Founder action:</strong> Respondents frustrated by "No data/feedback" and
        "No progress tracking" show the highest conversion lift — these are your warmest leads.</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def page_classification():
    st.title("🎯 Classification — Will This Customer Convert?")
    st.markdown("Predicting pod conversion using **Random Forest** and **Logistic Regression**.")
    st.markdown("---")

    results = models.get("all_results", {})
    clf = results.get("classification", {})
    rf  = clf.get("rf", {}); lr = clf.get("lr", {})
    if not rf:
        st.error("Classification results not found."); return

    metrics = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
    rf_vals = [rf["acc"],rf["prec"],rf["rec"],rf["f1"],rf["auc"]]
    lr_vals = [lr["acc"],lr["prec"],lr["rec"],lr["f1"],lr["auc"]]

    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    for i,(m,rv,lv) in enumerate(zip(metrics,rf_vals,lr_vals)):
        cols[i].metric(m, f"{rv:.3f}", f"RF vs LR: {rv-lv:+.3f}")

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Random Forest", x=metrics, y=rf_vals, marker_color=PRIMARY,
                               text=[f"{v:.3f}" for v in rf_vals], textposition="outside"))
    fig_comp.add_trace(go.Bar(name="Logistic Regression", x=metrics, y=lr_vals, marker_color=SECONDARY,
                               text=[f"{v:.3f}" for v in lr_vals], textposition="outside"))
    fig_comp.update_layout(barmode="group", height=320,
                           plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#ccc"), yaxis=dict(range=[0,1.15],showgrid=False,color="#aaa"),
                           legend=dict(font=dict(color="#ccc")),
                           font=dict(color="#ccc"), margin=dict(t=10,b=10,l=0,r=0))
    st.plotly_chart(fig_comp, use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=rf["fpr"],y=rf["tpr"],mode="lines",
                                      name=f"Random Forest (AUC={rf['auc']:.3f})",line=dict(color=PRIMARY,width=2.5)))
        fig_roc.add_trace(go.Scatter(x=lr["fpr"],y=lr["tpr"],mode="lines",
                                      name=f"Logistic Reg (AUC={lr['auc']:.3f})",line=dict(color=SECONDARY,width=2,dash="dash")))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Baseline",line=dict(color="#555",width=1,dash="dot")))
        fig_roc.update_layout(height=380, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title="False Positive Rate",color="#ccc",showgrid=True,gridcolor="#333"),
                              yaxis=dict(title="True Positive Rate",color="#aaa",showgrid=True,gridcolor="#333"),
                              legend=dict(font=dict(color="#ccc",size=11)),
                              font=dict(color="#ccc"), margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        st.markdown("**Confusion Matrix — Random Forest**")
        cm = np.array(rf["cm"])
        labels_cm = ["Not Interested (0)","Interested (1)"]
        fig_cm = ff.create_annotated_heatmap(z=cm, x=labels_cm, y=labels_cm,
                                              colorscale=[[0,"#1a1a2e"],[1,PRIMARY]],
                                              annotation_text=cm.astype(str), showscale=True)
        fig_cm.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"),
                             margin=dict(t=30,b=60,l=0,r=0),
                             xaxis=dict(title="Predicted",color="#ccc"),
                             yaxis=dict(title="Actual",color="#ccc"))
        st.plotly_chart(fig_cm, use_container_width=True)
        metrics_tbl = pd.DataFrame({
            "Metric": ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"],
            "Random Forest": [f"{rf[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]],
            "Logistic Reg.": [f"{lr[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]],
        })
        st.dataframe(metrics_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Feature Importance — Random Forest (Top 20)</div>', unsafe_allow_html=True)
    feat_imp = clf.get("feat_imp", {})
    feat_names_map = {
        "role_num":"Cricket Role","pod_interest":"Pod Interest","data_importance":"Data Importance",
        "income_num":"Income","past_exp_count":"Past Exp Count","tech_num":"Tech Adoption",
        "nps_score":"NPS Score","feat_count":"Feature Interest Count","addon_count":"Add-on Interest",
        "barrier_count":"Barrier Count","age_num":"Age","city_num":"City Tier","spend_num":"Rec Spend",
        "practice_num":"Practice Days","frust_count":"Frustration Count",
        "bar_aidistrust":"Barrier: AI Distrust","bar_notserious":"Barrier: Not Serious",
        "past_boxcricket":"Past: Box Cricket","past_vr":"Past: VR Gaming",
        "feat_ai":"Feature: AI Analysis","mem_num":"Membership WTP",
        "dist_num":"Distance Tolerance","digital_num":"Digital Spend",
        "gender_num":"Gender","edu_num":"Education",
    }
    if feat_imp:
        fi_series = pd.Series(feat_imp).sort_values(ascending=True)
        fi_labels = [feat_names_map.get(k,k) for k in fi_series.index]
        colors = [PRIMARY if v>=fi_series.quantile(0.75) else ACCENT if v>=fi_series.quantile(0.50) else "#555" for v in fi_series.values]
        fig_fi = go.Figure(go.Bar(x=fi_series.values, y=fi_labels, orientation="h",
                                   marker_color=colors,
                                   text=[f"{v:.4f}" for v in fi_series.values], textposition="outside"))
        fig_fi.update_layout(height=520, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(showgrid=False,color="#aaa",title="Importance Score"),
                             yaxis=dict(color="#ccc"), font=dict(color="#ccc"), margin=dict(t=10,b=20,l=0,r=80))
        st.plotly_chart(fig_fi, use_container_width=True)
        top3 = [feat_names_map.get(k,k) for k in list(pd.Series(feat_imp).sort_values(ascending=False).head(3).index)]
        st.markdown(f"""<div class="insight-box">📌 <strong>Top 3 predictors of conversion:</strong> {top3[0]}, {top3[1]}, {top3[2]}.</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Top High-Probability Leads (from Survey Data)</div>', unsafe_allow_html=True)
    try:
        rf_model   = models["rf_classifier"]
        scaler_clf = models["scaler_clf"]
        clf_feats  = models["clf_features"]
        valid_mask = df_enc["pod_conversion_binary"].notna()
        df_score   = df_enc[valid_mask].copy()
        X_score    = df_score[clf_feats].fillna(df_score[clf_feats].median())
        probs      = rf_model.predict_proba(scaler_clf.transform(X_score))[:,1]
        df_score["conversion_probability"] = probs
        df_score["lead_grade"] = pd.cut(probs, bins=[0,0.45,0.65,0.80,1.0], labels=["Cold","Warm","Hot","Very Hot"])
        display_cols = ["respondent_id","true_segment","city_tier","income_bracket","cricket_role","conversion_probability","lead_grade"]
        avail_d = [c for c in display_cols if c in df_score.columns]
        top_leads = df_score.sort_values("conversion_probability",ascending=False)[avail_d].head(30)
        top_leads["conversion_probability"] = top_leads["conversion_probability"].round(3)
        st.dataframe(top_leads, use_container_width=True, hide_index=True)
        grade_counts = df_score["lead_grade"].value_counts()
        c1,c2,c3,c4 = st.columns(4)
        for col,grade,color in zip([c1,c2,c3,c4],["Very Hot","Hot","Warm","Cold"],[DANGER,ACCENT,PRIMARY,"#888"]):
            col.markdown(f"""<div class="metric-card"><div class="val" style="color:{color}">{int(grade_counts.get(grade,0))}</div>
            <div class="lbl">{grade} Leads</div></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not score leads: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

DISCOUNT_MAP = {
    "Rising Star":           ("Student / U-18 discount + free first session", "WhatsApp groups, school coaches"),
    "Elite Competitor":      ("Buy-5-get-1 + AI coaching bundle at 15% off",  "Coach network, BCCI academies"),
    "Corporate Cricket Fan": ("Corporate group package + employer tie-up",     "LinkedIn, HR managers"),
    "Recreational Player":   ("Weekend off-peak discount + referral offer",    "Instagram, Google Maps"),
    "Sceptic / Disengaged":  ("Free trial session — no commitment needed",     "YouTube content, re-targeting ads"),
}

def page_clustering():
    st.title("👥 Clustering — Customer Personas")
    st.markdown("K-Means segmentation to discover natural customer groups for personalised offers.")
    st.markdown("---")

    results = models.get("all_results", {})
    clust   = results.get("clustering", {})
    if not clust:
        st.error("Clustering results not found."); return

    persona_map = models.get("persona_map", {})
    k_vals = list(range(2, 9))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Elbow Curve (Inertia)</div>', unsafe_allow_html=True)
        fig_elbow = go.Figure(go.Scatter(x=k_vals, y=clust["inertias"], mode="lines+markers",
                                          line=dict(color=PRIMARY,width=2.5), marker=dict(color=ACCENT,size=8)))
        fig_elbow.add_vline(x=clust["best_k"], line_color=DANGER, line_dash="dash",
                             annotation_text=f"Chosen k={clust['best_k']}", annotation_font_color=DANGER)
        fig_elbow.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(title="k",color="#ccc"), yaxis=dict(title="Inertia",color="#aaa",showgrid=False),
                                font=dict(color="#ccc"), margin=dict(t=35,b=40,l=0,r=0))
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Silhouette Score by k</div>', unsafe_allow_html=True)
        fig_sil = go.Figure(go.Scatter(x=k_vals, y=clust["silhouettes"], mode="lines+markers",
                                        line=dict(color=SECONDARY,width=2.5), marker=dict(color=ACCENT,size=8)))
        best_k_sil = k_vals[clust["silhouettes"].index(max(clust["silhouettes"]))]
        fig_sil.add_vline(x=best_k_sil, line_color=PRIMARY, line_dash="dash",
                           annotation_text=f"Best k={best_k_sil}", annotation_font_color=PRIMARY)
        fig_sil.update_layout(height=280, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title="k",color="#ccc"), yaxis=dict(title="Silhouette",color="#aaa",showgrid=False),
                              font=dict(color="#ccc"), margin=dict(t=35,b=40,l=0,r=0))
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Visualisation — PCA 2D Projection</div>', unsafe_allow_html=True)
    if "cluster" in df_enc.columns:
        X_c  = get_cluster_features(df_enc)
        X_cs = models["scaler_clust"].transform(X_c)
        pca  = PCA(n_components=2, random_state=42)
        pcs  = pca.fit_transform(X_cs)
        pca_df = pd.DataFrame({"PC1":pcs[:,0],"PC2":pcs[:,1],
                                "Cluster":df_enc["cluster"].astype(str),
                                "Persona":df_enc["cluster"].map(persona_map)})
        fig_pca = go.Figure()
        for c_id in sorted(pca_df["Cluster"].unique()):
            sub = pca_df[pca_df["Cluster"]==c_id]
            persona = persona_map.get(int(c_id), f"Cluster {c_id}")
            fig_pca.add_trace(go.Scatter(x=sub["PC1"],y=sub["PC2"],mode="markers",
                                          name=f"C{c_id}: {persona}",
                                          marker=dict(color=CLUSTER_COLORS[int(c_id)%len(CLUSTER_COLORS)],size=5,opacity=0.65)))
        fig_pca.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",color="#ccc",showgrid=True,gridcolor="#333"),
                              yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",color="#aaa",showgrid=True,gridcolor="#333"),
                              legend=dict(font=dict(color="#ccc",size=10)),
                              font=dict(color="#ccc"), margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Customer Personas & Recommended Strategies</div>', unsafe_allow_html=True)
    profiles = models.get("cluster_profiles", [])
    for prof in profiles:
        c_id    = prof["cluster"]
        persona = persona_map.get(c_id, f"Cluster {c_id}")
        disc, channel = DISCOUNT_MAP.get(persona, ("Free trial","Social media"))
        color   = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
        conv_pct = prof.get("conversion_rate",0) * 100
        avg_spend = prof.get("avg_spend",0)
        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {color}55;border-left:4px solid {color};
                    border-radius:12px;padding:1rem 1.2rem;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div><span style="background:{color}22;color:{color};font-size:0.7rem;font-weight:600;padding:2px 8px;border-radius:10px">CLUSTER {c_id}</span>
              <span style="font-size:1.05rem;font-weight:600;color:#fff;margin-left:10px">{persona}</span></div>
            <span style="font-size:0.8rem;color:#aaa">{prof['size']:,} respondents</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">
            <div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{conv_pct:.0f}%</div>
              <div style="font-size:0.72rem;color:#aaa">Conversion rate</div></div>
            <div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">₹{avg_spend:.0f}</div>
              <div style="font-size:0.72rem;color:#aaa">Avg monthly spend</div></div>
            <div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_pod_interest']:.1f}/5</div>
              <div style="font-size:0.72rem;color:#aaa">Pod interest</div></div>
            <div style="text-align:center"><div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_income']:.1f}/5</div>
              <div style="font-size:0.72rem;color:#aaa">Income score</div></div>
          </div>
          <div style="font-size:0.8rem;color:#ccc">
            <span style="color:{color};font-weight:600">Best offer:</span> {disc}<br>
            <span style="color:{color};font-weight:600">Reach via:</span> {channel}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Cluster Radar — Feature Profiles</div>', unsafe_allow_html=True)
    radar_feats  = ["age_num","income_num","role_num","practice_num","data_importance","pod_interest","spend_num","tech_num"]
    radar_labels = ["Age","Income","Cricket Role","Practice Days","Data Importance","Pod Interest","Rec Spend","Tech Adoption"]
    if "cluster" in df_enc.columns:
        avail_r = [c for c in radar_feats if c in df_enc.columns]
        avail_l = [radar_labels[radar_feats.index(c)] for c in avail_r]
        cluster_means = df_enc.groupby("cluster")[avail_r].mean()
        norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)
        fig_radar = go.Figure()
        for c_id, row in norm.iterrows():
            persona = persona_map.get(int(c_id), f"Cluster {c_id}")
            color   = CLUSTER_COLORS[int(c_id) % len(CLUSTER_COLORS)]
            vals    = row.tolist() + [row.tolist()[0]]
            lbls    = avail_l + [avail_l[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=lbls, fill="toself",
                                                 name=f"C{c_id}: {persona}",
                                                 line=dict(color=color), fillcolor=color, opacity=0.25))
        fig_radar.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)",
                                            radialaxis=dict(visible=True,range=[0,1],color="#666"),
                                            angularaxis=dict(color="#ccc")),
                                paper_bgcolor="rgba(0,0,0,0)", height=440,
                                legend=dict(font=dict(color="#ccc",size=10)),
                                font=dict(color="#ccc"), margin=dict(t=20,b=20,l=0,r=0))
        st.plotly_chart(fig_radar, use_container_width=True)

    if "cluster" in df_enc.columns and "true_segment" in df_enc.columns:
        st.markdown('<div class="section-header">Cluster Purity vs True Segment (Validation)</div>', unsafe_allow_html=True)
        ct = pd.crosstab(df_enc["cluster"].map(lambda x: persona_map.get(x,str(x))),
                         df_enc["true_segment"], normalize="index").round(3) * 100
        st.dataframe(ct.style.background_gradient(cmap="Greens",axis=1), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════

LABEL_MAP = {
    "feat_ai":"AI Analysis","feat_bowlingmachine":"Bowling Machine","feat_batspeed":"Bat Speed",
    "feat_footwork":"Footwork","feat_videoreplay":"Video Replay","feat_leaderboard":"Leaderboard",
    "feat_progressreport":"Progress Report","feat_appbooking":"App Booking",
    "addon_smartbat":"Smart Bat","addon_wearables":"Wearables Kit","addon_aicoaching":"AI Coaching Sub",
    "addon_highlights":"Video Highlights","addon_fitness":"Fitness Program","addon_merch":"Merchandise",
    "use_academy":"Uses Academy","use_boxcricket":"Box Cricket","use_bowlingmachine":"Bowling Machine Net",
    "use_gym":"Gym","use_mobilegame":"Mobile Cricket Game","use_videoanalysis":"Video Analysis",
    "act_gym":"Gym","act_yoga":"Yoga","act_othersport":"Other Sport","act_swimming":"Swimming",
    "act_videogaming":"Video Gaming","act_running":"Running",
    "stream_hotstar":"Hotstar","stream_netflix":"Netflix","stream_jiocinema":"JioCinema",
    "stream_prime":"Amazon Prime","stream_youtube":"YouTube","stream_sonyliv":"Sony LIV",
    "past_boxcricket":"Paid Box Cricket","past_trampoline":"Trampoline Park","past_vr":"VR Gaming",
    "past_bowling":"Bowling Alley","past_gokarting":"Go-Karting","past_fitclass":"Fitness Class",
    "past_academy":"Paid Academy","past_golf":"Golf Range",
    "frust_nodata":"Frustrated: No Data","frust_coachattention":"Frustrated: Coach Inattentive",
    "bar_price":"Barrier: Price","bar_aidistrust":"Barrier: AI Distrust",
    "disc_freetrial":"Prefers Free Trial","disc_referral":"Prefers Referral Disc",
    "disc_student":"Prefers Student Disc","disc_family":"Prefers Family Bundle",
    "hh_self":"Self User","hh_child":"Child User",
    "brand_mrf":"Brand MRF","brand_sg":"Brand SG","brand_decathlon":"Brand Decathlon",
}

def _pretty(items_str):
    parts = [s.strip() for s in str(items_str).split(",")]
    return " + ".join([LABEL_MAP.get(p,p) for p in parts])

def page_association():
    st.title("🔗 Association Rule Mining")
    st.markdown("Discovering what products, features and behaviours go together — using **Apriori** algorithm.")
    st.markdown("---")

    rules = models.get("assoc_rules")
    if rules is None or (hasattr(rules,"__len__") and len(rules)==0):
        st.warning("No association rules found."); return
    if not isinstance(rules, pd.DataFrame):
        rules = pd.DataFrame(rules)
    if rules.empty:
        st.warning("No rules found."); return

    st.markdown('<div class="section-header">Filter Rules</div>', unsafe_allow_html=True)
    col_f1,col_f2,col_f3 = st.columns(3)
    min_sup  = col_f1.slider("Min Support",    0.01, 0.30, 0.05, 0.01)
    min_conf = col_f2.slider("Min Confidence", 0.30, 0.95, 0.50, 0.05)
    min_lift = col_f3.slider("Min Lift",        1.0,  5.0,  1.2,  0.1)

    filtered = rules[(rules["support"]>=min_sup)&(rules["confidence"]>=min_conf)&(rules["lift"]>=min_lift)].copy()
    filtered = filtered.sort_values("lift",ascending=False).reset_index(drop=True)
    st.markdown(f"**{len(filtered)} rules** match current filters.")
    st.markdown("---")

    if len(filtered) > 0:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total Rules",    len(filtered))
        c2.metric("Max Lift",       f"{filtered['lift'].max():.3f}")
        c3.metric("Max Confidence", f"{filtered['confidence'].max():.3f}")
        c4.metric("Max Support",    f"{filtered['support'].max():.3f}")

        st.markdown('<div class="section-header">Top Rules — Ranked by Lift</div>', unsafe_allow_html=True)
        display_rules = filtered.head(30).copy()
        display_rules["antecedents"] = display_rules["antecedents"].apply(_pretty)
        display_rules["consequents"] = display_rules["consequents"].apply(_pretty)
        display_rules = display_rules[["antecedents","consequents","support","confidence","lift"]].round(4)
        display_rules.columns = ["IF (Antecedent)","THEN (Consequent)","Support","Confidence","Lift"]
        st.dataframe(display_rules, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Support vs Confidence — Bubble = Lift</div>', unsafe_allow_html=True)
        plot_df = filtered.head(60).copy()
        plot_df["ant_p"] = plot_df["antecedents"].apply(lambda x: _pretty(x)[:40])
        plot_df["con_p"] = plot_df["consequents"].apply(lambda x: _pretty(x)[:40])
        fig_sc = go.Figure(go.Scatter(x=plot_df["support"], y=plot_df["confidence"], mode="markers",
                                       marker=dict(size=np.clip(plot_df["lift"]*6,6,30), color=plot_df["lift"],
                                                   colorscale=[[0,"#333"],[0.5,SECONDARY],[1,PRIMARY]],
                                                   showscale=True, colorbar=dict(title="Lift",tickfont=dict(color="#ccc")), opacity=0.8),
                                       text=[f"IF: {a}<br>THEN: {c}<br>Lift={l:.3f}"
                                             for a,c,l in zip(plot_df["ant_p"],plot_df["con_p"],plot_df["lift"])],
                                       hoverinfo="text"))
        fig_sc.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(title="Support",color="#ccc",showgrid=True,gridcolor="#333"),
                             yaxis=dict(title="Confidence",color="#aaa",showgrid=True,gridcolor="#333"),
                             font=dict(color="#ccc"), margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("---")
        st.markdown('<div class="section-header">Top 10 Rules by Lift</div>', unsafe_allow_html=True)
        top10 = filtered.head(10).copy()
        top10["rule_label"] = [f"{_pretty(str(a))[:30]}… → {_pretty(str(c))[:20]}…"
                               for a,c in zip(top10["antecedents"],top10["consequents"])]
        top10_s = top10.sort_values("lift",ascending=True)
        fig_bar = go.Figure(go.Bar(y=top10_s["rule_label"], x=top10_s["lift"], orientation="h",
                                    marker_color=PRIMARY,
                                    text=[f"{v:.3f}" for v in top10_s["lift"]], textposition="outside"))
        fig_bar.update_layout(height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(showgrid=False,color="#aaa",title="Lift"),
                              yaxis=dict(color="#ccc"), font=dict(color="#ccc"), margin=dict(t=10,b=20,l=0,r=80))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No rules match current filter. Try lowering thresholds.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION
# ══════════════════════════════════════════════════════════════════════════════

COEF_LABELS = {
    "income_num":"Income","role_num":"Cricket Role","pod_interest":"Pod Interest",
    "practice_num":"Practice Days","data_importance":"Data Importance",
    "tech_num":"Tech Adoption","past_exp_count":"Past Experiences",
    "addon_count":"Add-on Interest","feat_count":"Feature Interest",
    "spend_num":"Current Rec Spend","nps_score":"NPS Score",
    "city_num":"City Tier","age_num":"Age","digital_num":"Digital Spend",
    "mem_num":"Membership WTP","frust_count":"Frustration Count",
    "dist_num":"Distance Tolerance","barrier_count":"Barrier Count",
}

def page_regression():
    st.title("📈 Regression — How Much Will They Spend?")
    st.markdown("Predicting **monthly spend** per customer using Ridge and Linear Regression.")
    st.markdown("---")

    results = models.get("all_results", {})
    reg = results.get("regression", {})
    if not reg:
        st.error("Regression results not found."); return

    ridge = reg["ridge"]; lr = reg["lr"]
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Ridge R²",    f"{ridge['r2']:.4f}")
    c2.metric("Ridge RMSE",  f"₹{ridge['rmse']:.0f}")
    c3.metric("Ridge MAE",   f"₹{ridge['mae']:.0f}")
    c4.metric("LinReg R²",   f"{lr['r2']:.4f}")
    c5.metric("LinReg RMSE", f"₹{lr['rmse']:.0f}")
    c6.metric("LinReg MAE",  f"₹{lr['mae']:.0f}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted Spend</div>', unsafe_allow_html=True)
        y_test = ridge["y_test"]; y_pred = ridge["y_pred"]
        max_val = max(max(y_test), max(y_pred))
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                                     marker=dict(color=PRIMARY,size=4,opacity=0.5), name="Predictions"))
        fig_ap.add_trace(go.Scatter(x=[0,max_val], y=[0,max_val], mode="lines",
                                     line=dict(color=DANGER,dash="dash",width=1.5), name="Perfect fit"))
        fig_ap.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(title="Actual ₹",color="#ccc",showgrid=True,gridcolor="#333"),
                             yaxis=dict(title="Predicted ₹",color="#aaa",showgrid=True,gridcolor="#333"),
                             legend=dict(font=dict(color="#ccc")),
                             font=dict(color="#ccc"), margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig_ap, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Residual Plot</div>', unsafe_allow_html=True)
        residuals = np.array(y_pred) - np.array(y_test)
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                                      marker=dict(color=SECONDARY,size=4,opacity=0.5)))
        fig_res.add_hline(y=0, line_color=DANGER, line_dash="dash", line_width=1.5)
        fig_res.update_layout(height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                              xaxis=dict(title="Predicted ₹",color="#ccc",showgrid=True,gridcolor="#333"),
                              yaxis=dict(title="Residual",color="#aaa",showgrid=True,gridcolor="#333"),
                              font=dict(color="#ccc"), margin=dict(t=10,b=40,l=0,r=0))
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
    coef_imp = reg.get("coef_imp", {})
    if coef_imp:
        st.markdown('<div class="section-header">Feature Coefficients — Ridge (|abs| importance)</div>', unsafe_allow_html=True)
        ci_series = pd.Series(coef_imp).sort_values(ascending=True)
        ci_labels = [COEF_LABELS.get(k,k) for k in ci_series.index]
        fig_coef = go.Figure(go.Bar(x=ci_series.values, y=ci_labels, orientation="h",
                                     marker_color=[PRIMARY if v>=ci_series.quantile(0.7) else ACCENT if v>=ci_series.quantile(0.4) else "#555" for v in ci_series.values],
                                     text=[f"{v:.2f}" for v in ci_series.values], textposition="outside"))
        fig_coef.update_layout(height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                               xaxis=dict(showgrid=False,color="#aaa",title="|Coefficient|"),
                               yaxis=dict(color="#ccc"), font=dict(color="#ccc"), margin=dict(t=10,b=20,l=0,r=80))
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")
    if "cluster" in df_enc.columns and "realistic_monthly_spend" in df_enc.columns:
        st.markdown('<div class="section-header">Predicted Spend Distribution by Cluster</div>', unsafe_allow_html=True)
        persona_map = models.get("persona_map", {})
        spend_by_cluster = df_enc.groupby("cluster")["realistic_monthly_spend"].agg(["mean","median","std"]).reset_index()
        spend_by_cluster["persona"] = spend_by_cluster["cluster"].map(persona_map)
        spend_by_cluster = spend_by_cluster.sort_values("mean",ascending=False)
        fig_cs = go.Figure()
        for _,row in spend_by_cluster.iterrows():
            c = CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)]
            fig_cs.add_trace(go.Bar(name=str(row["persona"]), x=[str(row["persona"])], y=[row["mean"]],
                                     error_y=dict(type="data",array=[row["std"]],visible=True,color="#666"),
                                     marker_color=c, text=f"₹{row['mean']:.0f}", textposition="outside"))
        fig_cs.update_layout(height=340, barmode="group",
                             plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                             xaxis=dict(color="#ccc"), yaxis=dict(title="Avg Monthly Spend (₹)",showgrid=False,color="#aaa"),
                             legend=dict(font=dict(color="#ccc")), font=dict(color="#ccc"), margin=dict(t=10,b=60,l=0,r=0))
        st.plotly_chart(fig_cs, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Interactive Revenue Forecaster</div>', unsafe_allow_html=True)
    fc1,fc2,fc3,fc4 = st.columns(4)
    n_pods       = fc1.slider("Number of pods",          1,  50,  3)
    sessions_day = fc2.slider("Sessions/pod/day",        5,  30, 15)
    avg_price    = fc3.slider("Avg price per session ₹", 100,500,220)
    occupancy    = fc4.slider("Occupancy rate %",        20, 100, 60)
    days_month   = 26
    monthly_sessions = n_pods * sessions_day * days_month * (occupancy / 100)
    monthly_revenue  = monthly_sessions * avg_price
    annual_revenue   = monthly_revenue * 12
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Monthly Sessions",   f"{int(monthly_sessions):,}")
    m2.metric("Monthly Revenue",    f"₹{monthly_revenue:,.0f}")
    m3.metric("Annual Revenue",     f"₹{annual_revenue:,.0f}")
    m4.metric("Revenue per Pod/mo", f"₹{monthly_revenue/max(n_pods,1):,.0f}")
    months   = list(range(1,13))
    rev_proj = [monthly_revenue * (1+0.02)**m for m in range(12)]
    fig_proj = go.Figure(go.Scatter(x=[f"M{m}" for m in months], y=rev_proj, mode="lines+markers",
                                     line=dict(color=PRIMARY,width=2.5), marker=dict(color=ACCENT,size=7),
                                     fill="tozeroy", fillcolor=f"{PRIMARY}22",
                                     text=[f"₹{v:,.0f}" for v in rev_proj], textposition="top center"))
    fig_proj.update_layout(title="12-month revenue projection (2% MoM growth assumed)",
                           height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#ccc"), yaxis=dict(title="Revenue ₹",color="#aaa",showgrid=False),
                           font=dict(color="#ccc"), margin=dict(t=40,b=20,l=0,r=0))
    st.plotly_chart(fig_proj, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW CUSTOMER PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def _action(prob, persona):
    if prob >= 0.75:   return "🔥 HOT LEAD — Offer free trial session immediately"
    elif prob >= 0.55: return "⭐ WARM — Send ₹100 first-session discount voucher"
    elif prob >= 0.40: return "🔄 NURTURE — Share AI demo video + feature explainer"
    else:              return "📧 COLD — Add to monthly newsletter only"

def _channel(persona):
    ch = {
        "Rising Star":           "WhatsApp / school coach network",
        "Elite Competitor":      "Cricket academy / coach referral",
        "Corporate Cricket Fan": "LinkedIn / employer HR tie-up",
        "Recreational Player":   "Instagram / Google Maps SEO",
        "Sceptic / Disengaged":  "YouTube retargeting / free trial push",
    }
    return ch.get(str(persona), "Social media / digital ads")

def score_new_data(new_df):
    defaults = {
        "age_group":"19-25","gender":"Male","city_tier":"Metro",
        "occupation":"Salaried private","income_bracket":"40K-75K",
        "education":"Bachelors","cricket_role":"Regular","practice_days":"1-2",
        "data_importance":3,"fantasy_cricket":"Occasional","pod_interest":3,
        "monthly_rec_spend":"501-1000","psm_too_cheap":"100-149",
        "psm_reasonable":"200-249","psm_expensive":"400-499","psm_too_expensive":"600-799",
        "membership_wtp":"500-999","digital_spend":"201-500","food_delivery_freq":"1-2/week",
        "influence_source":"Self","tech_adoption":"Early majority",
        "distance_tolerance":"Up to 5km","preferred_timeslot":"Evening","nps_score":7,
    }
    for col, val in defaults.items():
        if col not in new_df.columns:
            new_df[col] = val
    for col in MULTI_SELECT_COLS:
        if col not in new_df.columns:
            new_df[col] = 0

    df_e = encode(new_df)
    clust_feats = models.get("cluster_features", CLUSTERING_FEATURES)
    avail_c = [f for f in clust_feats if f in df_e.columns]
    X_c  = df_e[avail_c].fillna(df_e[avail_c].median())
    X_cs = models["scaler_clust"].transform(X_c)
    clusters  = models["kmeans"].predict(X_cs)
    pmap      = models.get("persona_map", {})
    personas  = [pmap.get(c, f"Cluster {c}") for c in clusters]

    clf_feats = models.get("clf_features", CLASSIFICATION_FEATURES)
    avail_f   = [f for f in clf_feats if f in df_e.columns]
    X_clf     = df_e[avail_f].fillna(df_e[avail_f].median())
    probs     = models["rf_classifier"].predict_proba(models["scaler_clf"].transform(X_clf))[:,1]

    reg_feats = models.get("reg_features", REGRESSION_FEATURES)
    avail_r   = [f for f in reg_feats if f in df_e.columns]
    X_reg     = df_e[avail_r].fillna(df_e[avail_r].median())
    spend_pred = np.clip(models["ridge_regressor"].predict(models["scaler_reg"].transform(X_reg)), 0, 5000)

    out = new_df.copy()
    out["conversion_probability"] = np.round(probs, 3)
    out["lead_grade"]             = pd.cut(probs, bins=[0,0.40,0.55,0.75,1.01], labels=["Cold","Warm","Hot","Very Hot"])
    out["predicted_cluster"]      = clusters
    out["persona"]                = personas
    out["predicted_spend_pm"]     = np.round(spend_pred, 0)
    out["recommended_action"]     = [_action(p, per) for p, per in zip(probs, personas)]
    out["recommended_channel"]    = [_channel(per) for per in personas]
    return out

def page_predictor():
    st.title("🚀 New Customer Predictor")
    st.markdown("Upload new survey responses → instantly score each lead with **conversion probability**, **persona**, **predicted spend**, and **recommended action**.")
    st.markdown("---")

    st.markdown('<div class="section-header">Step 1 — Download Sample Template</div>', unsafe_allow_html=True)
    sample_cols = ["age_group","gender","city_tier","occupation","income_bracket","education",
                   "cricket_role","practice_days","data_importance","fantasy_cricket","pod_interest",
                   "monthly_rec_spend","psm_too_cheap","psm_reasonable","psm_expensive","psm_too_expensive",
                   "membership_wtp","digital_spend","food_delivery_freq","influence_source","tech_adoption",
                   "distance_tolerance","preferred_timeslot","nps_score"]
    sample_rows = [
        ["19-25","Male","Metro","Salaried private","40K-75K","Bachelors","Regular","3-4",4,"Active",4,"1001-2500","100-149","200-249","400-499","600-799","1000-1999","201-500","3-4/week","Friends/Teammates","Early majority","Up to 5km","Evening",8],
        ["15-18","Male","Tier 1","School student","Below 20K","Up to 10th","Competitive","5-6",5,"Occasional",5,"501-1000","50-99","150-199","300-399","500-599","500-999","1-200","1-2/week","Parents/Family","Early adopter","Up to 3km","Early morning",9],
        ["26-35","Female","Metro","Professional","75K-150K","Masters+","Fan only","0",2,"Not interested",2,"1001-2500","150-199","250-299","500-599","800+","Would not subscribe","501-1000","1-2/week","Social media","Late majority","Any distance","Evening",6],
    ]
    sample_df = pd.DataFrame(sample_rows, columns=sample_cols)
    buf = io.StringIO(); sample_df.to_csv(buf, index=False)
    st.download_button("⬇️ Download sample template CSV", data=buf.getvalue(),
                       file_name="new_customers_template.csv", mime="text/csv")
    st.markdown("---")

    st.markdown('<div class="section-header">Step 2 — Upload New Customer Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV with new respondents", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows × {new_df.shape[1]} columns")
            st.dataframe(new_df.head(5), use_container_width=True)
            with st.spinner("Scoring all leads..."):
                scored = score_new_data(new_df.copy())
            st.markdown("---")
            st.markdown('<div class="section-header">Scoring Results</div>', unsafe_allow_html=True)
            grade_counts = scored["lead_grade"].value_counts()
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Total Scored",  len(scored))
            c2.metric("🔥 Very Hot",   int(grade_counts.get("Very Hot",0)))
            c3.metric("⭐ Hot",        int(grade_counts.get("Hot",0)))
            c4.metric("🔄 Warm",       int(grade_counts.get("Warm",0)))
            c5.metric("📧 Cold",       int(grade_counts.get("Cold",0)))
            col_a, col_b = st.columns(2)
            with col_a:
                fig_g = go.Figure(go.Pie(labels=grade_counts.index, values=grade_counts.values, hole=0.45,
                                          marker_colors=[DANGER,ACCENT,PRIMARY,"#555"]))
                fig_g.update_layout(title="Lead Grade Distribution", height=300,
                                    paper_bgcolor="rgba(0,0,0,0)", legend=dict(font=dict(color="#ccc")),
                                    font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
                st.plotly_chart(fig_g, use_container_width=True)
            with col_b:
                persona_counts = scored["persona"].value_counts()
                fig_p = go.Figure(go.Bar(x=persona_counts.values, y=persona_counts.index, orientation="h",
                                          marker_color=SECONDARY, text=persona_counts.values, textposition="outside"))
                fig_p.update_layout(title="Personas Detected", height=300,
                                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                    xaxis=dict(showgrid=False,color="#aaa"), yaxis=dict(color="#ccc"),
                                    font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=60))
                st.plotly_chart(fig_p, use_container_width=True)

            display_scored = scored[[c for c in scored.columns if c in new_df.columns or
                                      c in ["conversion_probability","lead_grade","persona","predicted_spend_pm","recommended_action","recommended_channel"]]]
            st.markdown("**Full scored results:**")
            st.dataframe(display_scored, use_container_width=True, hide_index=True)
            out_buf = io.StringIO(); display_scored.to_csv(out_buf, index=False)
            st.download_button("⬇️ Download scored leads CSV", data=out_buf.getvalue(),
                               file_name="scored_leads.csv", mime="text/csv")

            st.markdown('<div class="section-header">Predicted Monthly Spend by Lead Grade</div>', unsafe_allow_html=True)
            spend_by_grade = scored.groupby("lead_grade")["predicted_spend_pm"].mean().reset_index()
            fig_sg = go.Figure(go.Bar(x=spend_by_grade["lead_grade"].astype(str),
                                       y=spend_by_grade["predicted_spend_pm"],
                                       marker_color=[DANGER,ACCENT,PRIMARY,"#555"],
                                       text=[f"₹{v:.0f}" for v in spend_by_grade["predicted_spend_pm"]],
                                       textposition="outside"))
            fig_sg.update_layout(height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                                  xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa",title="Avg ₹/month"),
                                  font=dict(color="#ccc"), margin=dict(t=10,b=20,l=0,r=0))
            st.plotly_chart(fig_sg, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)
    else:
        st.info("👆 Upload a CSV file above to start scoring new customers.")
        st.markdown("""<div class="insight-box">
        📌 <strong>How this works:</strong><br>
        1. Your CSV is encoded using the same pipeline as training data<br>
        2. Missing columns are filled with training-set defaults automatically<br>
        3. Random Forest scores conversion probability (0.0 – 1.0)<br>
        4. K-Means assigns each person to a persona cluster<br>
        5. Ridge Regression predicts their monthly spend<br>
        6. Each row gets a recommended marketing action and channel
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════

if   "Home"           in page: page_home()
elif "Descriptive"    in page: page_descriptive()
elif "Diagnostic"     in page: page_diagnostic()
elif "Classification" in page: page_classification()
elif "Clustering"     in page: page_clustering()
elif "Association"    in page: page_association()
elif "Regression"     in page: page_regression()
elif "Predictor"      in page: page_predictor()
