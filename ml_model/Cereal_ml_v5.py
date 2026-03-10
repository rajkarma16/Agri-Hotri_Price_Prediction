"""
Cereal Price Prediction — ML Pipeline v5
════════════════════════════════════════════════════════════
Changes from v4:
  1. Hyperparameter Tuning via RandomizedSearchCV + TimeSeriesSplit(n_splits=5)
       - ALL models tuned on training data using time-series-aware CV
       - Scaled models (Ridge, Lasso) wrapped in sklearn Pipeline so
         the StandardScaler is re-fit inside each CV fold — zero leakage
       - n_iter=10 random combinations tried per model per commodity
       - Best params printed after each model
  2. Best-model feature importance (new Fig 10)
       - After tuning, identifies best model per commodity (highest test R²)
       - Tree models  -> Gini impurity decrease (.feature_importances_)
       - Linear models -> |standardised coefficient| (.coef_)
       - HistGradientBoosting (no native importance) -> falls back to Extra Trees
  3. Fixed Cereal feature_imps to use Extra Trees (was incorrectly using |Lasso coef|)
  4. best_params_log dict — stores every model's best hyperparams per commodity
Split: 70% Train | 15% Validation | 15% Test (temporal order preserved)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from matplotlib.patches import Patch
from scipy.stats import norm

# ---------------------------------------------
# 1. LOAD & PREPROCESS
# ---------------------------------------------
print("=" * 65)
print("  CEREAL PRICE PREDICTION — ML PIPELINE v5")
print("  Split: 70% Train | 15% Validation | 15% Test")
print("  Tuning: RandomizedSearchCV + TimeSeriesSplit(n_splits=5)")
print("=" * 65)

df = pd.read_csv('Agri-Hotri_Price_Prediction/Data/Cereal_Final_v3.csv',
                 parse_dates=['date'])
df = df.sort_values(['commodity', 'date']).reset_index(drop=True)
df = df.dropna(subset=['modal_price'])

print(f"  Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Dates   : {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"  Commod. : {df['commodity'].unique().tolist()}")

# -- Full feature list ----------------------------------------
NUMERIC_FEATURES_WANTED = [
    'arrivals', 'temperature', 'rainfall', 'solar_radiation', 'wind_speed',
    'rainfall_7d', 'rainfall_15d', 'temp_7d_avg',
    'rainfall_shock_7d', 'temp_deviation_14d',
    'rainfall_30d', 'rainfall_60d', 'temp_14d_avg',
    'rainfall_shock_30d',
    'msp', 'price_to_msp_ratio', 'below_msp_flag',
    'price_above_msp', 'msp_yearly_growth',
    'Petrol_price', 'Diesel_price',
    'diesel_lag_7', 'diesel_lag_30', 'petrol_lag_7', 'petrol_lag_30',
    'diesel_pct_change_30', 'diesel_pct_change_7', 'petrol_pct_change_7',
    'Year', 'Month_Num', 'DayOfYear', 'WeekOfYear', 'season_enc',
    'zero_arrival_flag', 'market_closed_flag',
    'arrivals_lag_7', 'arrivals_pct_change_7',
    'arrival_lag_3',
    'arrival_rolling_7', 'arrival_rolling_14', 'arrival_rolling_30',
    'arrival_shock', 'supply_stress_index',
    'supply_tightness', 'supply_shock_7v30',
    'price_lag_1', 'price_lag_3', 'price_lag_7', 'price_lag_14', 'price_lag_30',
    'price_rolling_mean_14', 'price_rolling_mean_30',
    'price_volatility_7', 'price_volatility_14', 'price_volatility_30',
    'price_pct_change_3', 'price_pct_change_7', 'price_pct_change_30',
]

NUMERIC_FEATURES = [f for f in NUMERIC_FEATURES_WANTED if f in df.columns]
missing = [f for f in NUMERIC_FEATURES_WANTED if f not in df.columns]

print(f"\n  Features used   : {len(NUMERIC_FEATURES)}")
if missing:
    print(f"   Not in CSV (skipped): {missing}")
    print("    -> Run Cereal_Final_v3.py first to generate all features")
else:
    print("   All expected features present in CSV")

TARGET      = 'modal_price'
COMMODITIES = df['commodity'].unique()

# ---------------------------------------------
# 2. HELPERS
# ---------------------------------------------
def prepare_data(data):
    X = data[NUMERIC_FEATURES].copy()
    y = data[TARGET].values
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"   Warning: {nan_count} NaNs found — filling with column median")
        X = X.fillna(X.median())
    return X.values, y

def metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def evaluate(y_true, y_pred, name, split_label):
    m = metrics(y_true, y_pred)
    print(f"  [{split_label:>10}] {name:32s} | MAE={m['MAE']:8.2f} | "
          f"RMSE={m['RMSE']:8.2f} | R²={m['R2']:.4f} | MAPE={m['MAPE']:.2f}%")
    return m

def get_feature_importance(model, feature_names):
    """
    Extract feature importances from a fitted model.
    Returns (pd.Series or None, label_string).
    - Tree ensembles  -> .feature_importances_  (Gini impurity decrease)
    - Linear models   -> |.coef_|               (standardised coefficient magnitude)
    - HistGradBoost   -> None  (no native importance; caller falls back to Extra Trees)
    """
    if hasattr(model, 'feature_importances_'):
        return pd.Series(model.feature_importances_, index=feature_names), 'Gini Importance'
    elif hasattr(model, 'coef_'):
        return pd.Series(np.abs(model.coef_), index=feature_names), '|Coefficient| (std. features)'
    return None, None

# Linear models require scaled data — Pipeline handles scaling inside CV folds
SCALED_MODELS = {'Ridge Regression', 'Lasso Regression'}

def build_models():
    """Return base estimators with sensible starting hyperparameters."""
    return {
        'Ridge Regression':
            Ridge(alpha=50, random_state=42),

        'Lasso Regression':
            Lasso(alpha=40, max_iter=2000, random_state=42),

        'Extra Trees':
            ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                min_samples_leaf=5, random_state=42, n_jobs=-1),

        'Gradient Boosting':
            GradientBoostingRegressor(n_estimators=500, learning_rate=0.02,
                                      max_depth=2, subsample=0.7,
                                      min_samples_leaf=20, random_state=42),

        'Hist Gradient Boosting':
            HistGradientBoostingRegressor(max_iter=500, learning_rate=0.02,
                                          max_depth=3, min_samples_leaf=30,
                                          l2_regularization=1.0, random_state=42),

        'XGBoost':
            XGBRegressor(n_estimators=800, learning_rate=0.02, max_depth=5,
                         min_child_weight=3, gamma=0.1, subsample=0.8,
                         colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                         random_state=42, n_jobs=-1),

        'Random Forest':
            RandomForestRegressor(n_estimators=600, max_depth=8,
                                  min_samples_split=10, min_samples_leaf=5,
                                  max_features='sqrt', bootstrap=True,
                                  random_state=42, n_jobs=-1),
    }

# ---------------------------------------------
# HYPERPARAMETER SEARCH SPACES
# ---------------------------------------------
# Note: For SCALED_MODELS the keys will be prefixed with 'model__'
# automatically in the training loop (Pipeline convention).
PARAM_GRIDS = {
    'Ridge Regression': {
        'alpha': [0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000],
    },
    'Lasso Regression': {
        'alpha':    [0.01, 0.1, 1, 5, 10, 40, 100, 200],
        'max_iter': [1000, 2000, 5000],
    },
    'Extra Trees': {
        'n_estimators':   [100, 200, 300],
        'max_depth':      [6, 8, 10, 12, None],
        'min_samples_leaf': [2, 4, 6, 10],
        'max_features':   ['sqrt', 0.5, 0.7],
    },
    'Gradient Boosting': {
        'n_estimators':     [200, 400, 600],
        'learning_rate':    [0.01, 0.02, 0.05, 0.1],
        'max_depth':        [2, 3, 4],
        'subsample':        [0.6, 0.7, 0.8],
        'min_samples_leaf': [10, 20, 30],
    },
    'Hist Gradient Boosting': {
        'max_iter':         [200, 400, 600],
        'learning_rate':    [0.01, 0.02, 0.05, 0.1],
        'max_depth':        [3, 4, 5, 6],
        'min_samples_leaf': [15, 25, 40],
        'l2_regularization': [0.0, 0.5, 1.0, 2.0],
    },
    'XGBoost': {
        'n_estimators':    [300, 500, 700],
        'learning_rate':   [0.01, 0.02, 0.05],
        'max_depth':       [3, 4, 5, 6],
        'min_child_weight': [1, 3, 5],
        'subsample':       [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha':       [0.0, 0.1, 0.5],
        'reg_lambda':      [0.5, 1.0, 2.0],
    },
    'Random Forest': {
        'n_estimators':     [200, 400, 600],
        'max_depth':        [6, 8, 10, None],
        'min_samples_leaf': [3, 5, 8, 12],
        'max_features':     ['sqrt', 0.5, 0.7],
        'min_samples_split': [5, 10, 15],
    },
}

N_ITER = 10   # random combinations to try per model per commodity
              # increase for better tuning (slower); decrease to speed up

# ---------------------------------------------
# 3. TRAIN / VALIDATE / TEST  with Tuning
# ---------------------------------------------
print(f"\n  Hyperparameter search: RandomizedSearchCV  "
      f"n_iter={N_ITER}  cv=TimeSeriesSplit(5)")
print("  NOTE: This section may take 20-40 min depending on hardware.\n")

tscv = TimeSeriesSplit(n_splits=5)

all_val_results  = {}
all_test_results = {}
all_preds        = {}
feature_imps     = {}
lasso_coefs      = {}
best_model_imps  = {}   # feature importance from best test model
best_model_name  = {}   # name of best test model
best_params_log  = {}   # best hyperparams per commodity per model
all_data         = {}

for commodity in COMMODITIES:
    print(f"\n{'-'*65}")
    print(f"  Commodity: {commodity.upper()}")
    print(f"{'-'*65}")

    sub = df[df['commodity'] == commodity].copy().sort_values('date').reset_index(drop=True)
    all_data[commodity] = sub
    X, y = prepare_data(sub)
    n    = len(X)

    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X_train, y_train = X[:t1],   y[:t1]
    X_val,   y_val   = X[t1:t2], y[t1:t2]
    X_test,  y_test  = X[t2:],   y[t2:]

    print(f"  Rows -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    val_list   = []
    test_list  = []
    preds_dict = {'y_val': y_val, 'y_test': y_test}
    models     = build_models()
    fitted     = {}   # stores core estimators (unwrapped from Pipeline)
    bp_log     = {}

    for name, base_model in models.items():
        print(f"\n  - [{name}] — searching {N_ITER} hyperparameter combinations …")
        params = PARAM_GRIDS.get(name, {})

        if name in SCALED_MODELS:
            # -- Pipeline: scaler is re-fit inside every CV fold — no leakage --
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model',  clone(base_model))
            ])
            # Prefix keys with 'model__' for Pipeline routing
            search_params = {f'model__{k}': v for k, v in params.items()}
            search = RandomizedSearchCV(
                pipe, search_params,
                n_iter=N_ITER, cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=42, n_jobs=-1, verbose=0, refit=True
            )
            search.fit(X_train, y_train)
            best_est   = search.best_estimator_          # fitted Pipeline
            core_model = best_est.named_steps['model']   # Ridge / Lasso
            p_val  = best_est.predict(X_val)
            p_test = best_est.predict(X_test)

        else:
            # -- Tree / boosting models — no scaling needed --
            search = RandomizedSearchCV(
                clone(base_model), params,
                n_iter=N_ITER, cv=tscv,
                scoring='neg_mean_absolute_error',
                random_state=42, n_jobs=-1, verbose=0, refit=True
            )
            search.fit(X_train, y_train)
            best_est   = search.best_estimator_
            core_model = best_est
            p_val  = best_est.predict(X_val)
            p_test = best_est.predict(X_test)

        bp_log[name] = search.best_params_
        print(f"    CV MAE = {-search.best_score_:,.1f}  |  best params: {search.best_params_}")

        vm = evaluate(y_val,  p_val,  name, 'VALIDATION')
        tm = evaluate(y_test, p_test, name, 'TEST')

        val_list.append({'model': name, **vm})
        test_list.append({'model': name, **tm})
        preds_dict[f'val_{name}']  = p_val
        preds_dict[f'test_{name}'] = p_test
        fitted[name] = core_model   # unwrapped core estimator

    all_val_results[commodity]  = pd.DataFrame(val_list)
    all_test_results[commodity] = pd.DataFrame(test_list)
    all_preds[commodity]        = preds_dict
    best_params_log[commodity]  = bp_log

    # -- Feature importance from Extra Trees (consistent with Veg pipeline) --
    feature_imps[commodity] = pd.Series(
        fitted['Extra Trees'].feature_importances_, index=NUMERIC_FEATURES
    ).sort_values(ascending=False).head(15)

    # -- Lasso coefficients (signed — positive ↑ price, negative ↓ price) --
    lasso_coefs[commodity] = pd.Series(
        fitted['Lasso Regression'].coef_, index=NUMERIC_FEATURES
    )

    # -- Best model feature importance ------------------------------------
    best_c = all_test_results[commodity].sort_values('R2', ascending=False).iloc[0]['model']
    fi, _  = get_feature_importance(fitted[best_c], NUMERIC_FEATURES)
    if fi is None:
        # HistGradientBoosting has no native importance -> use Extra Trees
        fi, _ = get_feature_importance(fitted['Extra Trees'], NUMERIC_FEATURES)
        print(f"  Note: {best_c} has no native importance — showing Extra Trees importance")
    best_model_imps[commodity] = fi.sort_values(ascending=False).head(20)
    best_model_name[commodity] = best_c

# ---------------------------------------------
# 4. STYLING
# ---------------------------------------------
COLORS = {'arhar (tur dal)': '#E84B4B', 'rice': '#F5C842', 'wheat': '#4DA6E8'}
LABEL  = {'arhar (tur dal)': 'Arhar (Tur Dal)', 'rice': 'Rice', 'wheat': 'Wheat'}
MODEL_COLORS = ['#4e79a7', '#e15759', '#f28e2b', '#76b7b2', '#59a14f', '#b07aa1', '#ff9da7']
BG_DARK  = '#0f1117'
BG_PANEL = '#1a1d2e'
SPINE_C  = '#333344'
VAL_COL  = '#A78BFA'
TEST_COL = '#34D399'

def style_ax(ax):
    ax.set_facecolor(BG_PANEL)
    ax.tick_params(colors='white', labelsize=9)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines[['top','right']].set_visible(False)
    for sp in ax.spines.values(): sp.set_color(SPINE_C)

print("\nGenerating visualizations...")

# ══════════════════════════════════════════════════════════
# FIG 1 — EDA Overview
# ══════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(22, 18))
fig1.patch.set_facecolor(BG_DARK)
gs1  = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.5, wspace=0.38)
fig1.suptitle('Cereal Price Dataset — Exploratory Data Analysis',
              fontsize=21, fontweight='bold', color='white', y=0.99)

ax = fig1.add_subplot(gs1[0, :2]); style_ax(ax)
vparts = ax.violinplot(
    [df[df['commodity']==c]['modal_price'].dropna().values for c in COMMODITIES],
    positions=[1,2,3], showmedians=True, showextrema=True)
for pc, c in zip(vparts['bodies'], COMMODITIES):
    pc.set_facecolor(COLORS[c]); pc.set_alpha(0.75)
vparts['cmedians'].set_color('white')
for k in ['cbars','cmins','cmaxes']: vparts[k].set_color('grey')
ax.set_xticks([1,2,3])
ax.set_xticklabels([LABEL[c] for c in COMMODITIES], color='white')
ax.set_ylabel('Modal Price (₹)'); ax.set_title('Price Distribution by Commodity', fontsize=13)

ax2 = fig1.add_subplot(gs1[0, 2]); ax2.set_facecolor(BG_PANEL)
counts = df['commodity'].value_counts()
wedges, texts, auts = ax2.pie(
    counts.values, labels=[LABEL[c] for c in counts.index],
    autopct='%1.1f%%', colors=[COLORS[c] for c in counts.index],
    startangle=90, pctdistance=0.75, textprops={'color':'white','fontsize':10})
for a in auts: a.set_fontsize(9)
ax2.set_title('Records per Commodity', color='white', fontsize=13)

ax3 = fig1.add_subplot(gs1[1, :]); style_ax(ax3)
for c in COMMODITIES:
    s = df[df['commodity']==c].set_index('date')['modal_price'].resample('W').mean()
    ax3.plot(s.index, s.values, color=COLORS[c], label=LABEL[c], alpha=0.85, linewidth=1.5)
msp_vals = df.groupby('commodity')['msp'].median()
for c in COMMODITIES:
    ax3.axhline(msp_vals[c], color=COLORS[c], linestyle=':', linewidth=1, alpha=0.5,
                label=f'{LABEL[c]} MSP')
ax3.set_ylabel('Avg Weekly Price (₹)'); ax3.set_xlabel('Date')
ax3.set_title('Modal Price Over Time — Weekly Average  (dotted = MSP floor)', fontsize=13)
ax3.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, ncol=2, fontsize=8)

CORR_COLS = ['modal_price', 'price_lag_1', 'price_lag_7', 'price_rolling_median_7',
             'price_volatility_7', 'msp', 'price_to_msp_ratio',
             'price_above_msp', 'supply_stress_index']
CORR_COLS = [c for c in CORR_COLS if c in df.columns]

for i, c in enumerate(COMMODITIES):
    ax4 = fig1.add_subplot(gs1[2, i]); ax4.set_facecolor(BG_PANEL)
    sub  = df[df['commodity']==c][CORR_COLS].dropna()
    corr = sub.corr()[['modal_price']].drop('modal_price')
    sns.heatmap(corr, ax=ax4, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, cbar=False,
                annot_kws={'size':8,'color':'white'})
    ax4.set_title(f'{LABEL[c]}\nCorr. with Price', color='white', fontsize=10)
    ax4.tick_params(colors='white', labelsize=7)
    for _, sp in ax4.spines.items(): sp.set_color(SPINE_C)

plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig1_eda.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig1_eda.png")

# ══════════════════════════════════════════════════════════
# FIG 2 — MSP Analysis
# ══════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(22, 14))
fig2.patch.set_facecolor(BG_DARK)
gs2  = gridspec.GridSpec(2, 3, figure=fig2, hspace=0.45, wspace=0.38)
fig2.suptitle('MSP & Market Dynamics Analysis', fontsize=20, fontweight='bold',
              color='white', y=0.99)

for i, c in enumerate(COMMODITIES):
    sub = all_data[c]; col = COLORS[c]

    ax = fig2.add_subplot(gs2[0, i]); style_ax(ax)
    monthly     = sub.set_index('date')['modal_price'].resample('ME').mean()
    msp_monthly = sub.set_index('date')['msp'].resample('ME').first()
    ax.plot(monthly.index, monthly.values, color=col, linewidth=1.4, label='Market Price')
    ax.plot(msp_monthly.index, msp_monthly.values, color='white', linewidth=1.2,
            linestyle='--', label='MSP', alpha=0.7)
    ax.fill_between(monthly.index, monthly.values, msp_monthly.values,
                    where=(monthly.values < msp_monthly.values),
                    color='red', alpha=0.3, label='Below MSP')
    ax.fill_between(monthly.index, monthly.values, msp_monthly.values,
                    where=(monthly.values >= msp_monthly.values),
                    color=col, alpha=0.15, label='Above MSP')
    ax.set_title(f'{LABEL[c]} — Price vs MSP', fontsize=12)
    ax.set_ylabel('Price (₹)'); ax.set_xlabel('Date')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', fontsize=7, framealpha=0.6)

    ax2 = fig2.add_subplot(gs2[1, i]); style_ax(ax2)
    season_colors = {'kharif':'#F5A623','rabi':'#4DA6E8','zaid':'#A8E063','unknown':'#aaa'}
    for s in sub['season'].dropna().unique():
        vals = sub[sub['season']==s]['price_to_msp_ratio'].dropna().values
        if len(vals) > 10:
            ax2.hist(vals, bins=30, alpha=0.65, color=season_colors.get(s,'grey'),
                     label=s.capitalize(), edgecolor='none', density=True)
    ax2.axvline(1.0, color='white', linestyle='--', linewidth=1.5, alpha=0.8,
                label='MSP = Market Price')
    ax2.set_title(f'{LABEL[c]} — Price-to-MSP by Season', fontsize=11)
    ax2.set_xlabel('Price / MSP'); ax2.set_ylabel('Density')
    ax2.legend(facecolor=BG_PANEL, labelcolor='white', fontsize=8, framealpha=0.6)

plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig2_msp_analysis.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig2_msp_analysis.png")

# ══════════════════════════════════════════════════════════
# FIG 3 — Validation vs Test Metrics: Grouped Bar Charts
# ══════════════════════════════════════════════════════════
fig3, axes = plt.subplots(3, 3, figsize=(26, 22))
fig3.patch.set_facecolor(BG_DARK)
fig3.suptitle('Validation vs Test Metrics — All Models & Commodities',
              fontsize=20, fontweight='bold', color='white', y=0.99)

x     = np.arange(len(all_val_results[COMMODITIES[0]]))
width = 0.38

for row, c in enumerate(COMMODITIES):
    vr = all_val_results[c]
    tr = all_test_results[c]

    for col_idx, (metric, xlabel) in enumerate([('R2','R² Score'),('MAE','MAE (₹)'),('MAPE','MAPE (%)')]):
        ax = axes[row, col_idx]; style_ax(ax)
        b1 = ax.bar(x - width/2, vr[metric], width, label='Validation',
                    color=VAL_COL, alpha=0.85, edgecolor='none')
        b2 = ax.bar(x + width/2, tr[metric], width, label='Test',
                    color=TEST_COL, alpha=0.85, edgecolor='none')
        ax.set_xticks(x)
        ax.set_xticklabels(vr['model'], rotation=25, ha='right', fontsize=7.5)
        ax.set_ylabel(xlabel)
        ax.set_title(f'{LABEL[c]} — {metric}', fontsize=12)
        if metric == 'R2':
            ax.set_ylim(max(0, min(vr[metric].min(), tr[metric].min()) - 0.08), 1.06)
        if row == 0 and col_idx == 0:
            ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.7, fontsize=10)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2,
                        h + (0.002 if metric=='R2' else h*0.01),
                        f'{h:.3f}' if metric=='R2' else f'{h:.0f}',
                        ha='center', va='bottom', color='white', fontsize=5.5, fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig3_val_vs_test_metrics.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig3_val_vs_test_metrics.png")

# ══════════════════════════════════════════════════════════
# FIG 4 — Validation Actual vs Predicted
# ══════════════════════════════════════════════════════════
fig4, axes = plt.subplots(3, 2, figsize=(20, 18))
fig4.patch.set_facecolor(BG_DARK)
fig4.suptitle('Validation Set — Actual vs Predicted (Best Model per Commodity)',
              fontsize=18, fontweight='bold', color='white', y=0.99)

for row, c in enumerate(COMMODITIES):
    best_name = all_val_results[c].sort_values('R2', ascending=False).iloc[0]['model']
    y_true    = all_preds[c]['y_val']
    y_pred    = all_preds[c][f'val_{best_name}']
    idx       = np.arange(len(y_true))

    ax = axes[row, 0]; style_ax(ax)
    ax.plot(idx, y_true, color='white', alpha=0.75, linewidth=1.3, label='Actual')
    ax.plot(idx, y_pred, color=VAL_COL, alpha=0.9,  linewidth=1.3, linestyle='--',
            label=f'Predicted ({best_name})')
    ax.fill_between(idx, y_true, y_pred, alpha=0.12, color=VAL_COL)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.02, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            color=VAL_COL, fontsize=11, fontweight='bold', va='top')
    ax.set_title(f'{LABEL[c]} — Validation Forecast', fontsize=13)
    ax.set_xlabel('Validation Samples'); ax.set_ylabel('Price (₹)')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=9)

    ax2 = axes[row, 1]; style_ax(ax2)
    sc = ax2.scatter(y_true, y_pred, c=np.abs(y_true-y_pred), cmap='plasma',
                     alpha=0.5, s=14, edgecolors='none')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo,hi],[lo,hi],'w--', linewidth=1.5, label='Perfect Fit')
    cb = plt.colorbar(sc, ax=ax2); cb.set_label('|Error| (₹)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    mae = mean_absolute_error(y_true, y_pred)
    ax2.text(0.02, 0.95, f'MAE = ₹{mae:.0f}', transform=ax2.transAxes,
             color=VAL_COL, fontsize=11, fontweight='bold', va='top')
    ax2.set_xlabel('Actual Price (₹)'); ax2.set_ylabel('Predicted Price (₹)')
    ax2.set_title(f'{LABEL[c]} — Validation Scatter', fontsize=13)
    ax2.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig4_validation_predictions.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig4_validation_predictions.png")

# ══════════════════════════════════════════════════════════
# FIG 5 — Test Actual vs Predicted
# ══════════════════════════════════════════════════════════
fig5, axes = plt.subplots(3, 2, figsize=(20, 18))
fig5.patch.set_facecolor(BG_DARK)
fig5.suptitle('Test Set — Actual vs Predicted (Best Model per Commodity)',
              fontsize=18, fontweight='bold', color='white', y=0.99)

for row, c in enumerate(COMMODITIES):
    best_name = all_test_results[c].sort_values('R2', ascending=False).iloc[0]['model']
    y_true    = all_preds[c]['y_test']
    y_pred    = all_preds[c][f'test_{best_name}']
    idx       = np.arange(len(y_true))

    ax = axes[row, 0]; style_ax(ax)
    ax.plot(idx, y_true, color='white', alpha=0.75, linewidth=1.3, label='Actual')
    ax.plot(idx, y_pred, color=TEST_COL, alpha=0.9,  linewidth=1.3, linestyle='--',
            label=f'Predicted ({best_name})')
    ax.fill_between(idx, y_true, y_pred, alpha=0.12, color=TEST_COL)
    r2 = r2_score(y_true, y_pred)
    ax.text(0.02, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            color=TEST_COL, fontsize=11, fontweight='bold', va='top')
    ax.set_title(f'{LABEL[c]} — Test Forecast', fontsize=13)
    ax.set_xlabel('Test Samples'); ax.set_ylabel('Price (₹)')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=9)

    ax2 = axes[row, 1]; style_ax(ax2)
    sc = ax2.scatter(y_true, y_pred, c=np.abs(y_true-y_pred), cmap='plasma',
                     alpha=0.5, s=14, edgecolors='none')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo,hi],[lo,hi],'w--', linewidth=1.5, label='Perfect Fit')
    cb = plt.colorbar(sc, ax=ax2); cb.set_label('|Error| (₹)', color='white')
    cb.ax.yaxis.set_tick_params(color='white')
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    mae = mean_absolute_error(y_true, y_pred)
    ax2.text(0.02, 0.95, f'MAE = ₹{mae:.0f}', transform=ax2.transAxes,
             color=TEST_COL, fontsize=11, fontweight='bold', va='top')
    ax2.set_xlabel('Actual Price (₹)'); ax2.set_ylabel('Predicted Price (₹)')
    ax2.set_title(f'{LABEL[c]} — Test Scatter', fontsize=13)
    ax2.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig5_test_predictions.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig5_test_predictions.png")

# ══════════════════════════════════════════════════════════
# FIG 6 — R² Drift: Validation -> Test
# ══════════════════════════════════════════════════════════
fig6, axes = plt.subplots(1, 3, figsize=(22, 7))
fig6.patch.set_facecolor(BG_DARK)
fig6.suptitle('R² Drift: Validation -> Test  (gap = potential overfitting)',
              fontsize=17, fontweight='bold', color='white', y=1.02)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    vr  = all_val_results[c].set_index('model')['R2']
    tr  = all_test_results[c].set_index('model')['R2']
    xi  = np.arange(len(vr))

    ax.plot(xi, vr.values, 'o-', color=VAL_COL,  linewidth=2, markersize=8, label='Validation R²')
    ax.plot(xi, tr.values, 's-', color=TEST_COL,  linewidth=2, markersize=8, label='Test R²')
    ax.fill_between(xi, vr.values, tr.values,
                    where=(vr.values > tr.values), alpha=0.2, color='red',   label='Overfit zone')
    ax.fill_between(xi, vr.values, tr.values,
                    where=(vr.values <= tr.values), alpha=0.2, color='lime', label='Test better')
    ax.set_xticks(xi)
    ax.set_xticklabels(list(vr.index), rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('R² Score')
    ax.set_ylim(max(0, min(vr.min(), tr.min()) - 0.1), 1.05)
    ax.set_title(LABEL[c], color=COLORS[c], fontsize=14, fontweight='bold')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=8)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig6_r2_drift.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig6_r2_drift.png")

# ══════════════════════════════════════════════════════════
# FIG 7 — Feature Importance (Extra Trees) & Test Residuals
# ══════════════════════════════════════════════════════════
fig7, axes = plt.subplots(3, 2, figsize=(20, 18))
fig7.patch.set_facecolor(BG_DARK)
fig7.suptitle('Feature Importance (Extra Trees) & Test Residual Distribution',
              fontsize=18, fontweight='bold', color='white', y=0.99)

for row, c in enumerate(COMMODITIES):
    col = COLORS[c]

    ax = axes[row, 0]; style_ax(ax)
    fi  = feature_imps[c]
    pal = sns.color_palette("Blues_r", len(fi))
    ax.barh(fi.index[::-1], fi.values[::-1], color=pal, edgecolor='none', alpha=0.9)
    ax.set_title(f'{LABEL[c]} — Top 15 Feature Importances (Extra Trees)', fontsize=13)
    ax.set_xlabel('Gini Importance Score')

    ax2 = axes[row, 1]; style_ax(ax2)
    best_name = all_test_results[c].sort_values('R2', ascending=False).iloc[0]['model']
    y_true    = all_preds[c]['y_test']
    y_pred    = all_preds[c][f'test_{best_name}']
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=50, color=col, alpha=0.7, edgecolor='none', density=True)
    mu, sigma = residuals.mean(), residuals.std()
    xr = np.linspace(residuals.min(), residuals.max(), 300)
    ax2.plot(xr, norm.pdf(xr, mu, sigma), 'w-', linewidth=2,
             label=f'Normal fit  μ={mu:.0f}  σ={sigma:.0f}')
    ax2.axvline(0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'{LABEL[c]} — Test Residuals ({best_name})', fontsize=12)
    ax2.set_xlabel('Residual (₹)'); ax2.set_ylabel('Density')
    ax2.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=9)

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig7_features_residuals.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig7_features_residuals.png")

# ══════════════════════════════════════════════════════════
# FIG 8 — Final Leaderboard (Test R²)
# ══════════════════════════════════════════════════════════
fig8, axes = plt.subplots(1, 3, figsize=(22, 8))
fig8.patch.set_facecolor(BG_DARK)
fig8.suptitle('Test Set Leaderboard — R² Score Across Cereal Commodities',
              fontsize=18, fontweight='bold', color='white', y=1.02)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    res  = all_test_results[c].sort_values('R2', ascending=True)
    model_names = list(all_test_results[c]['model'])
    color_map = {m: MODEL_COLORS[i % len(MODEL_COLORS)] for i, m in enumerate(model_names)}
    bcol = [color_map[m] for m in res['model']]
    bars = ax.barh(res['model'], res['R2'], color=bcol, edgecolor='none', height=0.55, alpha=0.9)
    ax.set_xlim(max(0, res['R2'].min()-0.08), 1.08)
    ax.set_title(LABEL[c], color=COLORS[c], fontsize=15, fontweight='bold')
    ax.set_xlabel('R² Score (Test)')
    for bar, val in zip(bars, res['R2']):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', color='white',
                fontsize=9, fontweight='bold')
    best_idx = list(res['R2']).index(res['R2'].max())
    bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig8_leaderboard.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig8_leaderboard.png")

# ══════════════════════════════════════════════════════════
# FIG 9 — Lasso Coefficients  (implicit feature selection)
# ══════════════════════════════════════════════════════════
fig9, axes = plt.subplots(1, 3, figsize=(26, 9))
fig9.patch.set_facecolor(BG_DARK)
fig9.suptitle('Lasso Regression Coefficients — Implicit Feature Selection\n'
              '(coefficients are on standardised features — zero = eliminated by Lasso)',
              fontsize=17, fontweight='bold', color='white', y=1.02)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    coefs   = lasso_coefs[c]
    nonzero = coefs[coefs != 0].sort_values()
    neg     = nonzero[nonzero < 0].head(10)
    pos     = nonzero[nonzero > 0].tail(10)
    display = pd.concat([neg, pos])

    bar_colors = ['#E84B4B' if v < 0 else '#34D399' for v in display.values]
    ax.barh(display.index, display.values, color=bar_colors, edgecolor='none', alpha=0.88)
    ax.axvline(0, color='white', linewidth=1.2, alpha=0.6)

    n_zero    = (coefs == 0).sum()
    n_nonzero = (coefs != 0).sum()
    ax.set_title(f'{LABEL[c]}\nActive: {n_nonzero} features  |  Zeroed out: {n_zero}',
                 color=COLORS[c], fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient Value (on standardised features)')
    ax.legend(handles=[Patch(color='#34D399', label='Positive  (↑ price)'),
                        Patch(color='#E84B4B', label='Negative  (↓ price)')],
              facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=8)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig9_lasso_coefficients.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig9_lasso_coefficients.png")

# ══════════════════════════════════════════════════════════
# FIG 10 — Best Model Feature Importances  ◄ NEW in v5
# Shows top 20 features from the model with highest test R²
# Tree models   -> Gini impurity decrease
# Linear models -> |standardised coefficient|
# HistGradBoost -> falls back to Extra Trees (no native importance)
# ══════════════════════════════════════════════════════════
fig10, axes = plt.subplots(1, 3, figsize=(28, 11))
fig10.patch.set_facecolor(BG_DARK)
fig10.suptitle(
    'Best Model — Top 20 Feature Importances per Commodity  (v5 Hyperparameter-Tuned)\n'
    'Tree models -> Gini impurity decrease  |  Linear models -> |standardised coefficient|',
    fontsize=16, fontweight='bold', color='white', y=1.03
)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    fi    = best_model_imps[c]
    bname = best_model_name[c]
    n     = len(fi)

    is_tree   = any(k in bname for k in ['Forest', 'Trees', 'Boosting', 'XGBoost'])
    imp_label = 'Gini Importance' if is_tree else '|Coefficient| (standardised features)'

    # Gradient: warm yellow-orange-red palette
    pal  = sns.color_palette('YlOrRd_r', n)
    bars = ax.barh(fi.index[::-1], fi.values[::-1], color=pal, edgecolor='none', alpha=0.92)

    max_val = fi.values.max()
    for bar, val in zip(bars, fi.values[::-1]):
        ax.text(val + max_val * 0.012, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', color='white', fontsize=7.5, fontweight='bold')

    ax.set_xlim(0, max_val * 1.22)
    ax.set_title(
        f'{LABEL[c]}\nBest Model: {bname}  (Test R²={all_test_results[c].sort_values("R2",ascending=False).iloc[0]["R2"]:.4f})',
        color=COLORS[c], fontsize=12, fontweight='bold'
    )
    ax.set_xlabel(imp_label)

    # Annotate rank numbers
    for i, (feat, val) in enumerate(zip(fi.index[::-1], fi.values[::-1])):
        ax.text(-max_val * 0.005, i, f'#{i+1}', va='center', ha='right',
                color='#aaaaaa', fontsize=7)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig10_best_model_feature_importance.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig10_best_model_feature_importance.png")

# ══════════════════════════════════════════════════════════
# FIG 11 — Hyperparameter Tuning Summary  ◄ NEW in v5
# Heatmap of best CV MAE per model per commodity
# ══════════════════════════════════════════════════════════
fig11, axes = plt.subplots(1, 3, figsize=(26, 8))
fig11.patch.set_facecolor(BG_DARK)
fig11.suptitle('Hyperparameter Tuning Summary — Best CV MAE & Validation R² per Model\n'
               f'(RandomizedSearchCV  n_iter={N_ITER}  |  TimeSeriesSplit n_splits=5)',
               fontsize=16, fontweight='bold', color='white', y=1.03)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    vr = all_val_results[c].set_index('model')
    tr = all_test_results[c].set_index('model')

    model_list = list(vr.index)
    xi = np.arange(len(model_list))
    w  = 0.38

    b1 = ax.bar(xi - w/2, vr['R2'], w, color=VAL_COL, alpha=0.85,
                edgecolor='none', label='Val R²')
    b2 = ax.bar(xi + w/2, tr['R2'], w, color=TEST_COL, alpha=0.85,
                edgecolor='none', label='Test R²')

    # Gold star on best test model
    best_m = tr['R2'].idxmax()
    bi     = model_list.index(best_m)
    ax.annotate('★', xy=(xi[bi] + w/2, tr.loc[best_m,'R2']),
                fontsize=14, color='gold', ha='center', va='bottom')

    ax.set_xticks(xi)
    ax.set_xticklabels(model_list, rotation=28, ha='right', fontsize=8)
    ax.set_ylabel('R² Score')
    ax.set_ylim(max(0, min(vr['R2'].min(), tr['R2'].min()) - 0.08), 1.10)
    ax.set_title(f'{LABEL[c]}\nBest: {best_m}  (Test R²={tr.loc[best_m,"R2"]:.4f})',
                 color=COLORS[c], fontsize=12, fontweight='bold')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=9)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x()+bar.get_width()/2, h+0.003,
                    f'{h:.3f}', ha='center', va='bottom',
                    color='white', fontsize=6, fontweight='bold')

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig11_tuning_summary.png',
            dpi=150, bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig11_tuning_summary.png")

# ---------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------
print("\n" + "="*88)
print("  FINAL SUMMARY — BEST MODEL PER COMMODITY  (after hyperparameter tuning)")
print("="*88)
print(f"{'Commodity':<18} {'Best Model':<28} {'Val R²':>8} {'Test R²':>8} "
      f"{'Val MAE':>9} {'Test MAE':>9} {'Val MAPE':>9} {'Test MAPE':>9}")
print("-"*88)
for c in COMMODITIES:
    vb = all_val_results[c].sort_values('R2', ascending=False).iloc[0]
    tb = all_test_results[c].sort_values('R2', ascending=False).iloc[0]
    print(f"{LABEL[c]:<18} {vb['model']:<28} {vb['R2']:>8.4f} {tb['R2']:>8.4f} "
          f"{vb['MAE']:>9.1f} {tb['MAE']:>9.1f} {vb['MAPE']:>8.2f}% {tb['MAPE']:>8.2f}%")
print("="*88)

print(f"\n  Lasso Feature Selection Summary:")
print(f"  {'Commodity':<18} {'Active Features':>16} {'Zeroed Out':>12}")
print(f"  {'-'*48}")
for c in COMMODITIES:
    coefs = lasso_coefs[c]
    print(f"  {LABEL[c]:<18} {(coefs != 0).sum():>16} {(coefs == 0).sum():>12}")

print(f"\n  Best Model Feature Importances (Top 5 per Commodity):")
print(f"  {'-'*70}")
for c in COMMODITIES:
    fi    = best_model_imps[c].head(5)
    bname = best_model_name[c]
    print(f"\n  {LABEL[c]}  ->  Best: {bname}")
    for rank, (feat, val) in enumerate(fi.items(), 1):
        print(f"    #{rank:>2}  {feat:<35}  {val:.5f}")

print(f"\n  Hyperparameter Tuning — Best Params per Model:")
print(f"  {'-'*70}")
for c in COMMODITIES:
    print(f"\n  {LABEL[c].upper()}")
    for model_name, bp in best_params_log[c].items():
        # Strip 'model__' prefix for display
        bp_display = {k.replace('model__',''):v for k,v in bp.items()}
        print(f"    {model_name:<28} -> {bp_display}")

print(f"\n  Total features used : {len(NUMERIC_FEATURES)}")
print("   All outputs saved to ./Agri-Hotri_Price_Prediction/outputs/Cereal/")