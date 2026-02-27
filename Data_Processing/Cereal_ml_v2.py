"""
Cereal Price Prediction — ML Pipeline v2
Split: 70% Train | 15% Validation | 15% Test (temporal order preserved)
Metrics reported separately for Validation and Test sets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from scipy.stats import norm

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 65)
print("  CEREAL PRICE PREDICTION — ML PIPELINE v2")
print("  Split: 70% Train | 15% Validation | 15% Test")
print("=" * 65)

df = pd.read_csv('Agri-Hotri_Price_Prediction/Data/Cereal_Final.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)
df = df.dropna(subset=['modal_price'])

le_season       = LabelEncoder()
df['season_enc'] = le_season.fit_transform(df['season'].fillna('unknown'))

NUMERIC_FEATURES = [
    'arrivals', 'temperature', 'rainfall', 'solar_radiation', 'wind_speed',
    'rainfall_30d', 'rainfall_60d', 'temp_14d_avg',
    'msp', 'price_to_msp_ratio', 'below_msp_flag',
    'arrivals_lag_7', 'arrivals_pct_change_7', 'arrival_rolling_14',
    'supply_stress_index', 'rolling_median_7',
    'Petrol_price', 'Diesel_price', 'Month_Num',
    'diesel_lag_7', 'diesel_lag_30', 'petrol_lag_7', 'petrol_lag_30',
    'diesel_pct_change_30', 'zero_arrival_flag', 'market_closed_flag',
    'season_enc', 'Year'
]
TARGET      = 'modal_price'
COMMODITIES = df['commodity'].unique()

# ─────────────────────────────────────────────
# 2. HELPERS
# ─────────────────────────────────────────────
def prepare_data(data):
    X = data[NUMERIC_FEATURES].copy()
    y = data[TARGET].values
    imputer = SimpleImputer(strategy='median')
    X_imp   = imputer.fit_transform(X)
    return X_imp, y, imputer

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

def build_models():
    return {
        'Ridge Regression':        Ridge(alpha=10),
        'Random Forest':           RandomForestRegressor(n_estimators=200, max_depth=12,
                                                         min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Extra Trees':             ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                                       min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':       GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                             max_depth=5, random_state=42),
        'Hist Gradient Boosting':  HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                  max_depth=6, random_state=42),
    }

# ─────────────────────────────────────────────
# 3. TRAIN / VALIDATE / TEST
# ─────────────────────────────────────────────
all_val_results  = {}
all_test_results = {}
all_preds        = {}
feature_imps     = {}
all_data         = {}

for commodity in COMMODITIES:
    print(f"\n{'-'*65}")
    print(f"  Commodity: {commodity.upper()}")
    print(f"{'-'*65}")

    sub = df[df['commodity'] == commodity].copy().reset_index(drop=True)
    all_data[commodity] = sub
    X, y, imputer = prepare_data(sub)
    n = len(X)

    t1 = int(n * 0.70)
    t2 = int(n * 0.85)

    X_train, y_train = X[:t1],    y[:t1]
    X_val,   y_val   = X[t1:t2],  y[t1:t2]
    X_test,  y_test  = X[t2:],    y[t2:]

    print(f"  Rows -> Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    val_list   = []
    test_list  = []
    preds_dict = {'y_val': y_val, 'y_test': y_test}
    models     = build_models()
    fitted     = {}

    for name, model in models.items():
        if name == 'Ridge Regression':
            model.fit(X_train_s, y_train)
            p_val  = model.predict(X_val_s)
            p_test = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            p_val  = model.predict(X_val)
            p_test = model.predict(X_test)

        vm = evaluate(y_val,  p_val,  name, 'VALIDATION')
        tm = evaluate(y_test, p_test, name, 'TEST')

        val_list.append({'model': name, **vm})
        test_list.append({'model': name, **tm})
        preds_dict[f'val_{name}']  = p_val
        preds_dict[f'test_{name}'] = p_test
        fitted[name] = model

    all_val_results[commodity]  = pd.DataFrame(val_list)
    all_test_results[commodity] = pd.DataFrame(test_list)
    all_preds[commodity]        = preds_dict

    fi_model = fitted['Extra Trees']
    feature_imps[commodity] = pd.Series(
        fi_model.feature_importances_, index=NUMERIC_FEATURES
    ).sort_values(ascending=False).head(15)

# ─────────────────────────────────────────────
# 4. STYLING
# ─────────────────────────────────────────────
COLORS = {'arhar (tur dal)': '#E84B4B', 'rice': '#F5C842', 'wheat': '#4DA6E8'}
LABEL  = {'arhar (tur dal)': 'Arhar (Tur Dal)', 'rice': 'Rice', 'wheat': 'Wheat'}
MODEL_COLORS = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f']
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
vparts = ax.violinplot([df[df['commodity']==c]['modal_price'].dropna().values for c in COMMODITIES],
                       positions=[1,2,3], showmedians=True, showextrema=True)
for pc, c in zip(vparts['bodies'], COMMODITIES):
    pc.set_facecolor(COLORS[c]); pc.set_alpha(0.75)
vparts['cmedians'].set_color('white')
for k in ['cbars','cmins','cmaxes']: vparts[k].set_color('grey')
ax.set_xticks([1,2,3]); ax.set_xticklabels([LABEL[c] for c in COMMODITIES], color='white')
ax.set_ylabel('Modal Price (₹)'); ax.set_title('Price Distribution by Commodity', fontsize=13)

ax2 = fig1.add_subplot(gs1[0, 2]); ax2.set_facecolor(BG_PANEL)
counts = df['commodity'].value_counts()
wedges, texts, auts = ax2.pie(counts.values, labels=[LABEL[c] for c in counts.index],
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

corr_cols = ['modal_price','arrivals','temperature','rainfall',
             'msp','price_to_msp_ratio','supply_stress_index',
             'rolling_median_7','Diesel_price']
for i, c in enumerate(COMMODITIES):
    ax4 = fig1.add_subplot(gs1[2, i]); ax4.set_facecolor(BG_PANEL)
    sub = df[df['commodity']==c][corr_cols].dropna()
    corr = sub.corr()[['modal_price']].drop('modal_price')
    sns.heatmap(corr, ax=ax4, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, cbar=False, annot_kws={'size':8,'color':'white'})
    ax4.set_title(f'{LABEL[c]}\nCorr. with Price', color='white', fontsize=10)
    ax4.tick_params(colors='white', labelsize=8)
    for _, sp in ax4.spines.items(): sp.set_color(SPINE_C)

plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig1_eda.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
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

plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig2_msp_analysis.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig2_msp_analysis.png")

# ══════════════════════════════════════════════════════════
# FIG 3 — Validation vs Test Metrics: Grouped Bar Charts
# ══════════════════════════════════════════════════════════
fig3, axes = plt.subplots(3, 3, figsize=(24, 20))
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
        ax.set_xticklabels(vr['model'], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(xlabel)
        ax.set_title(f'{LABEL[c]} — {metric}', fontsize=12)
        if metric == 'R2':
            ax.set_ylim(max(0, min(vr[metric].min(), tr[metric].min()) - 0.08), 1.06)
        if row == 0 and col_idx == 0:
            ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.7, fontsize=10)
        for bar in list(b1) + list(b2):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x()+bar.get_width()/2, h + (0.002 if metric=='R2' else h*0.01),
                        f'{h:.3f}' if metric=='R2' else f'{h:.0f}',
                        ha='center', va='bottom', color='white', fontsize=6, fontweight='bold')

plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig3_val_vs_test_metrics.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
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
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig4_validation_predictions.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
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
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig5_test_predictions.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig5_test_predictions.png")

# ══════════════════════════════════════════════════════════
# FIG 6 — R² Drift: Validation → Test
# ══════════════════════════════════════════════════════════
fig6, axes = plt.subplots(1, 3, figsize=(22, 7))
fig6.patch.set_facecolor(BG_DARK)
fig6.suptitle('R² Drift: Validation → Test  (gap = potential overfitting)',
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
    ax.set_xticks(xi); ax.set_xticklabels(list(vr.index), rotation=20, ha='right', fontsize=8.5)
    ax.set_ylabel('R² Score')
    ax.set_ylim(max(0, min(vr.min(), tr.min()) - 0.1), 1.05)
    ax.set_title(LABEL[c], color=COLORS[c], fontsize=14, fontweight='bold')
    ax.legend(facecolor=BG_PANEL, labelcolor='white', framealpha=0.6, fontsize=8)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig6_r2_drift.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig6_r2_drift.png")

# ══════════════════════════════════════════════════════════
# FIG 7 — Feature Importance & Test Residuals
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
    ax.set_title(f'{LABEL[c]} — Top Feature Importances', fontsize=13)
    ax.set_xlabel('Importance Score')

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
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig7_features_residuals.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig7_features_residuals.png")

# ══════════════════════════════════════════════════════════
# FIG 8 — Final Leaderboard (Test R²)
# ══════════════════════════════════════════════════════════
fig8, axes = plt.subplots(1, 3, figsize=(22, 7))
fig8.patch.set_facecolor(BG_DARK)
fig8.suptitle('Test Set Leaderboard — R² Score Across Cereal Commodities',
              fontsize=18, fontweight='bold', color='white', y=1.02)

for ax, c in zip(axes, COMMODITIES):
    style_ax(ax)
    res  = all_test_results[c].sort_values('R2', ascending=True)
    bcol = [MODEL_COLORS[list(all_test_results[c]['model']).index(m)] for m in res['model']]
    bars = ax.barh(res['model'], res['R2'], color=bcol, edgecolor='none', height=0.6, alpha=0.9)
    ax.set_xlim(max(0, res['R2'].min()-0.08), 1.08)
    ax.set_title(LABEL[c], color=COLORS[c], fontsize=15, fontweight='bold')
    ax.set_xlabel('R² Score (Test)')
    for bar, val in zip(bars, res['R2']):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', color='white', fontsize=9, fontweight='bold')
    best_idx = list(res['R2']).index(res['R2'].max())
    bars[best_idx].set_edgecolor('gold'); bars[best_idx].set_linewidth(2.5)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Cereal/cereal_fig8_leaderboard.png', dpi=150,
            bbox_inches='tight', facecolor=BG_DARK)
plt.close(); print("   cereal_fig8_leaderboard.png")

# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*82)
print("  FINAL SUMMARY — BEST MODEL PER COMMODITY")
print("="*82)
print(f"{'Commodity':<18} {'Best Model':<28} {'Val R²':>8} {'Test R²':>8} "
      f"{'Val MAE':>9} {'Test MAE':>9} {'Val MAPE':>9} {'Test MAPE':>9}")
print("-"*82)
for c in COMMODITIES:
    vb = all_val_results[c].sort_values('R2', ascending=False).iloc[0]
    tb = all_test_results[c].sort_values('R2', ascending=False).iloc[0]
    print(f"{LABEL[c]:<18} {vb['model']:<28} {vb['R2']:>8.4f} {tb['R2']:>8.4f} "
          f"{vb['MAE']:>9.1f} {tb['MAE']:>9.1f} {vb['MAPE']:>8.2f}% {tb['MAPE']:>8.2f}%")
print("="*82)
print("\n All outputs saved to ./Agri-Hotri_Price_Prediction/outputs/Cereal/")