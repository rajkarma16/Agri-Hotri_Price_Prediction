"""
Robust Machine Learning Pipeline for Vegetable Price Prediction
Dataset: Onion, Potato, Tomato prices (Maharashtra, India)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, HistGradientBoostingRegressor)
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────
print("=" * 60)
print("  VEGETABLE PRICE PREDICTION — ML PIPELINE")
print("=" * 60)

df = pd.read_csv('Agri-Hotri_Price_Prediction/Data/Vegetable_Final.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Drop rows where target is missing
df = df.dropna(subset=['modal_price'])

# Features to use
NUMERIC_FEATURES = [
    'arrivals', 'temperature', 'rainfall', 'solar_radiation', 'wind_speed',
    'rainfall_7d', 'rainfall_15d', 'temp_7d_avg', 'Petrol_price', 'Diesel_price',
    'Month_Num', 'diesel_lag_7', 'diesel_lag_30', 'petrol_lag_7', 'petrol_lag_30',
    'diesel_pct_change_30', 'price_lag_1', 'price_lag_3', 'price_lag_7',
    'price_pct_change_3', 'arrival_lag_3', 'arrival_shock', 'arrival_rolling_7',
    'supply_tightness', 'price_volatility_7', 'Year'
]

TARGET = 'modal_price'
COMMODITIES = df['commodity'].unique()

# ─────────────────────────────────────────────
# 2. HELPER FUNCTIONS
# ─────────────────────────────────────────────
def prepare_data(data):
    X = data[NUMERIC_FEATURES].copy()
    y = data[TARGET].values
    imputer = SimpleImputer(strategy='median')
    X_imp = imputer.fit_transform(X)
    return X_imp, y, imputer

def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    print(f"  {name:30s} | MAE={mae:8.2f} | RMSE={rmse:8.2f} | R²={r2:.4f} | MAPE={mape:.2f}%")
    return {'model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def build_models():
    return {
        'Ridge Regression':          Ridge(alpha=10),
        'Random Forest':             RandomForestRegressor(n_estimators=200, max_depth=12,
                                                           min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Extra Trees':               ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                                         min_samples_leaf=5, random_state=42, n_jobs=-1),
        'Gradient Boosting':         GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                               max_depth=5, random_state=42),
        'Hist Gradient Boosting':    HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                                   max_depth=6, random_state=42),
    }

# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE PER COMMODITY
# ─────────────────────────────────────────────
all_results   = {}
all_preds     = {}
feature_imps  = {}

for commodity in COMMODITIES:
    print(f"\n{'='*60}")
    print(f"  Commodity: {commodity.upper()}")
    print(f"{'='*60}")
    
    sub = df[df['commodity'] == commodity].copy()
    X, y, imputer = prepare_data(sub)
    
    # Temporal split (80/20, preserve time order)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    
    results_list  = []
    preds_dict    = {'y_true': y_test}
    models        = build_models()
    fitted_models = {}
    
    for name, model in models.items():
        if name == 'Ridge Regression':
            model.fit(X_train_s, y_train)
            pred = model.predict(X_test_s)
        elif name == 'Hist Gradient Boosting':
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        
        res = evaluate(y_test, pred, name)
        results_list.append(res)
        preds_dict[name] = pred
        fitted_models[name] = model
    
    all_results[commodity]  = pd.DataFrame(results_list)
    all_preds[commodity]    = preds_dict
    
    # Feature importance from best tree model (Extra Trees for all)
    best_tree_model = fitted_models['Extra Trees']
    feature_imps[commodity] = pd.Series(
        best_tree_model.feature_importances_, index=NUMERIC_FEATURES
    ).sort_values(ascending=False).head(15)

# ─────────────────────────────────────────────
# 4. MEGA VISUALIZATION
# ─────────────────────────────────────────────
print("\nGenerating visualizations...")

COLORS = {
    'onion':  '#E84B4B',
    'potato': '#F5A623',
    'tomato': '#4CAF50',
}
MODEL_COLORS = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']
COMMODITY_DISPLAY = {'onion': 'Onion', 'potato': 'Potato', 'tomato': 'Tomato'}

# ──────────────────────────────────────────────────────────
# FIG 1 — EDA Overview
# ──────────────────────────────────────────────────────────
fig1 = plt.figure(figsize=(22, 16))
fig1.patch.set_facecolor('#0f1117')
gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)
fig1.suptitle('Vegetable Price Dataset — Exploratory Data Analysis',
              fontsize=20, fontweight='bold', color='white', y=0.98)

# 1a. Price distribution per commodity (violin)
ax = fig1.add_subplot(gs1[0, :2])
ax.set_facecolor('#1a1d2e')
data_violin = [df[df['commodity'] == c]['modal_price'].dropna().values for c in COMMODITIES]
parts = ax.violinplot(data_violin, positions=[1,2,3], showmedians=True, showextrema=True)
for i, (pc, c) in enumerate(zip(parts['bodies'], COMMODITIES)):
    pc.set_facecolor(COLORS[c]); pc.set_alpha(0.7)
parts['cmedians'].set_color('white'); parts['cbars'].set_color('grey')
parts['cmins'].set_color('grey');    parts['cmaxes'].set_color('grey')
ax.set_xticks([1,2,3]); ax.set_xticklabels([c.capitalize() for c in COMMODITIES], color='white')
ax.set_ylabel('Modal Price (₹)', color='white'); ax.set_title('Price Distribution by Commodity', color='white', fontsize=13)
ax.tick_params(colors='white'); ax.spines[['top','right']].set_visible(False)
for spine in ax.spines.values(): spine.set_color('#333')

# 1b. Records per commodity
ax2 = fig1.add_subplot(gs1[0, 2])
ax2.set_facecolor('#1a1d2e')
counts = df['commodity'].value_counts()
wedges, texts, autotexts = ax2.pie(counts.values, labels=[c.capitalize() for c in counts.index],
    autopct='%1.1f%%', colors=[COLORS[c] for c in counts.index],
    startangle=90, pctdistance=0.75, textprops={'color':'white', 'fontsize':11})
for at in autotexts: at.set_fontsize(10)
ax2.set_title('Records per Commodity', color='white', fontsize=13)

# 1c. Price over time per commodity
ax3 = fig1.add_subplot(gs1[1, :])
ax3.set_facecolor('#1a1d2e')
for c in COMMODITIES:
    sub = df[df['commodity'] == c].set_index('date')['modal_price'].resample('W').mean()
    ax3.plot(sub.index, sub.values, color=COLORS[c], label=c.capitalize(), alpha=0.85, linewidth=1.4)
ax3.set_ylabel('Avg Weekly Price (₹)', color='white'); ax3.set_xlabel('Date', color='white')
ax3.set_title('Modal Price Over Time (Weekly Average)', color='white', fontsize=13)
ax3.legend(facecolor='#1a1d2e', labelcolor='white', framealpha=0.6)
ax3.tick_params(colors='white'); ax3.spines[['top','right']].set_visible(False)
for spine in ax3.spines.values(): spine.set_color('#333')

# 1d–1f. Correlation heatmap per commodity
for i, c in enumerate(COMMODITIES):
    ax4 = fig1.add_subplot(gs1[2, i])
    ax4.set_facecolor('#1a1d2e')
    sub = df[df['commodity'] == c][['modal_price','arrivals','temperature','rainfall',
                                     'price_lag_1','price_lag_7','supply_tightness',
                                     'price_volatility_7']].dropna()
    corr = sub.corr()[['modal_price']].drop('modal_price')
    sns.heatmap(corr, ax=ax4, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, cbar=False,
                annot_kws={'size':8, 'color':'white'})
    ax4.set_title(f'{c.capitalize()} — Correlation with Price', color='white', fontsize=11)
    ax4.tick_params(colors='white', labelsize=8)
    for _, spine in ax4.spines.items(): spine.set_color('#333')

plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Vegetable/fig1_eda.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("   fig1_eda.png saved")

# ──────────────────────────────────────────────────────────
# FIG 2 — Model Performance Comparison
# ──────────────────────────────────────────────────────────
fig2, axes = plt.subplots(3, 3, figsize=(22, 18))
fig2.patch.set_facecolor('#0f1117')
fig2.suptitle('Model Performance Comparison Across Commodities',
              fontsize=20, fontweight='bold', color='white', y=0.99)

METRICS = ['MAE', 'RMSE', 'R2', 'MAPE']
MODEL_NAMES = list(all_results[COMMODITIES[0]]['model'])

for row, c in enumerate(COMMODITIES):
    res = all_results[c]
    
    # Col 0: MAE bar
    ax = axes[row, 0]; ax.set_facecolor('#1a1d2e')
    bars = ax.barh(res['model'], res['MAE'], color=MODEL_COLORS, edgecolor='none', alpha=0.85)
    ax.set_xlabel('MAE (₹)', color='white'); ax.set_title(f'{c.capitalize()} — MAE', color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=9); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    for bar, val in zip(bars, res['MAE']):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}', va='center', ha='left', color='white', fontsize=8)
    
    # Col 1: R² bar
    ax = axes[row, 1]; ax.set_facecolor('#1a1d2e')
    bars = ax.barh(res['model'], res['R2'], color=MODEL_COLORS, edgecolor='none', alpha=0.85)
    ax.set_xlabel('R² Score', color='white'); ax.set_title(f'{c.capitalize()} — R²', color='white', fontsize=12)
    ax.set_xlim(max(0, res['R2'].min() - 0.05), 1.02)
    ax.tick_params(colors='white', labelsize=9); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    for bar, val in zip(bars, res['R2']):
        ax.text(min(bar.get_width()+0.005, 1.0), bar.get_y()+bar.get_height()/2,
                f'{val:.3f}', va='center', ha='left', color='white', fontsize=8)
    
    # Col 2: MAPE bar
    ax = axes[row, 2]; ax.set_facecolor('#1a1d2e')
    bars = ax.barh(res['model'], res['MAPE'], color=MODEL_COLORS, edgecolor='none', alpha=0.85)
    ax.set_xlabel('MAPE (%)', color='white'); ax.set_title(f'{c.capitalize()} — MAPE', color='white', fontsize=12)
    ax.tick_params(colors='white', labelsize=9); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    for bar, val in zip(bars, res['MAPE']):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f'{val:.1f}%', va='center', ha='left', color='white', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Vegetable/fig2_model_comparison.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("   fig2_model_comparison.png saved")

# ──────────────────────────────────────────────────────────
# FIG 3 — Actual vs Predicted (best model per commodity)
# ──────────────────────────────────────────────────────────
fig3, axes = plt.subplots(3, 2, figsize=(20, 18))
fig3.patch.set_facecolor('#0f1117')
fig3.suptitle('Actual vs Predicted — Best Model (Hist Gradient Boosting) Per Commodity',
              fontsize=18, fontweight='bold', color='white', y=0.99)

for row, c in enumerate(COMMODITIES):
    preds  = all_preds[c]
    y_true = preds['y_true']
    y_pred = preds['Hist Gradient Boosting']
    color  = COLORS[c]
    n      = len(y_true)
    idx    = np.arange(n)
    
    # Time series comparison
    ax = axes[row, 0]; ax.set_facecolor('#1a1d2e')
    ax.plot(idx, y_true, color='white',  alpha=0.7, linewidth=1.2, label='Actual')
    ax.plot(idx, y_pred, color=color,    alpha=0.9, linewidth=1.2, label='Predicted', linestyle='--')
    ax.fill_between(idx, y_true, y_pred, alpha=0.15, color=color)
    ax.set_title(f'{c.capitalize()} — Price Forecast vs Actual', color='white', fontsize=13)
    ax.set_xlabel('Test Samples (chronological)', color='white')
    ax.set_ylabel('Price (₹)', color='white')
    ax.legend(facecolor='#1a1d2e', labelcolor='white', framealpha=0.6)
    ax.tick_params(colors='white'); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    r2 = r2_score(y_true, y_pred)
    ax.text(0.02, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            color=color, fontsize=11, fontweight='bold', va='top')
    
    # Scatter: Actual vs Predicted
    ax2 = axes[row, 1]; ax2.set_facecolor('#1a1d2e')
    sc = ax2.scatter(y_true, y_pred, c=np.abs(y_true-y_pred), cmap='plasma',
                     alpha=0.5, s=12, edgecolors='none')
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo, hi], [lo, hi], 'w--', linewidth=1.5, label='Perfect Fit')
    cb = plt.colorbar(sc, ax=ax2); cb.set_label('|Error| (₹)', color='white')
    cb.ax.yaxis.set_tick_params(color='white'); 
    plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')
    ax2.set_xlabel('Actual Price (₹)', color='white'); ax2.set_ylabel('Predicted Price (₹)', color='white')
    ax2.set_title(f'{c.capitalize()} — Scatter Plot', color='white', fontsize=13)
    mae = mean_absolute_error(y_true, y_pred)
    ax2.text(0.02, 0.95, f'MAE = ₹{mae:.1f}', transform=ax2.transAxes,
             color=color, fontsize=11, fontweight='bold', va='top')
    ax2.legend(facecolor='#1a1d2e', labelcolor='white', framealpha=0.6)
    ax2.tick_params(colors='white'); ax2.spines[['top','right']].set_visible(False)
    for spine in ax2.spines.values(): spine.set_color('#333')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Vegetable/fig3_actual_vs_predicted.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("   fig3_actual_vs_predicted.png saved")

# ──────────────────────────────────────────────────────────
# FIG 4 — Feature Importance & Residuals
# ──────────────────────────────────────────────────────────
fig4, axes = plt.subplots(3, 2, figsize=(20, 18))
fig4.patch.set_facecolor('#0f1117')
fig4.suptitle('Feature Importance (Hist Gradient Boosting) & Residual Analysis',
              fontsize=18, fontweight='bold', color='white', y=0.99)

for row, c in enumerate(COMMODITIES):
    color = COLORS[c]
    
    # Feature importance
    ax = axes[row, 0]; ax.set_facecolor('#1a1d2e')
    fi = feature_imps[c]
    palette = sns.color_palette("Blues_r", len(fi))
    ax.barh(fi.index[::-1], fi.values[::-1], color=palette, edgecolor='none', alpha=0.9)
    ax.set_title(f'{c.capitalize()} — Top Feature Importances', color='white', fontsize=13)
    ax.set_xlabel('Importance Score', color='white')
    ax.tick_params(colors='white', labelsize=9); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    
    # Residual distribution
    ax2 = axes[row, 1]; ax2.set_facecolor('#1a1d2e')
    y_true = all_preds[c]['y_true']
    y_pred = all_preds[c]['Hist Gradient Boosting']
    residuals = y_true - y_pred
    ax2.hist(residuals, bins=50, color=color, alpha=0.7, edgecolor='none', density=True)
    from scipy.stats import norm
    mu, sigma = residuals.mean(), residuals.std()
    xr = np.linspace(residuals.min(), residuals.max(), 200)
    ax2.plot(xr, norm.pdf(xr, mu, sigma), 'w-', linewidth=2, label=f'Normal fit\nμ={mu:.1f}, σ={sigma:.1f}')
    ax2.axvline(0, color='yellow', linestyle='--', linewidth=1.5, alpha=0.8)
    ax2.set_title(f'{c.capitalize()} — Residual Distribution', color='white', fontsize=13)
    ax2.set_xlabel('Residual (₹)', color='white'); ax2.set_ylabel('Density', color='white')
    ax2.legend(facecolor='#1a1d2e', labelcolor='white', framealpha=0.6)
    ax2.tick_params(colors='white'); ax2.spines[['top','right']].set_visible(False)
    for spine in ax2.spines.values(): spine.set_color('#333')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Vegetable/fig4_feature_importance_residuals.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("   fig4_feature_importance_residuals.png saved")

# ──────────────────────────────────────────────────────────
# FIG 5 — Summary Leaderboard
# ──────────────────────────────────────────────────────────
fig5, axes = plt.subplots(1, 3, figsize=(22, 7))
fig5.patch.set_facecolor('#0f1117')
fig5.suptitle('Model Leaderboard — R² Score Across Commodities',
              fontsize=18, fontweight='bold', color='white', y=1.02)

for ax, c in zip(axes, COMMODITIES):
    ax.set_facecolor('#1a1d2e')
    res = all_results[c].sort_values('R2', ascending=True)
    bar_colors = [MODEL_COLORS[list(all_results[c]['model']).index(m)] for m in res['model']]
    bars = ax.barh(res['model'], res['R2'], color=bar_colors, edgecolor='none', height=0.6, alpha=0.9)
    ax.set_xlim(max(0, res['R2'].min()-0.05), 1.05)
    ax.set_title(f'{c.capitalize()}', color=COLORS[c], fontsize=15, fontweight='bold')
    ax.set_xlabel('R² Score', color='white')
    ax.tick_params(colors='white', labelsize=10); ax.spines[['top','right']].set_visible(False)
    for spine in ax.spines.values(): spine.set_color('#333')
    for bar, val in zip(bars, res['R2']):
        ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', ha='left', color='white', fontsize=9, fontweight='bold')
    # Highlight best
    best_idx = res['R2'].idxmax()
    best_bar_pos = list(res.index).index(best_idx)
    bars[best_bar_pos].set_edgecolor('gold'); bars[best_bar_pos].set_linewidth(2)

plt.tight_layout()
plt.savefig('./Agri-Hotri_Price_Prediction/outputs/Vegetable/fig5_leaderboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0f1117')
plt.close()
print("   fig5_leaderboard.png saved")

# ──────────────────────────────────────────────────────────
# PRINT FINAL SUMMARY TABLE
# ──────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  FINAL SUMMARY — BEST MODEL PER COMMODITY (by R²)")
print("="*70)
print(f"{'Commodity':<12} {'Best Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
print("-"*70)
for c in COMMODITIES:
    best = all_results[c].sort_values('R2', ascending=False).iloc[0]
    print(f"{c.capitalize():<12} {best['model']:<25} {best['MAE']:>8.2f} {best['RMSE']:>8.2f} "
          f"{best['R2']:>8.4f} {best['MAPE']:>7.2f}%")
print("="*70)
print("\n All outputs saved to ./Agri-Hotri_Price_Prediction/outputs/Vegetable/")