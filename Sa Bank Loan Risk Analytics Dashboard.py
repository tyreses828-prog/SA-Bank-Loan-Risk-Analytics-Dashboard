# ============================================================
# SA BANK LOAN RISK ANALYTICS
# Full Finance Dataset Analysis + ML Models
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PART 1 — CREATE THE DATASET
# ============================================================
# In real life you would load this with pd.read_csv('loans.csv')
# Here we simulate a realistic SA bank loan dataset

np.random.seed(42)
n = 500

data = {
    'customer_id': [f'CUST_{i:04d}' for i in range(1, n+1)],
    'age': np.random.randint(22, 65, n),
    'annual_income': np.random.normal(450000, 150000, n).clip(80000, 1200000).round(-3),
    'loan_amount': np.random.normal(250000, 100000, n).clip(50000, 800000).round(-3),
    'loan_term_months': np.random.choice([12, 24, 36, 48, 60, 72], n),
    'credit_score': np.random.normal(650, 80, n).clip(300, 850).astype(int),
    'employment_years': np.random.randint(0, 30, n),
    'num_existing_loans': np.random.randint(0, 5, n),
    'monthly_expenses': np.random.normal(15000, 5000, n).clip(5000, 40000).round(-2),
    'education': np.random.choice(['High School', 'Diploma', 'Degree', 'Postgraduate'], n,
                                   p=[0.2, 0.3, 0.35, 0.15]),
    'loan_purpose': np.random.choice(['Home', 'Vehicle', 'Business', 'Personal', 'Education'], n,
                                      p=[0.3, 0.25, 0.2, 0.15, 0.1]),
    'province': np.random.choice(['Gauteng', 'Western Cape', 'KZN', 'Eastern Cape', 'Limpopo'], n,
                                  p=[0.35, 0.25, 0.2, 0.1, 0.1]),
}

df = pd.DataFrame(data)

# Derived features (new columns from existing ones)
df['debt_to_income'] = (df['loan_amount'] / df['annual_income']).round(3)
df['monthly_income'] = (df['annual_income'] / 12).round(0)
df['loan_to_income'] = (df['loan_amount'] / df['annual_income']).round(3)

# Create default label based on realistic risk logic
default_prob = (
    (df['credit_score'] < 580).astype(int) * 0.4 +
    (df['debt_to_income'] > 0.6).astype(int) * 0.3 +
    (df['num_existing_loans'] > 3).astype(int) * 0.2 +
    (df['employment_years'] < 2).astype(int) * 0.1
)
df['defaulted'] = (np.random.random(n) < default_prob.clip(0.05, 0.85)).astype(int)

# Add some missing values (realistic)
df.loc[np.random.choice(n, 20), 'credit_score'] = np.nan
df.loc[np.random.choice(n, 15), 'employment_years'] = np.nan

print("=== DATASET CREATED ===")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Default rate: {df['defaulted'].mean()*100:.1f}%")


# ============================================================
# PART 2 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

# Color palette
GOLD='#FFD700'; GREEN='#00FF9C'; RED='#FF4757'; BLUE='#00B4FF'
PURPLE='#B44FFF'; BG='#0a0e1a'; CARD='#111827'; TEXT='#E8EAF0'; GRAY='#8892A4'

def style_ax(ax, title):
    """Apply dark theme styling to any chart"""
    ax.set_facecolor(CARD)
    ax.tick_params(colors=GRAY, labelsize=9)
    ax.spines['bottom'].set_color('#1e2a3a')
    ax.spines['left'].set_color('#1e2a3a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(title, color=TEXT, fontsize=11, fontweight='bold', pad=12)
    ax.yaxis.label.set_color(GRAY)
    ax.xaxis.label.set_color(GRAY)

fig = plt.figure(figsize=(20, 24))
fig.patch.set_facecolor(BG)
gs = GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# Title
ax_title = fig.add_subplot(gs[0, :])
ax_title.set_facecolor(CARD)
ax_title.text(0.5, 0.65, '🏦  SA BANK LOAN RISK ANALYTICS DASHBOARD',
              transform=ax_title.transAxes, fontsize=22, fontweight='bold',
              color=GOLD, ha='center', va='center')
ax_title.text(0.5, 0.25, 'Advanced Financial Data Analysis  |  500 Customers  |  South Africa',
              transform=ax_title.transAxes, fontsize=12, color=GRAY, ha='center')
ax_title.spines[:].set_visible(False)
ax_title.set_xticks([]); ax_title.set_yticks([])

# Chart 1 — Default Rate by Province
ax1 = fig.add_subplot(gs[1, 0])
prov_default = df.groupby('province')['defaulted'].mean() * 100
colors_prov = [RED if v > 25 else GREEN for v in prov_default.values]
bars = ax1.barh(prov_default.index, prov_default.values, color=colors_prov, edgecolor='none', height=0.6)
for bar, val in zip(bars, prov_default.values):
    ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
             va='center', color=TEXT, fontsize=9, fontweight='bold')
style_ax(ax1, '📍 Default Rate by Province')
ax1.set_xlabel('Default Rate (%)')
ax1.axvline(x=prov_default.mean(), color=GOLD, linestyle='--', alpha=0.7, linewidth=1.5)

# Chart 2 — Credit Score Distribution
ax2 = fig.add_subplot(gs[1, 1])
defaults = df[df['defaulted']==1]['credit_score'].dropna()
non_defaults = df[df['defaulted']==0]['credit_score'].dropna()
ax2.hist(non_defaults, bins=30, alpha=0.7, color=GREEN, label='No Default', edgecolor='none')
ax2.hist(defaults, bins=30, alpha=0.7, color=RED, label='Defaulted', edgecolor='none')
ax2.axvline(x=580, color=GOLD, linestyle='--', linewidth=2, label='Risk Threshold (580)')
style_ax(ax2, '📊 Credit Score Distribution')
ax2.set_xlabel('Credit Score')
ax2.set_ylabel('Count')
ax2.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor='#1e2a3a')

# Chart 3 — Loan Purpose Pie Chart
ax3 = fig.add_subplot(gs[1, 2])
purpose_counts = df['loan_purpose'].value_counts()
colors_pie = [GOLD, BLUE, GREEN, PURPLE, RED]
wedges, texts, autotexts = ax3.pie(purpose_counts.values, labels=purpose_counts.index,
                                    autopct='%1.1f%%', colors=colors_pie,
                                    pctdistance=0.75, startangle=90,
                                    wedgeprops=dict(edgecolor=BG, linewidth=2))
for text in texts: text.set_color(TEXT); text.set_fontsize(8)
for autotext in autotexts: autotext.set_color(BG); autotext.set_fontsize(8); autotext.set_fontweight('bold')
style_ax(ax3, '🎯 Loan Purpose Breakdown')

# Chart 4 — Income vs Loan Amount Scatter
ax4 = fig.add_subplot(gs[2, 0])
default_mask = df['defaulted'] == 1
ax4.scatter(df[~default_mask]['annual_income']/1000, df[~default_mask]['loan_amount']/1000,
            alpha=0.4, color=GREEN, s=15, label='No Default')
ax4.scatter(df[default_mask]['annual_income']/1000, df[default_mask]['loan_amount']/1000,
            alpha=0.6, color=RED, s=20, label='Defaulted')
style_ax(ax4, '💰 Income vs Loan Amount')
ax4.set_xlabel("Annual Income (R'000)")
ax4.set_ylabel("Loan Amount (R'000)")
ax4.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor='#1e2a3a')

# Chart 5 — Default Rate by Education
ax5 = fig.add_subplot(gs[2, 1])
edu_order = ['High School', 'Diploma', 'Degree', 'Postgraduate']
edu_default = df.groupby('education')['defaulted'].mean() * 100
edu_default = edu_default.reindex(edu_order)
bar_colors = [RED if v > 20 else BLUE for v in edu_default.values]
bars = ax5.bar(edu_default.index, edu_default.values, color=bar_colors, edgecolor='none', width=0.5)
for bar, val in zip(bars, edu_default.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', color=TEXT, fontsize=9, fontweight='bold')
style_ax(ax5, '🎓 Default Rate by Education')
ax5.set_ylabel('Default Rate (%)')
ax5.tick_params(axis='x', labelrotation=15)

# Chart 6 — Debt-to-Income Boxplot
ax6 = fig.add_subplot(gs[2, 2])
data_box = [df[df['defaulted']==0]['debt_to_income'].values,
            df[df['defaulted']==1]['debt_to_income'].values]
bp = ax6.boxplot(data_box, patch_artist=True, widths=0.4,
                  medianprops=dict(color=GOLD, linewidth=2.5),
                  whiskerprops=dict(color=GRAY),
                  capprops=dict(color=GRAY),
                  flierprops=dict(marker='o', color=GRAY, alpha=0.3, markersize=3))
bp['boxes'][0].set_facecolor(GREEN); bp['boxes'][0].set_alpha(0.7)
bp['boxes'][1].set_facecolor(RED); bp['boxes'][1].set_alpha(0.7)
ax6.set_xticklabels(['No Default', 'Defaulted'])
ax6.axhline(y=0.6, color=GOLD, linestyle='--', linewidth=1.5, label='Risk Threshold (0.6)')
style_ax(ax6, '⚖️ Debt-to-Income Ratio')
ax6.set_ylabel('Debt-to-Income Ratio')
ax6.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor='#1e2a3a')

# Chart 7 — Correlation Heatmap
ax7 = fig.add_subplot(gs[3, :2])
numeric_cols = ['age', 'annual_income', 'loan_amount', 'credit_score',
                'employment_years', 'debt_to_income', 'num_existing_loans', 'defaulted']
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, ax=ax7,
            annot_kws={'size': 9, 'color': 'black'},
            linewidths=0.5, linecolor=BG,
            cbar_kws={'shrink': 0.8})
style_ax(ax7, '🔥 Correlation Heatmap — Key Financial Metrics')
ax7.tick_params(axis='x', labelrotation=30)

# Chart 8 — KPI Summary Box
ax8 = fig.add_subplot(gs[3, 2])
ax8.set_facecolor(CARD)
ax8.set_xlim(0, 1); ax8.set_ylim(0, 1)
ax8.set_xticks([]); ax8.set_yticks([])
ax8.spines[:].set_visible(False)
ax8.set_title('📈 Key Risk Metrics', color=TEXT, fontsize=11, fontweight='bold', pad=12)

kpis = [
    ('Total Customers', '500', BLUE),
    ('Default Rate', f'{df["defaulted"].mean()*100:.1f}%', RED),
    ('Avg Credit Score', f'{df["credit_score"].mean():.0f}', GREEN),
    ('Avg Loan Amount', f'R{df["loan_amount"].mean()/1000:.0f}k', GOLD),
    ('Avg Debt/Income', f'{df["debt_to_income"].mean():.2f}', PURPLE),
    ('High Risk Customers', f'{(df["credit_score"]<580).sum()}', RED),
]

for i, (label, value, color) in enumerate(kpis):
    y = 0.88 - i * 0.15
    ax8.text(0.05, y, label, color=GRAY, fontsize=9)
    ax8.text(0.95, y, value, color=color, fontsize=12, fontweight='bold', ha='right')
    ax8.axhline(y=y - 0.03, xmin=0.05, xmax=0.95, color='#1e2a3a', linewidth=0.5)

plt.savefig('finance_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.show()
print("Dashboard saved!")


# ============================================================
# PART 3 — MACHINE LEARNING (Loan Default Prediction)
# ============================================================

# Step 1 — Clean missing values
df['credit_score'] = df['credit_score'].fillna(df['credit_score'].median())
df['employment_years'] = df['employment_years'].fillna(df['employment_years'].median())

# Step 2 — Encode text columns into numbers (ML needs numbers)
le = LabelEncoder()
for col in ['education', 'loan_purpose', 'province']:
    df[col] = le.fit_transform(df[col])

# Step 3 — Define features (inputs) and target (what we predict)
features = ['age', 'annual_income', 'loan_amount', 'credit_score',
            'employment_years', 'debt_to_income', 'num_existing_loans',
            'education', 'loan_purpose', 'province']
X = df[features]   # inputs
y = df['defaulted'] # output (0=no default, 1=default)

# Step 4 — Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5 — Scale features (bring all numbers to same range)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6 — Train Model 1: Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test_scaled)[:,1])

# Step 7 — Train Model 2: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:,1])

# Step 8 — Print Results
print("\n=== LOGISTIC REGRESSION RESULTS ===")
print(classification_report(y_test, lr_pred))
print(f"AUC Score: {lr_auc:.3f}")

print("\n=== RANDOM FOREST RESULTS ===")
print(classification_report(y_test, rf_pred))
print(f"AUC Score: {rf_auc:.3f}")

# Step 9 — Feature Importance (what matters most?)
importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n=== TOP FEATURES FOR PREDICTING DEFAULT ===")
print(importance)

# Step 10 — Visualise ML Results
fig2 = plt.figure(figsize=(18, 10))
fig2.patch.set_facecolor(BG)
gs2 = GridSpec(2, 3, figure=fig2, hspace=0.5, wspace=0.4)

# ROC Curve
ax1 = fig2.add_subplot(gs2[:, 0])
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr.predict_proba(X_test_scaled)[:,1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:,1])
ax1.plot(lr_fpr, lr_tpr, color=BLUE, linewidth=2.5, label=f'Logistic Regression (AUC={lr_auc:.3f})')
ax1.plot(rf_fpr, rf_tpr, color=GOLD, linewidth=2.5, label=f'Random Forest (AUC={rf_auc:.3f})')
ax1.plot([0,1],[0,1], color=GRAY, linestyle='--', linewidth=1, label='Random (AUC=0.5)')
ax1.fill_between(rf_fpr, rf_tpr, alpha=0.1, color=GOLD)
style_ax(ax1, '📈 ROC Curve — Model Comparison')
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, edgecolor='#1e2a3a')

# Feature Importance
ax2 = fig2.add_subplot(gs2[:, 1])
imp_sorted = importance.sort_values('importance')
bar_colors = [GOLD if v > 0.12 else BLUE if v > 0.08 else GRAY for v in imp_sorted['importance']]
bars = ax2.barh(imp_sorted['feature'], imp_sorted['importance'], color=bar_colors, edgecolor='none', height=0.6)
for bar, val in zip(bars, imp_sorted['importance']):
    ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', color=TEXT, fontsize=8)
style_ax(ax2, '🔑 Feature Importance (Random Forest)')
ax2.set_xlabel('Importance Score')

# Confusion Matrix - Logistic Regression
ax3 = fig2.add_subplot(gs2[0, 2])
cm_lr = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'],
            cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
style_ax(ax3, '🔵 Confusion Matrix — Logistic Regression')
ax3.set_ylabel('Actual'); ax3.set_xlabel('Predicted')

# Confusion Matrix - Random Forest
ax4 = fig2.add_subplot(gs2[1, 2])
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='YlOrRd', ax=ax4,
            xticklabels=['No Default','Default'],
            yticklabels=['No Default','Default'],
            cbar=False, annot_kws={'size': 12, 'weight': 'bold'})
style_ax(ax4, '🟡 Confusion Matrix — Random Forest')
ax4.set_ylabel('Actual'); ax4.set_xlabel('Predicted')

plt.suptitle('🤖  ML MODEL PERFORMANCE — LOAN DEFAULT PREDICTION',
             color=GOLD, fontsize=16, fontweight='bold', y=1.02)
plt.savefig('finance_ml_results.png', dpi=150, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.show()
print("\nML results saved!")
print("\n✅ FULL ANALYSIS COMPLETE!")