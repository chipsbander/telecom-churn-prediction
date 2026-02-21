import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

# ─── 1. Load Data ───────────────────────────────────────────────────────────────
df = pd.read_csv('/mnt/user-data/uploads/telecom_customer_churn.csv')
zipcode_pop = pd.read_csv('/mnt/user-data/uploads/telecom_zipcode_population.csv')
dd = pd.read_csv('/mnt/user-data/uploads/telecom_data_dictionary.csv', encoding='latin1')

print(f"Dataset shape: {df.shape}")
print(f"Churn distribution:\n{df['Customer Status'].value_counts()}")

# ─── 2. Merge zipcode population ────────────────────────────────────────────────
zipcode_pop.columns = zipcode_pop.columns.str.strip()
df = df.merge(zipcode_pop[['Zip Code', 'Population']], on='Zip Code', how='left')

# ─── 3. Create Target ────────────────────────────────────────────────────────────
df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)

# ─── 4. Drop leakage & irrelevant columns ───────────────────────────────────────
drop_cols = ['Customer ID', 'Customer Status', 'Churn Category', 'Churn Reason',
             'Latitude', 'Longitude', 'City', 'Zip Code']
df = df.drop(columns=drop_cols, errors='ignore')

# ─── 5. Feature Engineering ─────────────────────────────────────────────────────
df['Avg_Revenue_Per_Month'] = df['Total Revenue'] / (df['Tenure in Months'] + 1)
df['Refund_Rate'] = df['Total Refunds'] / (df['Total Revenue'] + 1)
df['Extra_Data_Flag'] = (df['Total Extra Data Charges'] > 0).astype(int)
df['Services_Count'] = (df[['Phone Service', 'Internet Service', 'Online Security',
                              'Online Backup', 'Device Protection Plan',
                              'Premium Tech Support', 'Streaming TV',
                              'Streaming Movies', 'Streaming Music']] == 'Yes').sum(axis=1)

# ─── 6. Identify feature types ──────────────────────────────────────────────────
cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
num_cols = [c for c in num_cols if c != 'Churn']

print(f"\nCategorical features: {len(cat_cols)}")
print(f"Numerical features: {len(num_cols)}")

# ─── 7. Split ───────────────────────────────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─── 8. Preprocessing pipeline ──────────────────────────────────────────────────
num_imputer = SimpleImputer(strategy="median")
num_transformer = StandardScaler()
cat_imputer = SimpleImputer(strategy='most_frequent')
cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imp', num_imputer), ('scl', num_transformer)]), num_cols),
    ('cat', Pipeline([('imp', cat_imputer), ('enc', cat_transformer)]), cat_cols)
])

# ─── 9. Train multiple models ────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(max_depth=8, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                             class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                     learning_rate=0.05, random_state=42)
}

results = {}
pipelines = {}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== Cross-Validation Results ===")
for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('clf', model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    
    results[name] = {
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
        'test_auc': roc_auc_score(y_test, y_prob),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_f1': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
    }
    pipelines[name] = pipe
    print(f"{name}: CV AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f} | Test AUC = {roc_auc_score(y_test, y_prob):.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]['test_auc'])
best_model = pipelines[best_name]
best_res = results[best_name]
print(f"\n✅ Best Model: {best_name} (Test AUC = {best_res['test_auc']:.4f})")
print(classification_report(y_test, best_res['y_pred'], target_names=['Stayed', 'Churned']))

# ─── 10. Feature Importance (Random Forest) ─────────────────────────────────────
rf_pipe = pipelines['Random Forest']
rf_model = rf_pipe.named_steps['clf']
ohe_features = rf_pipe.named_steps['prep'].named_transformers_['cat'].named_steps['enc'].get_feature_names_out(cat_cols)
all_features = num_cols + list(ohe_features)
fi = pd.Series(rf_model.feature_importances_, index=all_features).sort_values(ascending=False).head(20)

# ─── 11. Comprehensive Visualization ────────────────────────────────────────────
sns.set_style('whitegrid')
fig = plt.figure(figsize=(22, 26))
fig.suptitle('Telecom Customer Churn Prediction — Model Report', fontsize=18, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Plot 1: Churn Distribution ---
ax1 = fig.add_subplot(gs[0, 0])
churn_counts = y.value_counts()
colors = ['#2ecc71', '#e74c3c']
bars = ax1.bar(['Stayed', 'Churned'], churn_counts.values, color=colors, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, churn_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'{val}\n({val/len(y)*100:.1f}%)',
             ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_title('Class Distribution', fontsize=13, fontweight='bold')
ax1.set_ylabel('Count')
ax1.set_ylim(0, churn_counts.max() * 1.15)

# --- Plot 2: Model Comparison ---
ax2 = fig.add_subplot(gs[0, 1])
model_names = list(results.keys())
auc_scores = [results[m]['test_auc'] for m in model_names]
f1_scores_list = [results[m]['test_f1'] for m in model_names]
x = np.arange(len(model_names))
width = 0.35
bars1 = ax2.bar(x - width/2, auc_scores, width, label='AUC-ROC', color='#3498db', alpha=0.85)
bars2 = ax2.bar(x + width/2, f1_scores_list, width, label='F1-Score', color='#e67e22', alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels([m.replace(' ', '\n') for m in model_names], fontsize=8)
ax2.set_ylim(0.5, 1.0)
ax2.set_title('Model Comparison', fontsize=13, fontweight='bold')
ax2.legend()
ax2.set_ylabel('Score')
for bar in bars1:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=7.5, fontweight='bold')
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{bar.get_height():.3f}',
             ha='center', va='bottom', fontsize=7.5, fontweight='bold')
# highlight best
best_idx = model_names.index(best_name)
ax2.axvline(x=best_idx, color='red', linestyle='--', alpha=0.3, linewidth=2)
ax2.text(best_idx, 0.95, '★ Best', ha='center', color='red', fontsize=9, fontweight='bold')

# --- Plot 3: ROC Curves ---
ax3 = fig.add_subplot(gs[0, 2])
colors_roc = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
for (name, res), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax3.plot(fpr, tpr, label=f"{name} ({res['test_auc']:.3f})", color=color, linewidth=2)
ax3.plot([0,1],[0,1],'k--', alpha=0.4)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves — All Models', fontsize=13, fontweight='bold')
ax3.legend(fontsize=8, loc='lower right')

# --- Plot 4: Confusion Matrix (Best Model) ---
ax4 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best_res['y_pred'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stayed', 'Churned'])
disp.plot(ax=ax4, colorbar=False, cmap='Blues')
ax4.set_title(f'Confusion Matrix\n({best_name})', fontsize=13, fontweight='bold')

# --- Plot 5: Precision-Recall Curve ---
ax5 = fig.add_subplot(gs[1, 1])
for (name, res), color in zip(results.items(), colors_roc):
    prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
    ax5.plot(rec, prec, label=name, color=color, linewidth=2)
baseline = y_test.mean()
ax5.axhline(y=baseline, color='black', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.2f})')
ax5.set_xlabel('Recall')
ax5.set_ylabel('Precision')
ax5.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
ax5.legend(fontsize=8)

# --- Plot 6: Feature Importance ---
ax6 = fig.add_subplot(gs[1, 2])
colors_fi = ['#e74c3c' if i < 5 else '#3498db' for i in range(len(fi))]
fi.plot(kind='barh', ax=ax6, color=colors_fi[::-1])
ax6.set_title('Top 20 Feature Importances\n(Random Forest)', fontsize=13, fontweight='bold')
ax6.set_xlabel('Importance')
ax6.invert_yaxis()

# --- Plot 7: Churn by Contract Type ---
ax7 = fig.add_subplot(gs[2, 0])
contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
bars = ax7.bar(contract_churn.index, contract_churn.values * 100, 
               color=['#e74c3c','#e67e22','#2ecc71'])
for bar, val in zip(bars, contract_churn.values):
    ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val*100:.1f}%',
             ha='center', va='bottom', fontweight='bold')
ax7.set_title('Churn Rate by Contract Type', fontsize=13, fontweight='bold')
ax7.set_ylabel('Churn Rate (%)')
ax7.set_ylim(0, contract_churn.max() * 100 * 1.15)

# --- Plot 8: Churn by Internet Type ---
ax8 = fig.add_subplot(gs[2, 1])
internet_churn = df.groupby('Internet Type')['Churn'].mean().sort_values(ascending=False)
palette = ['#e74c3c','#e67e22','#3498db','#2ecc71']
bars = ax8.bar(internet_churn.index, internet_churn.values * 100, color=palette[:len(internet_churn)])
for bar, val in zip(bars, internet_churn.values):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val*100:.1f}%',
             ha='center', va='bottom', fontweight='bold')
ax8.set_title('Churn Rate by Internet Type', fontsize=13, fontweight='bold')
ax8.set_ylabel('Churn Rate (%)')
ax8.set_ylim(0, internet_churn.max() * 100 * 1.2)

# --- Plot 9: Churn by Tenure Bucket ---
ax9 = fig.add_subplot(gs[2, 2])
df['Tenure_Bucket'] = pd.cut(df['Tenure in Months'], bins=[0, 6, 12, 24, 36, 72],
                               labels=['0-6m', '7-12m', '13-24m', '25-36m', '37m+'])
tenure_churn = df.groupby('Tenure_Bucket', observed=True)['Churn'].mean()
colors_t = ['#e74c3c','#e67e22','#f1c40f','#3498db','#2ecc71']
bars = ax9.bar(tenure_churn.index, tenure_churn.values * 100, color=colors_t)
for bar, val in zip(bars, tenure_churn.values):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val*100:.1f}%',
             ha='center', va='bottom', fontweight='bold', fontsize=9)
ax9.set_title('Churn Rate by Tenure', fontsize=13, fontweight='bold')
ax9.set_ylabel('Churn Rate (%)')
ax9.set_ylim(0, tenure_churn.max() * 100 * 1.2)

# --- Plot 10: Monthly Charge Distribution ---
ax10 = fig.add_subplot(gs[3, 0])
ax10.hist(df[df['Churn']==0]['Monthly Charge'], bins=40, alpha=0.7, label='Stayed', color='#2ecc71', density=True)
ax10.hist(df[df['Churn']==1]['Monthly Charge'], bins=40, alpha=0.7, label='Churned', color='#e74c3c', density=True)
ax10.set_title('Monthly Charge Distribution', fontsize=13, fontweight='bold')
ax10.set_xlabel('Monthly Charge ($)')
ax10.set_ylabel('Density')
ax10.legend()

# --- Plot 11: Tenure Distribution ---
ax11 = fig.add_subplot(gs[3, 1])
ax11.hist(df[df['Churn']==0]['Tenure in Months'], bins=40, alpha=0.7, label='Stayed', color='#2ecc71', density=True)
ax11.hist(df[df['Churn']==1]['Tenure in Months'], bins=40, alpha=0.7, label='Churned', color='#e74c3c', density=True)
ax11.set_title('Tenure Distribution', fontsize=13, fontweight='bold')
ax11.set_xlabel('Tenure (Months)')
ax11.set_ylabel('Density')
ax11.legend()

# --- Plot 12: Churn Probability Threshold Analysis ---
ax12 = fig.add_subplot(gs[3, 2])
thresholds = np.arange(0.1, 0.9, 0.05)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    preds = (best_res['y_prob'] >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f = f1_score(y_test, preds, zero_division=0)
    precisions.append(p); recalls.append(r); f1s.append(f)
ax12.plot(thresholds, precisions, label='Precision', color='#3498db', linewidth=2)
ax12.plot(thresholds, recalls, label='Recall', color='#e74c3c', linewidth=2)
ax12.plot(thresholds, f1s, label='F1-Score', color='#2ecc71', linewidth=2, linestyle='--')
best_thresh = thresholds[np.argmax(f1s)]
ax12.axvline(x=best_thresh, color='gray', linestyle=':', linewidth=1.5)
ax12.text(best_thresh + 0.01, 0.2, f'Best threshold\n= {best_thresh:.2f}', fontsize=8)
ax12.set_xlabel('Classification Threshold')
ax12.set_ylabel('Score')
ax12.set_title(f'Threshold Analysis\n({best_name})', fontsize=13, fontweight='bold')
ax12.legend()

plt.savefig('/mnt/user-data/outputs/churn_model_report.png', dpi=150, bbox_inches='tight',
            facecolor='white')
print("\nReport saved!")

# ─── 12. Save best model ─────────────────────────────────────────────────────────
joblib.dump(best_model, '/mnt/user-data/outputs/best_churn_model.pkl')
print("Model saved.")

# ─── 13. Print final summary ─────────────────────────────────────────────────────
print("\n" + "="*55)
print("         FINAL MODEL SUMMARY")
print("="*55)
for name, res in results.items():
    marker = " ★" if name == best_name else ""
    print(f"{name+marker:<28} | AUC: {res['test_auc']:.4f} | F1: {res['test_f1']:.4f} | Acc: {res['test_acc']:.4f}")
print("="*55)
print(f"\nBest Threshold (max F1): {best_thresh:.2f}")
print(f"\nTop 10 Churn Predictors (Random Forest):")
for feat, imp in fi.head(10).items():
    print(f"  {feat:<45} {imp:.4f}")
