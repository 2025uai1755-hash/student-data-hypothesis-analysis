import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

files = ['cleaned_student_data_full_139.csv', 'fdsc - Form Responses 1.csv']
df = None
for f in files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        break
if df is None:
    raise FileNotFoundError("Dataset file not found in Downloads folder!")

df['CGPA'] = pd.to_numeric(df['CGPA'], errors='coerce')
df['CP Rating'] = pd.to_numeric(df['CP Rating'], errors='coerce')
df['Companies Applied'] = pd.to_numeric(df['Companies Applied'], errors='coerce')
df['Offers Received'] = pd.to_numeric(df['Offers Received'], errors='coerce')

funnel_df = df[(df['Companies Applied'] > 0) & (df['Offers Received'].notna())].copy()
funnel_df['Conversion_Rate'] = funnel_df['Offers Received'] / funnel_df['Companies Applied']

valid_cgpa = funnel_df.dropna(subset=['CGPA', 'Conversion_Rate'])
valid_cp = funnel_df.dropna(subset=['CP Rating', 'Conversion_Rate'])

ALPHA = 0.05

r_cgpa, p_cgpa = stats.pearsonr(valid_cgpa['CGPA'], valid_cgpa['Conversion_Rate'])
n_c = len(valid_cgpa)
t_cgpa = r_cgpa * np.sqrt((n_c - 2) / (1 - r_cgpa**2 + 1e-10))
t_crit_c = stats.t.ppf(1 - ALPHA/2, n_c - 2)

r_cp, p_cp = stats.pearsonr(valid_cp['CP Rating'], valid_cp['Conversion_Rate'])
n_cp = len(valid_cp)
t_cp = r_cp * np.sqrt((n_cp - 2) / (1 - r_cp**2 + 1e-10))
t_crit_cp = stats.t.ppf(1 - ALPHA/2, n_cp - 2)

print("="*75)
print("           HYPOTHESIS 4: PLACEMENT CONVERSION FUNNEL EFFICIENCY")
print("="*75)
print("Null Hypothesis (H0)     : Neither CGPA nor CP Rating has a significant correlation with Conversion Rate (r = 0)")
print("Alternate Hypothesis (H1): CGPA and/or CP Rating has a statistically significant correlation with Conversion Rate (r != 0)")
print("-"*75)

print("1. CGPA vs Conversion Rate:")
print(f"   Pearson Correlation (r): {r_cgpa:.4f}")
print(f"   Calculated T (t_cal)    : {t_cgpa:.4f}")
print(f"   Critical T (t_crit)     : ±{t_crit_c:.4f} (df = {n_c - 2})")
print(f"   p-value                 : {p_cgpa:.4e}")
verdict_cgpa = "REJECT H0 -> Statistically significant correlation!" if p_cgpa < ALPHA else "FAIL TO REJECT H0"
print(f"   Verdict                 : {verdict_cgpa}")

print("\n2. CP Rating vs Conversion Rate:")
print(f"   Pearson Correlation (r): {r_cp:.4f}")
print(f"   Calculated T (t_cal)    : {t_cp:.4f}")
print(f"   Critical T (t_crit)     : ±{t_crit_cp:.4f} (df = {n_cp - 2})")
print(f"   p-value                 : {p_cp:.4e}")
verdict_cp = "REJECT H0 -> Statistically significant correlation!" if p_cp < ALPHA else "FAIL TO REJECT H0"
print(f"   Verdict                 : {verdict_cp}")
print("="*75)

# Plotting Bell Curve Graph
plt.figure(figsize=(10, 5))
max_t = max(abs(t_cgpa), abs(t_cp), t_crit_c) + 1
x = np.linspace(-4.5, max(4.5, max_t), 1000)
y = stats.t.pdf(x, n_c - 2)

plt.plot(x, y, label=f"t-Distribution (df={n_c - 2})", color="black", lw=2)
plt.fill_between(x, 0, y, where=(x >= t_crit_c), color="red", alpha=0.3, label="Rejection Region (α = 0.05)")
plt.fill_between(x, 0, y, where=(x <= -t_crit_c), color="red", alpha=0.3)

plt.axvline(t_crit_c, color="green", linestyle="--", lw=2, label=f"Critical T (+{t_crit_c:.2f})")
plt.axvline(-t_crit_c, color="green", linestyle="--", lw=2)
plt.axvline(t_cgpa, color="blue", linestyle="-", lw=2.5, label=f"CGPA Correlation T (t_cal = {t_cgpa:.2f})")
plt.axvline(t_cp, color="orange", linestyle="-", lw=2.5, label=f"CP Rating Correlation T (t_cal = {t_cp:.2f})")

plt.title("Hypothesis 4: Placement Conversion Funnel Significance", fontsize=13, fontweight='bold')
plt.xlabel("t-statistic")
plt.ylabel("Probability Density")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("h4_graph.png", dpi=300)
print("Graph saved as 'h4_graph.png'\n")
