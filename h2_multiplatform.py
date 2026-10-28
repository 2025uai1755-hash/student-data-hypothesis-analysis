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

df['Total Problems Solved'] = pd.to_numeric(df['Total Problems Solved'], errors='coerce')

is_multi = df['Coding Platforms Used'].astype(str).str.contains(',', na=False)
multi_group = df[is_multi]['Total Problems Solved'].dropna()
single_group = df[~is_multi]['Total Problems Solved'].dropna()

n1, n2 = len(multi_group), len(single_group)
mean1, mean2 = multi_group.mean(), single_group.mean()
s1, s2 = multi_group.std(ddof=1), single_group.std(ddof=1)

ALPHA = 0.05
t_stat, p_val = stats.ttest_ind(multi_group, single_group, equal_var=False)
df_welch = ((s1**2/n1 + s2**2/n2)**2) / (((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))
t_crit = stats.t.ppf(1 - ALPHA/2, df_welch)

print("="*75)
print("         HYPOTHESIS 2: MULTI-PLATFORM VS SINGLE-PLATFORM USERS")
print("="*75)
print("Null Hypothesis (H0)     : Multi-platform users solve the SAME number of problems as single-platform users (μ1 = μ2)")
print("Alternate Hypothesis (H1): Multi-platform users solve significantly MORE problems than single-platform users (μ1 > μ2)")
print("-"*75)
print(f"Multi-Platform (n1 = {n1})  : Mean = {mean1:.2f}, Std = {s1:.2f}")
print(f"Single-Platform (n2 = {n2}) : Mean = {mean2:.2f}, Std = {s2:.2f}")
print(f"Calculated T (t_cal)        : {t_stat:.4f}")
print(f"Critical T (t_crit α=0.05)  : ±{t_crit:.4f}")
print(f"Degrees of Freedom (df)     : {df_welch:.2f}")
print(f"p-value                     : {p_val:.4e}")
print("-" * 75)
if p_val < ALPHA:
    print("VERDICT: REJECT H0 -> Multi-platform users solve significantly more problems!")
else:
    print("VERDICT: FAIL TO REJECT H0 -> Difference is not statistically significant.")
print("="*75)

# Plotting Bell Curve Graph
plt.figure(figsize=(10, 5))
x = np.linspace(-4.5, max(4.5, abs(t_stat) + 1), 1000)
y = stats.t.pdf(x, df_welch)

plt.plot(x, y, label=f"t-Distribution (df={df_welch:.1f})", color="purple", lw=2)
plt.fill_between(x, 0, y, where=(x >= t_crit), color="orange", alpha=0.3, label="Rejection Region (α/2 = 0.025)")
plt.fill_between(x, 0, y, where=(x <= -t_crit), color="orange", alpha=0.3)

plt.axvline(t_crit, color="green", linestyle="--", lw=2, label=f"Critical T (+{t_crit:.2f})")
plt.axvline(-t_crit, color="green", linestyle="--", lw=2)
plt.axvline(t_stat, color="magenta", linestyle="-", lw=2.5, label=f"Calculated T (t_cal = {t_stat:.2f})")

plt.title("Hypothesis 2: Multi-Platform Exposure (Welch's t-Test Distribution)", fontsize=13, fontweight='bold')
plt.xlabel("t-statistic")
plt.ylabel("Probability Density")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("h2_graph.png", dpi=300)
print("Graph saved as 'h2_graph.png'\n")
