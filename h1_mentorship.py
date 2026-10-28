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

mentor_yes = df[df['Has Mentor'].astype(str).str.contains('Yes|yes', na=False)]['Total Problems Solved'].dropna()
mentor_no = df[df['Has Mentor'].astype(str).str.contains('No|no', na=False)]['Total Problems Solved'].dropna()

n1, n2 = len(mentor_yes), len(mentor_no)
mean1, mean2 = mentor_yes.mean(), mentor_no.mean()
s1, s2 = mentor_yes.std(ddof=1), mentor_no.std(ddof=1)

ALPHA = 0.05
t_stat, p_val = stats.ttest_ind(mentor_yes, mentor_no, equal_var=False)
df_welch = ((s1**2/n1 + s2**2/n2)**2) / (((s1**2/n1)**2)/(n1-1) + ((s2**2/n2)**2)/(n2-1))
t_crit = stats.t.ppf(1 - ALPHA/2, df_welch)

print("="*75)
print("              HYPOTHESIS 1: MENTORSHIP IMPACT ON PROBLEM SOLVING")
print("="*75)
print("Null Hypothesis (H0)     : Mentorship has NO significant impact on problems solved (μ1 = μ2)")
print("Alternate Hypothesis (H1): Mentored students solve significantly MORE problems than non-mentored (μ1 > μ2)")
print("-"*75)
print(f"Mentored Group (n1 = {n1})     : Mean = {mean1:.2f}, Std = {s1:.2f}")
print(f"Non-Mentored Group (n2 = {n2}) : Mean = {mean2:.2f}, Std = {s2:.2f}")
print(f"Calculated T (t_cal)           : {t_stat:.4f}")
print(f"Critical T (t_crit α=0.05)     : ±{t_crit:.4f}")
print(f"Degrees of Freedom (df)        : {df_welch:.2f}")
print(f"p-value                        : {p_val:.4e}")
print("-" * 75)
if p_val < ALPHA:
    print("VERDICT: REJECT H0 -> Mentorship has a statistically significant positive impact!")
else:
    print("VERDICT: FAIL TO REJECT H0 -> Insufficient statistical evidence to reject H0.")
print("="*75)

# Plotting Bell Curve Graph
plt.figure(figsize=(10, 5))
x = np.linspace(-4.5, max(4.5, abs(t_stat) + 1), 1000)
y = stats.t.pdf(x, df_welch)

plt.plot(x, y, label=f"t-Distribution (df={df_welch:.1f})", color="blue", lw=2)
plt.fill_between(x, 0, y, where=(x >= t_crit), color="red", alpha=0.3, label="Rejection Region (α/2 = 0.025)")
plt.fill_between(x, 0, y, where=(x <= -t_crit), color="red", alpha=0.3)

plt.axvline(t_crit, color="green", linestyle="--", lw=2, label=f"Critical T (+{t_crit:.2f})")
plt.axvline(-t_crit, color="green", linestyle="--", lw=2)
plt.axvline(t_stat, color="darkred", linestyle="-", lw=2.5, label=f"Calculated T (t_cal = {t_stat:.2f})")

plt.title("Hypothesis 1: Mentorship Impact (Welch's t-Test Distribution)", fontsize=13, fontweight='bold')
plt.xlabel("t-statistic")
plt.ylabel("Probability Density")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("h1_graph.png", dpi=300)
print("Graph saved as 'h1_graph.png'\n")
