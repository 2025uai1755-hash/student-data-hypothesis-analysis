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

df['Academic Workload Rating (1-5)'] = pd.to_numeric(df['Academic Workload Rating (1-5)'], errors='coerce')
df['Peer Culture Influence Rating (1-5)'] = pd.to_numeric(df['Peer Culture Influence Rating (1-5)'], errors='coerce')

high_workload = df[df['Academic Workload Rating (1-5)'] >= 4].copy()
regular_coding_mask = high_workload['Practices Coding Regularly'].astype(str).str.contains('regularly|occasionally', case=False, na=False)

high_peer = high_workload[high_workload['Peer Culture Influence Rating (1-5)'] >= 4]
low_peer = high_workload[high_workload['Peer Culture Influence Rating (1-5)'] < 4]

n1, n2 = len(high_peer), len(low_peer)
x1 = regular_coding_mask[high_peer.index].sum()
x2 = regular_coding_mask[low_peer.index].sum()

p1 = x1 / n1 if n1 > 0 else 0
p2 = x2 / n2 if n2 > 0 else 0

ALPHA = 0.05
p_pool = (x1 + x2) / (n1 + n2) if (n1 + n2) > 0 else 0
se_z = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2)) if 0 < p_pool < 1 else 1e-5

z_cal = (p1 - p2) / se_z if se_z > 0 else 0
z_crit = stats.norm.ppf(1 - ALPHA/2)
p_val = 2 * (1 - stats.norm.cdf(abs(z_cal)))
ratio = p1 / p2 if p2 > 0 else 0

print("="*75)
print("      HYPOTHESIS 3: PEER CULTURE BUFFER UNDER HIGH WORKLOAD (Workload >= 4)")
print("="*75)
print("Null Hypothesis (H0)     : Regular coding proportion in High Peer Culture equals Low Peer Culture (p1 = p2)")
print("Alternate Hypothesis (H1): High Peer Culture significantly increases regular coding retention under high workload (p1 > p2)")
print("-"*75)
print(f"High Peer Group (n1 = {n1}) : Regular Coders = {x1} ({p1*100:.1f}%)")
print(f"Low Peer Group  (n2 = {n2}) : Regular Coders = {x2} ({p2*100:.1f}%)")
print(f"Fold Difference Ratio (p1/p2): {ratio:.2f}x")
print(f"Calculated Z (z_cal)         : {z_cal:.4f}")
print(f"Critical Z (z_crit α=0.05)   : ±{z_crit:.4f}")
print(f"p-value                      : {p_val:.4e}")
print("-" * 75)
if p_val < ALPHA:
    print("VERDICT: REJECT H0 -> High Peer Culture acts as a significant statistical multiplier!")
else:
    print("VERDICT: FAIL TO REJECT H0 -> Difference is not statistically significant at α=0.05.")
print("="*75)

# Plotting Bell Curve Graph
plt.figure(figsize=(10, 5))
x = np.linspace(-4.5, max(4.5, abs(z_cal) + 1), 1000)
y = stats.norm.pdf(x)

plt.plot(x, y, label="Standard Normal Distribution N(0,1)", color="teal", lw=2)
plt.fill_between(x, 0, y, where=(x >= z_crit), color="red", alpha=0.3, label="Rejection Region (|Z| >= 1.96)")
plt.fill_between(x, 0, y, where=(x <= -z_crit), color="red", alpha=0.3)

plt.axvline(z_crit, color="green", linestyle="--", lw=2, label=f"Critical Z (+{z_crit:.2f})")
plt.axvline(-z_crit, color="green", linestyle="--", lw=2)
plt.axvline(z_cal, color="navy", linestyle="-", lw=2.5, label=f"Calculated Z (z_cal = {z_cal:.2f})")

plt.title("Hypothesis 3: Peer Culture Buffer (Two-Proportion Z-Test)", fontsize=13, fontweight='bold')
plt.xlabel("Z-score")
plt.ylabel("Probability Density")
plt.legend(loc="upper right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("h3_graph.png", dpi=300)
print("Graph saved as 'h3_graph.png'\n")
