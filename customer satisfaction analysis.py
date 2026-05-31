# ============================================================
#  Customer Satisfaction Analysis - BPO Dataset
#  Author: [Your Name]
#  Description: Exploratory Data Analysis on customer
#               satisfaction scores and behavioral data
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d0d1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#aaa",
    "xtick.color":      "#888",
    "ytick.color":      "#888",
    "text.color":       "#e0e0e0",
    "grid.color":       "#222",
    "grid.linewidth":   0.7,
    "font.family":      "monospace",
})
PALETTE = ["#00C9A7","#845EC2","#FF6B6B","#FFD93D",
           "#4D96FF","#F9C74F","#90BE6D","#F8961E",
           "#F3722C","#43AA8B","#577590","#277DA1"]

# ── 1. Load Data ─────────────────────────────────────────────
df = pd.read_csv("Customer_Satisfaction_Scores_and_Behavior_Data.csv")

print("=" * 55)
print("  CUSTOMER SATISFACTION ANALYSIS — BPO DATASET")
print("=" * 55)

# ── 2. Basic Info ─────────────────────────────────────────────
print("\n📋 DATASET OVERVIEW")
print(f"   Rows      : {df.shape[0]}")
print(f"   Columns   : {df.shape[1]}")
print(f"   Columns   : {list(df.columns)}")
print(f"\n   Missing values:\n{df.isnull().sum()}")

print("\n📊 SATISFACTION SCORE STATS")
print(df["Satisfaction_Score"].describe().round(2))

# ── 3. Average score by key factors ──────────────────────────
print("\n🔍 AVG SATISFACTION BY FACTOR")
for col in ["Loyalty_Level","Gender","Group","Support_Contacted","Purchase_History"]:
    print(f"\n  [{col}]")
    print(df.groupby(col)["Satisfaction_Score"]
            .mean().round(2)
            .sort_values(ascending=False)
            .to_string())

print("\n  [Satisfaction_Factor — Top & Bottom 3]")
factor_avg = (df.groupby("Satisfaction_Factor")["Satisfaction_Score"]
                .mean().round(2).sort_values(ascending=False))
print("  Top 3:\n", factor_avg.head(3).to_string())
print("  Bottom 3:\n", factor_avg.tail(3).to_string())

print("\n  [Location]")
print(df.groupby("Location")["Satisfaction_Score"]
        .mean().round(2)
        .sort_values(ascending=False)
        .to_string())

# ── 4. Visualisations ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Customer Satisfaction Analysis — BPO Dataset",
             fontsize=16, fontweight="bold", color="#00C9A7", y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

def style_ax(ax, title):
    ax.set_title(title, fontsize=10, color="#aaa",
                 fontweight="bold", pad=10, loc="left")
    ax.grid(axis="y", alpha=0.4)
    ax.spines[["top","right"]].set_visible(False)

# — 4a. Score Distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df["Satisfaction_Score"].value_counts().sort_index()
ax1.bar(counts.index, counts.values, color="#845EC2",
        edgecolor="#0d0d1a", linewidth=0.5)
ax1.set_xlabel("Score"); ax1.set_ylabel("Count")
style_ax(ax1, "Score Distribution")

# — 4b. Avg by Loyalty Level
ax2 = fig.add_subplot(gs[0, 1])
loyalty = df.groupby("Loyalty_Level")["Satisfaction_Score"].mean().reindex(["Low","Medium","High"])
bars = ax2.bar(loyalty.index, loyalty.values,
               color=["#FF6B6B","#FFD93D","#00C9A7"],
               edgecolor="#0d0d1a")
for b in bars:
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{b.get_height():.2f}", ha="center", fontsize=9, color="#ccc")
ax2.set_ylim(0, 11); ax2.set_ylabel("Avg Score")
style_ax(ax2, "Avg Score by Loyalty Level")

# — 4c. Support Contact Impact
ax3 = fig.add_subplot(gs[0, 2])
support = df.groupby("Support_Contacted")["Satisfaction_Score"].mean()
support.index = ["No Support","Contacted Support"]
bars = ax3.bar(support.index, support.values,
               color=["#00C9A7","#FF6B6B"], edgecolor="#0d0d1a")
for b in bars:
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{b.get_height():.2f}", ha="center", fontsize=9, color="#ccc")
ax3.set_ylim(0, 11); ax3.set_ylabel("Avg Score")
style_ax(ax3, "Support Contact Impact")

# — 4d. Satisfaction Factors (ranked)
ax4 = fig.add_subplot(gs[1, :])
factor_sorted = factor_avg.sort_values()
colors = [PALETTE[i % len(PALETTE)] for i in range(len(factor_sorted))]
bars = ax4.barh(factor_sorted.index, factor_sorted.values,
                color=colors, edgecolor="#0d0d1a")
for b in bars:
    ax4.text(b.get_width()+0.05, b.get_y()+b.get_height()/2,
             f"{b.get_width():.2f}", va="center", fontsize=9, color="#ccc")
ax4.set_xlim(0, 11); ax4.set_xlabel("Avg Satisfaction Score")
ax4.grid(axis="x", alpha=0.4)
ax4.spines[["top","right"]].set_visible(False)
ax4.set_title("Avg Score by Satisfaction Factor (Ranked)",
              fontsize=10, color="#aaa", fontweight="bold", pad=10, loc="left")

# — 4e. Location
ax5 = fig.add_subplot(gs[2, :2])
loc_avg = (df.assign(City=df["Location"].str.split(".").str[0])
             .groupby("City")["Satisfaction_Score"]
             .mean().sort_values(ascending=False))
loc_colors = [PALETTE[i % len(PALETTE)] for i in range(len(loc_avg))]
bars = ax5.bar(loc_avg.index, loc_avg.values,
               color=loc_colors, edgecolor="#0d0d1a")
for b in bars:
    ax5.text(b.get_x()+b.get_width()/2, b.get_height()+0.1,
             f"{b.get_height():.2f}", ha="center", fontsize=8, color="#ccc")
ax5.set_ylim(0, 11); ax5.set_ylabel("Avg Score")
plt.setp(ax5.get_xticklabels(), rotation=30, ha="right", fontsize=9)
style_ax(ax5, "Avg Score by City")

# — 4f. Gender & Group
ax6 = fig.add_subplot(gs[2, 2])
gender_grp = df.groupby(["Gender","Group"])["Satisfaction_Score"].mean().unstack()
gender_grp.plot(kind="bar", ax=ax6,
                color=["#FF6B6B","#4D96FF"], edgecolor="#0d0d1a", width=0.6)
ax6.set_ylim(0, 11); ax6.set_ylabel("Avg Score")
ax6.set_xlabel("")
plt.setp(ax6.get_xticklabels(), rotation=0)
ax6.legend(title="Group", fontsize=8, title_fontsize=8,
           facecolor="#1a1a2e", edgecolor="#333")
style_ax(ax6, "Gender × Group")

plt.savefig("customer_satisfaction_charts.png",
            dpi=150, bbox_inches="tight", facecolor="#0d0d1a")
print("\n✅ Charts saved → customer_satisfaction_charts.png")

# ── 5. Correlation — numeric encode ──────────────────────────
df_enc = df.copy()
df_enc["Loyalty_num"]  = df["Loyalty_Level"].map({"Low":1,"Medium":2,"High":3})
df_enc["Support_num"]  = df["Support_Contacted"].map({"No":0,"Yes":1})
df_enc["Purchase_num"] = df["Purchase_History"].map({"No":0,"Yes":1})
df_enc["Gender_num"]   = df["Gender"].map({"Male":0,"Female":1})
df_enc["Group_num"]    = df["Group"].map({"A":0,"B":1})

corr_cols = ["Satisfaction_Score","Loyalty_num","Support_num",
             "Purchase_num","Gender_num","Group_num","Age"]
corr = df_enc[corr_cols].corr()["Satisfaction_Score"].drop("Satisfaction_Score").sort_values()

print("\n📈 CORRELATION WITH SATISFACTION SCORE")
for feat, val in corr.items():
    bar = "█" * int(abs(val) * 20)
    sign = "+" if val > 0 else "-"
    print(f"   {feat:<18} {sign}{abs(val):.3f}  {bar}")

# ── 6. Summary Insights ───────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)
insights = [
    f"• Avg satisfaction score       : {df['Satisfaction_Score'].mean():.2f} / 10",
    f"• High-loyalty customers avg   : {df[df.Loyalty_Level=='High']['Satisfaction_Score'].mean():.2f}",
    f"• Low-loyalty customers avg    : {df[df.Loyalty_Level=='Low']['Satisfaction_Score'].mean():.2f}",
    f"• Contacted support avg        : {df[df.Support_Contacted=='Yes']['Satisfaction_Score'].mean():.2f}",
    f"• No support needed avg        : {df[df.Support_Contacted=='No']['Satisfaction_Score'].mean():.2f}",
    f"• Top satisfaction factor      : {factor_avg.idxmax()} ({factor_avg.max():.2f})",
    f"• Lowest satisfaction factor   : {factor_avg.idxmin()} ({factor_avg.min():.2f})",
    f"• Best city avg score          : {loc_avg.idxmax()} ({loc_avg.max():.2f})",
    f"• Lowest city avg score        : {loc_avg.idxmin()} ({loc_avg.min():.2f})",
]
for line in insights:
    print("  " + line)

print("\n✅ Analysis complete!")
