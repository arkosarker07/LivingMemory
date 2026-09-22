import matplotlib.pyplot as plt
import numpy as np

# Hardcoded data from the 10-conversation JSON you provided
data = {
    "naive_rag": {
        "f1_overall": 0.0575,
        "f1_by_type": {"single_hop": 0.0536, "multi_hop": 0.0148, "temporal": 0.0504, "open_domain": 0.0768},
        "tokens_per_query": 341.3,
        "memory_count_after_ingest": 588
    },
    "scoring_only": {
        "f1_overall": 0.0524,
        "f1_by_type": {"single_hop": 0.0581, "multi_hop": 0.0118, "temporal": 0.0622, "open_domain": 0.0651},
        "tokens_per_query": 280.6,
        "memory_count_after_ingest": 276
    },
    "consolidation_only": {
        "f1_overall": 0.0527,
        "f1_by_type": {"single_hop": 0.0600, "multi_hop": 0.0087, "temporal": 0.0605, "open_domain": 0.0672},
        "tokens_per_query": 239.2,
        "memory_count_after_ingest": 572
    },
    "full_living_memory": {
        "f1_overall": 0.0537,
        "f1_by_type": {"single_hop": 0.0576, "multi_hop": 0.0118, "temporal": 0.0628, "open_domain": 0.0677},
        "tokens_per_query": 278.3,
        "memory_count_after_ingest": 271
    }
}

labels = ["Naive RAG", "Scoring Only", "Consolidator", "Full Living Memory"]
keys = ["naive_rag", "scoring_only", "consolidation_only", "full_living_memory"]

def add_labels(rects, ax, format_str='{:.4f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90)

def add_labels_horizontal(rects, ax, format_str='{:.1f}'):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(format_str.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

# ---------------------------------------------------------
# CHART 1: F1 Score by Question Type
# ---------------------------------------------------------
single_hop = [data[k]["f1_by_type"]["single_hop"] for k in keys]
multi_hop = [data[k]["f1_by_type"]["multi_hop"] for k in keys]
temporal = [data[k]["f1_by_type"]["temporal"] for k in keys]
open_domain = [data[k]["f1_by_type"]["open_domain"] for k in keys]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - 1.5*width, single_hop, width, label='Single-hop', color='#4A90E2')
rects2 = ax.bar(x - 0.5*width, multi_hop, width, label='Multi-hop', color='#E94A4A')
rects3 = ax.bar(x + 0.5*width, temporal, width, label='Temporal', color='#50E3C2')
rects4 = ax.bar(x + 1.5*width, open_domain, width, label='Open-domain', color='#F5A623')

add_labels(rects1, ax)
add_labels(rects2, ax)
add_labels(rects3, ax)
add_labels(rects4, ax)

ax.set_ylabel('F1 Score')
ax.set_title('F1 Score by Question Type Across Conditions (10 Conversations)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper right')
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim(0, max(max(single_hop), max(multi_hop), max(temporal), max(open_domain)) * 1.3)

plt.tight_layout()
plt.savefig("f1_by_type_chart_10convo.png", dpi=300)
print("Saved: f1_by_type_chart_10convo.png")

# ---------------------------------------------------------
# CHART 2: Efficiency (Tokens vs Memory vs F1)
# ---------------------------------------------------------
memory_counts = [data[k]["memory_count_after_ingest"] for k in keys]
tokens = [data[k]["tokens_per_query"] for k in keys]
f1_overall = [data[k]["f1_overall"] for k in keys]

fig, ax1 = plt.subplots(figsize=(10, 7))
width = 0.25

color1 = '#34495e'
color2 = '#e74c3c'
bars1 = ax1.bar(x - width, memory_counts, width, label='Database Size (Memories)', color=color1)
bars2 = ax1.bar(x, tokens, width, label='Token Cost (per query)', color=color2)

ax1.set_xlabel('Condition')
ax1.set_ylabel('Count (Memories / Tokens)')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

add_labels_horizontal(bars1, ax1, '{:.0f}')
add_labels_horizontal(bars2, ax1, '{:.1f}')

ax2 = ax1.twinx()
color3 = '#2ecc71'
bars3 = ax2.bar(x + width, f1_overall, width, label='Overall F1 Score', color=color3)
ax2.set_ylabel('Overall F1 Score')

add_labels_horizontal(bars3, ax2, '{:.4f}')

ax1.set_ylim(0, max(max(memory_counts), max(tokens)) * 1.15)
ax2.set_ylim(0, max(f1_overall) * 1.15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('System Efficiency vs Overall Accuracy (10 Conversations)')
fig.tight_layout()
plt.savefig("efficiency_chart_10convo.png", dpi=300)
print("Saved: efficiency_chart_10convo.png")
