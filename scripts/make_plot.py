import matplotlib.pyplot as plt

# Latest results from eval.py
clean = 87.16
shift_no = 67.54
shift_ttt = 70.66
gain = shift_ttt - shift_no  # should be +3.12

labels = ["Clean (No-TTT)", "Shifted (No-TTT)", "Shifted (TTT)"]
vals = [clean, shift_no, shift_ttt]

plt.figure(figsize=(6, 3.6))
bars = plt.bar(labels, vals)
plt.ylabel("Accuracy (%)")
plt.title(f"TTT improves accuracy under shift (+{gain:.2f} pts)")
plt.xticks(rotation=15, ha="right")
plt.ylim(0, max(vals) + 10)

# Annotate bar values
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width()/2, h + 0.6, f"{h:.2f}", ha="center", va="bottom", fontsize=9)

# Highlight the gain between shifted bars
x1 = bars[1].get_x() + bars[1].get_width()/2
x2 = bars[2].get_x() + bars[2].get_width()/2
y = max(shift_no, shift_ttt) + 4
plt.annotate(
    f"+{gain:.2f} pts",
    xy=(x2, shift_ttt), xytext=((x1 + x2)/2, y),
    arrowprops=dict(arrowstyle="-|>", lw=1.2),
    ha="center", va="bottom", fontsize=10
)

plt.tight_layout()
plt.savefig("results.png", dpi=200)
print("✅ Plot saved as results.png")
