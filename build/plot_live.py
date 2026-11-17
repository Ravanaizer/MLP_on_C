"""Живой график обучения: python plot_live.py [training.csv]
Файл перечитывается раз в секунду, пока C-программа его дописывает."""
import csv
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

PATH = sys.argv[1] if len(sys.argv) > 1 else "training.csv"
BG, CARD, TEXT, MUTED = "#0f1218", "#181c25", "#e8ebf3", "#8b93a7"
C_LOSS, C_ACC = "#ff6b9d", "#5a9bff"


def read_log(path):
    rows = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append((int(r["step"]), int(r["epoch"]),
                                 float(r["train_loss"]), float(r["test_acc"])))
                except (ValueError, KeyError, TypeError):
                    pass  # недописанная строка
    except FileNotFoundError:
        pass
    return rows


def style(ax, title, color):
    ax.set_facecolor(CARD)
    ax.set_title(title, color=TEXT, loc="left", fontsize=12, pad=10)
    ax.tick_params(colors=MUTED)
    ax.grid(color="#262c3a", linewidth=0.8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("батч", color=MUTED)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=BG)
style(ax1, "Loss на обучении", C_LOSS)
style(ax2, "Точность на тесте", C_ACC)
(line1,) = ax1.plot([], [], color=C_LOSS, lw=2.5)
(line2,) = ax2.plot([], [], color=C_ACC, lw=2.5)
dot1, = ax1.plot([], [], "o", color=C_LOSS, ms=8)
dot2, = ax2.plot([], [], "o", color=C_ACC, ms=8)
fig.tight_layout(pad=2.5)
fills = []


def update(_):
    rows = read_log(PATH)
    if not rows:
        return
    steps = [r[0] for r in rows]
    loss = [r[2] for r in rows]
    acc = [r[3] for r in rows]

    line1.set_data(steps, loss)
    line2.set_data(steps, acc)
    dot1.set_data([steps[-1]], [loss[-1]])
    dot2.set_data([steps[-1]], [acc[-1]])

    for f in fills:
        f.remove()
    fills.clear()
    fills.append(ax1.fill_between(steps, loss, color=C_LOSS, alpha=0.12))
    fills.append(ax2.fill_between(steps, acc, min(acc), color=C_ACC, alpha=0.12))

    for ax, ys in ((ax1, loss), (ax2, acc)):
        ax.set_xlim(0, max(steps[-1], 1) * 1.02)
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.15 or 0.01
        ax.set_ylim(0 if ax is ax1 else lo - pad, hi + pad)

    fig.suptitle(f"эпоха {rows[-1][1]}   loss {loss[-1]:.4f}   точность {acc[-1] * 100:.2f}%",
                 color=TEXT, fontsize=13)


anim = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)
if __name__ == "__main__":
    plt.show()
