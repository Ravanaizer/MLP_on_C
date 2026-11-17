
"""График обучения прямо в терминале (без библиотек).
Запуск: python plot_term.py [training.csv]    (Ctrl+C - выход)
Понимает оба формата лога: step,epoch,train_loss,test_acc и epoch,train_loss,test_acc."""
import csv
import shutil
import sys
import time

args = [a for a in sys.argv[1:] if not a.startswith("--")]
PATH = args[0] if args else "training.csv"
ONCE = "--once" in sys.argv


def read_log(path):
    xs, loss, acc = [], [], []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    x = float(r.get("step") or r["epoch"])
                    l, a = float(r["train_loss"]), float(r["test_acc"])
                except (KeyError, ValueError, TypeError):
                    continue  # заголовок не того формата или недописанная строка
                xs.append(x)
                loss.append(l)
                acc.append(a)
    except FileNotFoundError:
        pass
    return xs, loss, acc


def chart(title, xs, ys, w, h, color, zero, fmt):
    lo, hi = (0.0, max(ys) * 1.05) if zero else (min(ys), max(ys))
    pad = (hi - lo) * 0.1 or 0.01
    lo, hi = (lo, hi) if zero else (lo - pad, min(hi + pad, 1.0))
    x0, x1 = xs[0], xs[-1]
    pw = w - 9

    def col(x):
        return 0 if x1 == x0 else round((x - x0) / (x1 - x0) * (pw - 1))

    def row(v):
        return max(0, min(h - 1, round((hi - v) / (hi - lo) * (h - 1))))

    grid = [[" "] * pw for _ in range(h)]
    for i in range(len(xs)):
        ca, ra = col(xs[i]), row(ys[i])
        if i + 1 < len(xs):  # соединяем соседние точки линейной интерполяцией
            cb, rb = col(xs[i + 1]), row(ys[i + 1])
            for c in range(ca, cb + 1):
                t = 0 if cb == ca else (c - ca) / (cb - ca)
                grid[round(ra + (rb - ra) * t)][c] = "·"
        grid[ra][ca] = "●"

    paint = lambda s: f"\033[38;5;{color}m{s}\033[0m"
    out = [f"\033[1m{title}\033[0m  {paint(fmt(ys[-1]))}"]
    for r in range(h):
        label = fmt(hi - (hi - lo) * r / (h - 1)) if r in (0, h // 2, h - 1) else ""
        out.append(f"{label:>7} │" + paint("".join(grid[r])))
    out.append(" " * 8 + "└" + "─" * pw)
    out.append(" " * 9 + f"{x0:g}".ljust(pw - 8) + f"{x1:g}".rjust(8))
    return "\n".join(out)


def render():
    xs, loss, acc = read_log(PATH)
    size = shutil.get_terminal_size((100, 40))
    if not xs:
        return f"Жду данные в {PATH} ...  (нужны колонки train_loss и test_acc)"
    h = max(6, (size.lines - 12) // 2)
    w = min(size.columns, 120)
    head = f"{PATH}: {len(xs)} точек, последняя x = {xs[-1]:g}"
    return "\n\n".join([
        head,
        chart("Loss на обучении", xs, loss, w, h, 205, True, lambda v: f"{v:.4f}"),
        chart("Точность на тесте", xs, acc, w, h, 75, False, lambda v: f"{v * 100:.2f}%"),
    ])


if __name__ == "__main__":
    try:
        while True:
            text = render()
            print(text if ONCE else "\033[H\033[J" + text, flush=True)
            if ONCE:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
