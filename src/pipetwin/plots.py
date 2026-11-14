"""Figures from the submitted predictions and observed validation labels."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .case1 import FINAL_MODEL as CASE1_MODEL
from .case2 import FINAL_MODEL as CASE2_MODEL


BLUE = "#17657d"
INK = "#172e3f"
MUTED = "#697887"
PALE = "#dce5eb"


def plot_profiles(output, warm, cold):
    figure = Figure(figsize=(12, 5.6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1)
    figure.suptitle("Прогноз минимальной остаточной толщины по участкам", fontsize=15, color=INK)
    for number, (ax, frame) in enumerate(zip(axes, (warm, cold)), start=1):
        kilometres = frame.center_m / 1000
        ax.plot(kilometres, frame.thickness_pred, color=BLUE, lw=1.15, label="Прогноз")
        if number == 1:
            anchor = frame.anchor_observed
            ax.scatter(kilometres[anchor], frame.loc[anchor, "anchor_mm"], s=5, alpha=.35,
                       color=MUTED, linewidths=0, label="ВТД 06.07.2022")
        ax.axhline(frame.nominal_mm.iloc[0], color=INK, ls="--", lw=.8, label="Номинал по паспорту")
        ax.set_title(
            f"Кейс {number} · {frame.id.iloc[0]} · {frame.target_date.iloc[0]:%d.%m.%Y} · {len(frame):,} ячеек".replace(",", " "),
            loc="left", color=INK,
        )
        ax.set(ylabel="Толщина, мм", xlabel="Расстояние от начала участка, км")
        ax.set_xlim(0, frame.end_m.max() / 1000)
        ax.grid(axis="y", color=PALE, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", ncols=3, frameon=False, fontsize=8.5, labelcolor=INK)
    figure.savefig(output / "profiles.png", dpi=170)


def _bars(ax, x, values, width, color, *, label=None):
    bars = ax.bar(x, values, width, color=color, label=label)
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2, color=INK)


def plot_validation(output, historic, transfer):
    figure = Figure(figsize=(12, 5), layout="constrained")
    FigureCanvasAgg(figure)
    left, right = figure.subplots(1, 2)
    figure.suptitle("Знаковая корреляция на отложенных измерениях", fontsize=15, color=INK)
    warm = historic.pivot(index="split", columns="model", values="pearson_r")
    warm = warm.loc[["spatial_2022", "late_2023"]]
    x = np.arange(len(warm))
    for offset, (model, label, color) in enumerate([
        ("persistence", "Прошлое ВТД", "#a8b4be"),
        ("annual_loss_blend", "Смесь для годовой потери", "#658a9c"),
        (CASE1_MODEL, "GAM + CatBoost", BLUE),
    ], start=-1):
        _bars(left, x + .24 * offset, warm[model], .22, color, label=label)
    left.set_xticks(x, ["525 ячеек цели\n508 меток, 4 блока", "Соседние участки\n53 метки, декабрь 2023"])
    left.set_title("Кейс 1 · исторические и поздние ВТД", loc="left", color=INK)
    left.legend(loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK)
    left.set_ylim(min(0.0, warm.min().min() - .08), max(.72, warm.max().max() + .14))

    order = ["donors_2018", "donors_2018_2022", "all_donors_unweighted", CASE2_MODEL]
    comparison = transfer.loc[transfer.held_main.eq("MAIN-3")].set_index("model").loc[order, "pearson_r"]
    x = np.arange(len(order))
    _bars(right, x, comparison, .63, ["#a8b4be", "#8ca7b6", "#658a9c", BLUE])
    right.set_xticks(x, ["2018", "2018, 2022", "Все три\nВТД", "Все три ВТД\nс весами"])
    right.set_title("Кейс 2 · перенос с MAIN-2 на MAIN-3", loc="left", color=INK)
    right.set_xlabel("Обучение на MAIN-2 · оценка по 94 меткам MAIN-3")
    right.set_ylim(min(0.0, comparison.min() - .05), max(.37, comparison.max() + .05))

    for ax in (left, right):
        ax.set_ylabel("Pearson r")
        ax.grid(axis="y", color=PALE, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "validation.png", dpi=170)
