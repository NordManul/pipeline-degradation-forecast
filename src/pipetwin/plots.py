"""Figures from the submitted predictions and observed validation labels."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .case1 import FINAL_MODEL as CASE1_MODEL
from .case2 import CATBOOST_CONTROL
from .models.cold_profile import BOOST_MODEL, FINAL_MODEL as CASE2_MODEL


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

    cold = transfer.loc[transfer.split.eq("holdout_main")].pivot(
        index=["held_main", "campaign"], columns="model", values="pearson_r"
    )
    x = np.arange(len(cold))
    for offset, (model, label, color) in enumerate([
        (CATBOOST_CONTROL, "CatBoost", "#a8b4be"),
        (BOOST_MODEL, "XGBoost", "#658a9c"),
        (CASE2_MODEL, "XGBoost + KNN", BLUE),
    ], start=-1):
        _bars(right, x + .25 * offset, cold[model], .23, color, label=label)
    right.set_xticks(x, [f"{main}\n{date:%m.%Y}" for main, date in cold.index])
    right.set_title("Кейс 2 · целая магистраль в контроле", loc="left", color=INK)
    right.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK)
    right.set_ylim(min(0.0, cold.min().min() - .04), max(.62, cold.max().max() + .1))

    for ax in (left, right):
        ax.set_ylabel("Pearson r")
        ax.grid(axis="y", color=PALE, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "validation.png", dpi=170)


def plot_observed(output, warm, cold):
    figure = Figure(figsize=(12, 4.8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    figure.suptitle("Прогноз и факт на отложенных ВТД", fontsize=15, color=INK)
    sets = [("Кейс 1 · блоки цели", warm),
            ("Кейс 2 · MAIN-2", cold.loc[cold.main_id.eq("MAIN-2")]),
            ("Кейс 2 · MAIN-3", cold.loc[cold.main_id.eq("MAIN-3")])]
    for ax, (title, frame) in zip(axes, sets):
        actual, predicted = frame.y_mm.to_numpy(), frame.thickness_pred.to_numpy()
        limit = min(actual.min(), predicted.min()) - .1, max(actual.max(), predicted.max()) + .1
        ax.plot(limit, limit, ls="--", color=MUTED, linewidth=.8)
        ax.scatter(actual, predicted, color=BLUE, s=12, alpha=.33, linewidths=0)
        ax.set(xlim=limit, ylim=limit, xlabel="Измеренная толщина, мм", ylabel="Прогноз, мм")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{title}\n{len(frame)} наблюдаемых минимумов", loc="left", color=INK)
        ax.grid(color=PALE, linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "predicted_observed.png", dpi=170)
