"""Reproducible figures from frozen forecasts and held-out measurements."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .validation import regression_metrics

INK, MUTED, PALE = "#192c40", "#647587", "#e3e9ee"
TEAL, PURPLE, GOLD, GREY, BG = "#087e8b", "#7258c9", "#c17e16", "#b2bdc8", "#f5f7fa"
MODELS = {
    "persistence": ("Прошлое ВТД", GREY),
    "annual_loss_blend": ("Годовая потеря", PURPLE),
    "direct_blend": ("GAM + CatBoost", TEAL),
    "catboost_weighted": ("CatBoost", GREY),
    "xgboost": ("XGBoost", PURPLE),
    "xgboost_knn": ("XGBoost + KNN", TEAL),
    "random_forest": ("Random Forest", GOLD),
}


def _figure(title, subtitle, size=(13, 7)):
    figure = Figure(figsize=size, facecolor=BG)
    FigureCanvasAgg(figure)
    figure.text(.055, .96, title, fontsize=20, weight="bold", color=INK, va="top")
    figure.text(.055, .91, subtitle, fontsize=10, color=MUTED, va="top")
    return figure


def _style(ax, title=None):
    ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(PALE)
    ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=6)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=PALE, lw=.7)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=12, color=INK, pad=14, weight="bold")


def _save(figure, output, name, note):
    figure.text(.055, .025, note, color=MUTED, fontsize=9, va="bottom")
    figure.savefig(output / name, dpi=160, facecolor=figure.get_facecolor())


def plot_profiles(output, warm, cold, priority):
    figure = _figure("Два участка. Два сценария прогноза.",
                     "Минимальная остаточная толщина в 10-метровой ячейке · целевые ВТД скрыты", (13, 9.5))
    grid = figure.add_gridspec(4, 1, height_ratios=[2.3, 1, 2.3, 1],
                              left=.08, right=.97, top=.83, bottom=.10, hspace=.85)
    for i, frame in enumerate((warm, cold)):
        ax, terrain = figure.add_subplot(grid[2*i]), figure.add_subplot(grid[2*i+1])
        km = frame.center_m / 1000
        chosen = frame.segment_id.isin(priority.loc[priority.id.eq(frame.id.iloc[0]), "segment_id"])
        ax.plot(km, frame.thickness_pred, color=TEAL, lw=1.1, label="Прогноз толщины")
        ax.scatter(km[chosen], frame.loc[chosen, "thickness_pred"], s=13, color=GOLD, zorder=4,
                   label="Приоритетные ячейки")
        if i == 0:
            anchor = frame.anchor_observed
            ax.scatter(km[anchor], frame.loc[anchor, "anchor_mm"], s=7, color=MUTED, alpha=.38,
                       lw=0, label="ВТД 06.07.2022")
        ax.axhline(frame.nominal_mm.iloc[0], color=PURPLE, ls="--", lw=1, label="Номинал")
        ax.set(ylabel="Толщина, мм", ylim=(0, frame.nominal_mm.iloc[0]*1.14), xlim=(0, frame.end_m.max()/1000))
        _style(ax, f"0{i+1} / {frame.id.iloc[0]} · {frame.target_date.iloc[0]:%d.%m.%Y} · {len(frame):,} ячеек".replace(",", " "))
        ax.legend(loc="lower right", ncols=4 if i == 0 else 3, frameon=False, fontsize=8)
        terrain.fill_between(km, 0, frame.altitude_relative_m, color=PURPLE, alpha=.12)
        terrain.plot(km, frame.altitude_relative_m, color=PURPLE, lw=1)
        terrain.set(xlabel="Расстояние от начала участка, км", ylabel="Высота*, м", xlim=ax.get_xlim())
        _style(terrain)
    _save(figure, output, "profiles.png",
          "* Относительно медианы профиля. В кейсе 2 приоритет задаёт Random Forest, толщину — XGBoost + KNN.")


def plot_validation(output, historic, transfer):
    figure = _figure("Качество, проверенное на открытых ВТД",
                     "Pearson² — квадрат корреляции, а не процент точных прогнозов · выше лучше")
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=.08, right=.97, top=.79, bottom=.22, wspace=.23)
    warm = historic.pivot(index="split", columns="model", values="pearson_squared").loc[["spatial_2022", "late_2023"]]
    cold = transfer.loc[transfer.split.eq("holdout_main")].pivot(
        index=["held_main", "campaign"], columns="model", values="pearson_squared")
    for ax, table, names, labels, title in [
        (axes[0], warm, ["persistence", "annual_loss_blend", "direct_blend"],
         ["История · 508 меток\n4 блока + буфер 100 м", "Позднее ВТД · 53 метки\nСоседние участки"], "01 / С прошлым ВТД"),
        (axes[1], cold, ["catboost_weighted", "xgboost", "xgboost_knn"],
         [f"{main}\n{date:%m.%Y}" for main, date in cold.index], "02 / Вся магистраль вне обучения"),
    ]:
        x = np.arange(len(table))
        for offset, name in enumerate(names, start=-1):
            label, color = MODELS[name]
            bars = ax.bar(x + offset*.25, table[name], width=.22, color=color, label=label)
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=4, color=INK)
        ax.set_xticks(x, labels)
        ax.set(ylabel="Pearson²", ylim=(0, .42))
        _style(ax, title)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    _save(figure, output, "validation.png",
          "Seed 42. Контрольные кампании использовались при исследовании моделей. Результат на скрытых целях неизвестен.")


def _validation_sets(warm, late, cold):
    return [("01 / Исторические блоки", warm), ("01 / Поздние соседние участки", late),
            ("02 / Отложена MAIN-2 · три кампании", cold.loc[cold.main_id.eq("MAIN-2")]),
            ("02 / Отложена MAIN-3", cold.loc[cold.main_id.eq("MAIN-3")])]


def plot_observed(output, warm, late, cold):
    figure = _figure("Как прогноз соотносится с измерением",
                     "Каждая точка — наблюдаемый минимум ячейки вне обучения · пунктир — идеальное совпадение", (13, 10))
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=.08, right=.97, top=.83, bottom=.10, wspace=.25, hspace=.48)
    for ax, (title, frame) in zip(axes.flat, _validation_sets(warm, late, cold)):
        y, pred = frame.y_mm.to_numpy(), frame.thickness_pred.to_numpy()
        lower, upper = min(y.min(), pred.min())-.15, max(y.max(), pred.max())+.15
        ax.plot([lower, upper], [lower, upper], ls="--", color=MUTED, lw=1)
        ax.scatter(y, pred, color=TEAL, s=18, alpha=.35, lw=0)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Измеренная толщина, мм", ylabel="Прогноз, мм")
        ax.set_aspect("equal", adjustable="box")
        metrics = regression_metrics(y, pred)
        _style(ax, f"{title}\nn = {len(frame)} · r = {metrics['pearson_r']:.3f} · MAE = {metrics['mae_mm']:.3f} мм")
    _save(figure, output, "predicted_observed.png",
          "На панели MAIN-2 кампании объединены для обзора. В итоговой метрике кейса 2 кампании и магистрали усредняются поровну.")


def plot_dependencies(output, warm_oof, cold_oof, cold):
    figure = _figure("Откуда берётся пространственный сигнал",
                     "Измерения и поведение модели показаны отдельно · наблюдаемая связь не устанавливает причину", (13, 9.8))
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=.08, right=.94, top=.82, bottom=.11, hspace=.47, wspace=.23)
    paired = warm_oof.loc[warm_oof.anchor_observed]
    ax = axes[0, 0]
    ax.scatter(paired.anchor_mm, paired.y_mm, color=TEAL, s=15, alpha=.4, lw=0)
    low, high = min(paired.anchor_mm.min(), paired.y_mm.min())-.1, max(paired.anchor_mm.max(), paired.y_mm.max())+.1
    ax.plot([low, high], [low, high], color=MUTED, ls="--", lw=1)
    ax.set(xlabel="Минимум ячейки в 2018, мм", ylabel="Минимум ячейки в 2022, мм")
    _style(ax, f"01 / Измерения: прошлое и следующее ВТД · n = {len(paired)}")
    ax = axes[0, 1]
    change = paired.y_mm - paired.anchor_mm
    ax.hist(change, bins=25, color=PURPLE, alpha=.85, edgecolor=BG)
    ax.axvline(0, color=INK, ls="--", lw=1)
    ax.set(xlabel="Минимум 2022 − минимум 2018, мм", ylabel="Число ячеек")
    _style(ax, f"01 / Измерения: рост минимума у {(change > 1e-8).sum()} из {len(change)}")
    ax = axes[1, 0]
    for main, color in [("MAIN-2", PURPLE), ("MAIN-3", TEAL)]:
        part = cold_oof.loc[cold_oof.main_id.eq(main)].copy()
        fraction = part.y_mm / part.nominal_mm
        centered = fraction - fraction.groupby([part.id, part.target_date]).transform("mean")
        ax.scatter(part.altitude_relative_m, centered, s=12, lw=0, alpha=.35, color=color, label=main)
    ax.axhline(0, color=MUTED, ls="--", lw=.8)
    ax.set(xlabel="Высота относительно медианы профиля, м", ylabel="Толщина / номинал − среднее*")
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "02 / Измерения: рельеф и толщина")
    ax = axes[1, 1]
    dots = ax.scatter(cold.altitude_relative_m, cold.thickness_pred, c=cold.position_fraction,
                      cmap="viridis", s=10, alpha=.7, lw=0)
    ax.set(xlabel="Высота относительно медианы профиля, м", ylabel="Прогноз толщины, мм")
    bar = figure.colorbar(dots, ax=ax, pad=.025, fraction=.035)
    bar.set_label("Доля длины участка", fontsize=9, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    _style(ax, "02 / Модель: рельеф и прогноз")
    _save(figure, output, "dependencies.png",
          "* Среднее по участку и кампании. Рост зарегистрированного минимума не означает восстановление металла. Валидные метки сохранены.")


def plot_residuals(output, warm, late, cold):
    figure = _figure("Где модель ошибается",
                     "Остаток = прогноз − измерение · выше нуля: модель завышает остаточную толщину", (13, 8.5))
    axes = figure.subplots(2, 2)
    figure.subplots_adjust(left=.08, right=.97, top=.81, bottom=.12, hspace=.53, wspace=.25)
    for ax, (title, frame) in zip(axes.flat, _validation_sets(warm, late, cold)):
        residual = frame.thickness_pred - frame.y_mm
        ax.scatter(frame.y_mm, residual, s=15, color=TEAL, lw=0, alpha=.4)
        ax.axhline(0, color=INK, ls="--", lw=1)
        ax.set(xlabel="Измеренная толщина, мм", ylabel="Ошибка, мм")
        _style(ax, f"{title}\nСреднее смещение {residual.mean():+.3f} мм · n = {len(frame)}")
    _save(figure, output, "residuals.png",
          "Завышение толщины на тонких ячейках особенно значимо для приоритизации. Здесь нет оценки скрытого целевого ВТД.")


def plot_stability(output, metrics):
    figure = _figure("Устойчивость: пространство и случайное зерно",
                     "Знаковый r выявляет обратное ранжирование · три seed проверяют повторяемость кейса 2")
    left, right = figure.subplots(1, 2)
    figure.subplots_adjust(left=.08, right=.97, top=.79, bottom=.22, wspace=.24)
    for model in ("persistence", "annual_loss_blend", "direct_blend"):
        data = metrics.loc[metrics.case.eq("case1") & metrics.model.eq(model)
                           & metrics.split.str.startswith("spatial_fold")].sort_values("split")
        label, color = MODELS[model]
        left.plot(range(1, len(data)+1), data.pearson_r, "o-", lw=1.5, color=color, label=label)
    left.axhline(0, color=INK, ls="--", lw=.8)
    left.set(xlabel="Непрерывный пространственный блок", ylabel="Pearson r", xticks=[1, 2, 3, 4], ylim=(-.16, .85))
    _style(left, "01 / Один блок остаётся трудным")
    data = metrics.loc[metrics.case.eq("case2") & metrics.split.eq("holdout_main")]
    macro = data.groupby(["model", "seed", "held_main"]).pearson_squared.mean().groupby(level=[0, 1]).mean()
    for model in ("xgboost", "xgboost_knn", "random_forest"):
        label, color = MODELS[model]
        part = macro.loc[model]
        right.plot(np.arange(len(part)), part, "o-", color=color, lw=1.5, label=label)
    right.set_xticks(np.arange(len(part)), [str(seed) for seed in part.index])
    right.set(xlabel="Seed", ylabel="Средний Pearson²", ylim=(0, .15))
    _style(right, "02 / Смесь выше XGBoost на трёх seed")
    for ax in (left, right):
        ax.legend(frameon=False, fontsize=9, loc="upper right")
    _save(figure, output, "stability.png",
          "Кейс 2: среднее по кампаниям, затем по магистралям. Разброс по seed не является доверительным интервалом.")
