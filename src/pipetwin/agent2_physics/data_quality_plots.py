"""Compact diagnostic figures for the measurement and source audit."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


BLUE = "#146783"
ORANGE = "#bf5a26"
GRAY = "#a2a9ae"


def write_plots(output, counts, inspection, availability, train1, warm, cold):
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.grid": False}):
        dates = counts.date.dt.strftime("%Y-%m-%d")
        groups = [f"{main}\n{date}" for main, date in zip(counts.main_id, dates)]
        x = np.arange(len(groups))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        axes[0].bar(x, counts.records / counts.cells, color=BLUE)
        axes[0].set(title="Сколько записей формируют один минимум", ylabel="Записей / размеченную ячейку",
                    xticks=x, xticklabels=groups)
        axes[1].bar(x, counts.corrosion_minima, color=BLUE, label="Коррозия, включая кластеры")
        axes[1].bar(x, counts.cells - counts.corrosion_minima, bottom=counts.corrosion_minima,
                    color=ORANGE, label="Остальные типы")
        axes[1].set(title="Тип записи с минимальной толщиной", ylabel="10-метровых ячеек",
                    xticks=x, xticklabels=groups)
        axes[1].legend(frameon=False)
        save(fig, output / "inspection_composition.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for date, color in [("2018-09-23", BLUE), ("2022-07-06", ORANGE)]:
            subset = inspection.loc[inspection.date.eq(date) & inspection.kind.eq("Metal loss")]
            depths = subset.depth_fraction.to_numpy() * 100
            axes[0].hist(depths, bins=np.arange(9.5, 65, 1), density=True, histtype="step",
                         linewidth=1.8, color=color, label=f"{date} (n={len(depths):,})")
        axes[0].set(title="Распределение глубин отдельных дефектов", xlabel="Глубина, % стенки", ylabel="Плотность")
        axes[0].legend(frameon=False)
        observed = train1.loc[train1.observed & train1.anchor_observed]
        loss = observed.anchor_mm - observed.y_mm
        axes[1].hist([loss[loss < 0], loss[loss >= 0]], bins=np.arange(-4, 4.01, 0.125),
                     stacked=True, color=[ORANGE, BLUE], label=["Минимум стал выше", "Минимум снизился или равен"])
        axes[1].axvline(0, color="#222222", linewidth=1)
        axes[1].set(title=f"Изменение минимума в одной ячейке (n={len(loss)})",
                    xlabel="Толщина 2018 − толщина 2022, мм", ylabel="Ячеек")
        axes[1].legend(frameon=False)
        save(fig, output / "case1_measurement_shift.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for main, group in availability.groupby("main_id"):
            axes[0].scatter(group.calendar_days / 365.25, group.known_water_fraction * 100,
                            label=main, alpha=.8, s=45)
            axes[1].scatter(group.calendar_days / 365.25, group.coverage * 100,
                            label=main, alpha=.8, s=45)
        axes[0].set_yscale("log")
        axes[0].set(title="Обводнённость известного режима", xlabel="Возраст на дату ВТД или прогноза, лет",
                    ylabel="Вода в потоке, % (лог. шкала)")
        axes[1].set(title="Покрытие календаря режимом TECH", xlabel="Возраст, лет", ylabel="Дней с записью, %")
        axes[1].legend(frameon=False)
        save(fig, output / "case2_exposure_shift.png")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for ax, table, validation, highlight in [
            (axes[0], warm, "Кейс 1 · пространственные фолды 2022", "gam_catboost_direct"),
            (axes[1], cold.loc[cold.positive_campaigns.eq(4)], "Кейс 2 · удержание целой MAIN", "xgb15"),
        ]:
            ax.scatter(table.mae_mm, table.pearson_squared, color=GRAY, label="Проверенные варианты", alpha=.65)
            ref = table.loc[table.model.eq(highlight)]
            ax.scatter(ref.mae_mm, ref.pearson_squared, color=BLUE, s=75, zorder=3, label="Текущий контроль")
            top = table.iloc[0]
            if top.model != highlight:
                ax.scatter(top.mae_mm, top.pearson_squared, color=ORANGE, marker="D", s=55,
                           zorder=3, label="Лучший по Pearson²")
            ax.set(title=validation, xlabel="MAE, мм · меньше лучше", ylabel="Pearson² · больше лучше")
            ax.legend(frameon=False, loc="best")
        save(fig, output / "model_comparison.png")


def save(fig, path):
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
