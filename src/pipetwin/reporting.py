"""Static, exportable research figures and a self-contained HTML run report."""

import base64
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_TITLES = {
    "development": "2018 → 2022: spatial development OOF",
    "cold_development": "Cold start: spatial OOF on MAIN-2 (2018)",
    "late": "2022 → 2023: later-campaign transfer",
    "cold_test": "Cold start: unseen MAIN-3 (2023)",
}


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _leaderboard(metrics, split, champion, path):
    table = metrics.loc[metrics.split.eq(split)].copy()
    # Negative-r candidates are visibly marked, never promoted by squaring their sign away.
    table["positive_r"] = table.pearson_r.gt(0)
    table = table.sort_values(["positive_r", "pearson_squared", "mae_mm"], ascending=[False, False, True], na_position="last")
    labels = [name + ("  *" if name == champion else "") for name in table.model]
    y = np.arange(len(table))
    fig, axes = plt.subplots(1, 2, figsize=(13, 9), sharey=True)
    colors = ["#159570" if name == champion else ("#426cba" if r >= 0 else "#d88030") for name, r in zip(table.model, table.pearson_r)]
    axes[0].barh(y, table.pearson_squared.fillna(0), color=colors)
    finite_scores = table.pearson_squared.dropna()
    axes[0].set_xlim(0, max(0.15, float(finite_scores.max()) * 1.55 if len(finite_scores) else 0.15))
    for index, row in enumerate(table.itertuples()):
        label = f"{row.pearson_squared:.3f}  (r={row.pearson_r:+.3f})" if np.isfinite(row.pearson_r) else "undefined: constant"
        axes[0].text((row.pearson_squared if np.isfinite(row.pearson_squared) else 0) + axes[0].get_xlim()[1] * 0.015, index, label, va="center", fontsize=8)
    axes[1].barh(y, table.mae_mm, color=colors)
    axes[1].set_xlim(0, max(0.1, float(table.mae_mm.max()) * 1.23))
    for index, value in enumerate(table.mae_mm):
        axes[1].text(value + axes[1].get_xlim()[1] * 0.015, index, f"{value:.3f}", va="center", fontsize=9)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Squared Pearson correlation — higher, with positive r")
    axes[1].set_xlabel("MAE, mm — lower")
    for axis in axes:
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(f"{SPLIT_TITLES[split]}\nSame {int(table.n.iloc[0])} reported target cells for every model", fontsize=14)
    fig.text(0.5, 0.012, "* Frozen choice from development only. Orange = negative signed correlation. Unknown target cells are unscored.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    _save(fig, path)


def _scatter(tables, selection, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, split in zip(axes, ["development", "late", "cold_test"]):
        scenario = "cold" if split == "cold_test" else "between"
        name = selection[scenario]["champion"]
        table = tables[split].loc[lambda frame: frame.observed]
        axis.scatter(table.y_mm, table[name], s=17, alpha=0.6, color="#426cba", edgecolors="none")
        low, high = min(table.y_mm.min(), table[name].min()), max(table.y_mm.max(), table[name].max())
        axis.plot([low, high], [low, high], linestyle="--", color="#666666", linewidth=1)
        axis.set_title(f"{split} | {name}\nn={len(table)}", fontsize=11)
        axis.set_xlabel("Observed thickness, mm")
        axis.set_ylabel("Predicted thickness, mm")
        axis.grid(alpha=0.2)
    fig.suptitle("Frozen model: predicted vs observed thickness", fontsize=14)
    fig.tight_layout()
    _save(fig, path)


def _profile(table, champion, path):
    fig, axis = plt.subplots(figsize=(14, 4.8))
    for number, (_, group) in enumerate(table.groupby("id", sort=False)):
        group = group.sort_values("chain_position_m")
        axis.plot(group.chain_position_m / 1000, group[champion], color="#159570", linewidth=1.2, label=champion if number == 0 else None)
        if champion != "persistence":
            axis.plot(group.chain_position_m / 1000, group.persistence, color="#8a91a0", linewidth=0.8, alpha=0.7, label="persistence" if number == 0 else None)
        observed = group.loc[group.observed]
        axis.scatter(observed.chain_position_m / 1000, observed.y_mm, color="#283a60", s=12, alpha=0.65, label="observed target" if number == 0 else None)
    axis.set(title="MAIN-2 development: buffered out-of-fold predictions along the pipe", xlabel="Main-chain distance, km", ylabel="Thickness, mm")
    axis.legend(loc="lower left", ncols=3)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    _save(fig, path)


def _coverage(coverage, path):
    fig, axis = plt.subplots(figsize=(11, 4.8))
    y = np.arange(len(coverage))
    axis.barh(y, coverage.observed, color="#426cba", label="Reported thickness")
    axis.barh(y, coverage.cells - coverage.observed, left=coverage.observed, color="#dce3ed", label="Unknown thickness")
    labels = coverage.main_id + " | " + pd.to_datetime(coverage.date).dt.strftime("%Y-%m-%d")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    for i, row in enumerate(coverage.itertuples()):
        axis.text(row.cells + 15, i, f"{row.observed}/{row.cells}", va="center", fontsize=10)
    axis.set_xlim(0, coverage.cells.max() * 1.15)
    axis.set(title="Target coverage: absence of a reported defect is not a healthy-wall label", xlabel="Local 10-metre cells")
    axis.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, path)


def create_report(output, metrics, tables, selection, coverage, intervals):
    output = Path(output)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    plots = []
    for split in SPLIT_TITLES:
        champion = selection["cold" if split.startswith("cold") else "between"]["champion"]
        filename = f"leaderboard_{split}.png"
        _leaderboard(metrics, split, champion, output / "plots" / filename)
        plots.append(filename)
    _scatter(tables, selection, output / "plots/scatter.png")
    _profile(tables["development"], selection["between"]["champion"], output / "plots/profile.png")
    _coverage(coverage, output / "plots/coverage.png")
    plots += ["scatter.png", "profile.png", "coverage.png"]
    correlation = pd.read_csv(output / "between_residual_correlations.csv", index_col=0)
    fig, axis = plt.subplots(figsize=(11, 10))
    im = axis.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(correlation)), correlation.columns, rotation=60, ha="right", fontsize=8)
    axis.set_yticks(range(len(correlation)), correlation.index, fontsize=8)
    axis.set_title("Development OOF residual correlation — ensemble diversity")
    fig.colorbar(im, ax=axis, shrink=0.75)
    fig.tight_layout()
    _save(fig, output / "plots/residual_correlation.png")
    plots.append("residual_correlation.png")
    chosen_rows = []
    for split in ["development", "late", "cold_development", "cold_test"]:
        model = selection["cold" if split.startswith("cold") else "between"]["champion"]
        chosen_rows.append(metrics.loc[metrics.split.eq(split) & metrics.model.eq(model)].iloc[0])
    chosen = pd.DataFrame(chosen_rows)
    lines = ["# Исторический прототип PipeTwin", "",
             "Выбор модели и весов выполнен по пространственным OOF-прогнозам до оценки поздней кампании и другой трубы.", "",
             "| Проверка | Зафиксированная модель | n | Pearson r | Pearson² | MAE, мм |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in chosen.itertuples():
        correlation_text = f"{row.pearson_r:.4f}" if np.isfinite(row.pearson_r) else "не определён"
        squared = f"{row.pearson_squared:.4f}" if np.isfinite(row.pearson_squared) else "не определён"
        lines.append(f"| {row.split} | {row.model} | {row.n} | {correlation_text} | {squared} | {row.mae_mm:.4f} |")
    lines += ["", "Сравниваются 16 отдельных кандидатов и два ансамбля. Рейтинг всех моделей находится в `metrics.csv`.",
              "Каждый кандидат оценён на одной и той же маске наблюдаемых целей; пропуски не заполнены номиналом.",
              "Предсказания ограничены диапазоном от 0 до проверенного паспортного номинала.", "",
              "Development — пространственная проверка одного перехода 2018→2022 с буфером 100 м.",
              "Это не обучение до 2022 года с полностью неизвестным обследованием 2022 года.",
              "Late — перенос на два коротких участка 2022→2023. Cold test — перенос с MAIN-2 на MAIN-3 без прошлого ВТД.",
              "Размечено 94 из 2185 ячеек cold test; большинство записей этой кампании относится к производственным дефектам.",
              "Показатели этой выборки нельзя считать доказательством прогноза будущей внутренней коррозии.", "",
              "Режимы используются строго до даты T; для будущего T потребуется сценарий будущего режима.",
              "Поздние данные ранее проходили аудит. После просмотра этих результатов повторный подбор будет исследовательским.",
              "ID, даты, сетка и состав официальной конкурсной цели пока не подтверждены; `selected_predictions.csv` — исторический результат.", "",
              "Интервалы построены по 10%/90% квантилям OOF-остатков выбранной модели. Подбор модели и калибровка используют одни OOF-данные;",
              "это диагностические интервалы без гарантии 80% покрытия, особенно при переносе на новую трубу.", ""]
    for split, values in intervals.items():
        lines.append(f"- {split}: покрытие интервала {values['coverage_80']:.1%}, средняя ширина {values['mean_width_mm']:.3f} мм.")
    lines += ["", "## Графики", ""] + [f"![{name}](plots/{name})\n" for name in plots]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    images = "\n".join(f'<figure><img alt="{escape(name)}" src="data:image/png;base64,{base64.b64encode((output / "plots" / name).read_bytes()).decode()}"></figure>' for name in plots)
    notes = " ".join(lines[12:25])
    html = f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>PipeTwin — исторический эксперимент</title>
<style>body{{max-width:1400px;margin:32px auto;padding:0 24px;font:16px/1.55 system-ui;color:#18304a;background:#f5f7fb}}h1{{font-size:30px}}table{{border-collapse:collapse;background:white;font-size:14px}}td,th{{padding:8px 12px;border:1px solid #dae1eb;text-align:right}}figure{{margin:24px 0;background:white;border-radius:12px;padding:12px}}img{{width:100%;height:auto}}.note{{padding:18px;background:#e7edf7;border-radius:10px}}</style>
<h1>PipeTwin: сравнение моделей на исторических данных</h1>
<p>Звёздочкой на графиках отмечен выбор по development. Показатели поздней кампании и другой трубы не использованы для его изменения.</p>
{chosen.to_html(index=False, float_format=lambda value: f"{value:.4f}", na_rep="не определён", border=0)}
<p class="note">{escape(notes)}</p>{images}
<h2>Полная таблица</h2>{metrics.to_html(index=False, float_format=lambda value: f"{value:.4f}", na_rep="не определён", border=0)}</html>'''
    (output / "report.html").write_text(html, encoding="utf-8")
