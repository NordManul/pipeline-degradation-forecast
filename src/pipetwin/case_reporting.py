"""Compact static scientific plots for the two case-cutoff experiments."""

from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_case_report(output, metrics, forecasts, selections):
    cases = list(selections)
    fig, axes = plt.subplots(len(cases), 2, figsize=(14, 5 * len(cases)), squeeze=False, layout="constrained")
    for row, case_id in enumerate(cases):
        table = metrics.loc[metrics.case_id.eq(case_id) & metrics.output.eq("bounded")].copy()
        # A negative r is not a useful model merely because its square is large.
        table = table.loc[table.pearson_r.gt(0)].sort_values(["pearson_squared", "mae_mm"], ascending=[False, True]).head(12).iloc[::-1]
        names = table.model.to_numpy()
        chosen = selections[case_id]["model"]
        colors = ["#237a57" if name == chosen else "#5186b2" for name in names]
        axes[row, 0].barh(names, table.pearson_squared, color=colors)
        axes[row, 0].set_xlabel("Squared Pearson r (positive r only)")
        axes[row, 0].set_title(f"{case_id}: spatial OOF, n={int(table.n.iloc[0]) if len(table) else 0}")
        axes[row, 1].barh(names, table.mae_mm, color=colors)
        axes[row, 1].set_xlabel("MAE, mm (lower is better)")
        axes[row, 1].set_title("Same models and observed score mask")
        for ax in axes[row]:
            ax.grid(axis="x", alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle("Auxiliary spatial comparison; future / new-pipeline accuracy is unknown", fontsize=13)
    fig.savefig(output / "comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(len(cases), 1, figsize=(13, 3.7 * len(cases)), squeeze=False, layout="constrained")
    for row, case_id in enumerate(cases):
        frame = forecasts.loc[forecasts.case_id.eq(case_id)].sort_values("center_m")
        ax = axes[row, 0]
        ax.plot(frame.center_m, frame.nominal_mm, color="#7b838b", linestyle="--", label="Passport nominal")
        measured = frame.anchor_observed.to_numpy(dtype=bool)
        if measured.any():
            ax.scatter(frame.loc[measured, "center_m"], frame.loc[measured, "anchor_mm"], s=5, alpha=.35, color="#526070", label="Previous ILI (different date)")
        ax.plot(frame.center_m, frame.thickness_pred, color="#237a57", linewidth=1, label="Provisional forecast")
        ax.set_title(f"{case_id}: {selections[case_id]['model']} | target ILI unavailable")
        ax.set_xlabel("Local distance, m")
        ax.set_ylabel("Remaining wall, mm")
        ax.set_ylim(0, float(frame.nominal_mm.max()) * 1.08)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(output / "forecast_profiles.png", dpi=150)
    plt.close(fig)
    parts = ["<!doctype html><html lang='ru'><meta charset='utf-8'><title>PipeTwin: общие кейсы</title>",
             "<style>body{max-width:1300px;margin:32px auto;font:16px system-ui;padding:0 20px}img{width:100%}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd}.note{background:#fff0d4;padding:16px}</style>",
             "<h1>Кейсы после уточнений ментора</h1><p class='note'>Фолды — вспомогательная пространственная проверка. Независимая точность на будущей дате и новой трубе не измерена. Целевые ВТД отсутствуют; сетка предварительная, официальный шаблон ожидается.</p>",
             "<img src='comparison.png' alt='Сравнение на общем OOF'><img src='forecast_profiles.png' alt='Предварительные профили'>"]
    for case_id in cases:
        selection = selections[case_id]
        parts.append(f"<h2>{escape(case_id)}: {escape(selection['model'])}</h2>")
        if selection["forecast_review_required"]:
            parts.append("<p class='note'>Выбранный по OOF прогноз выродился в константу или целиком вышел за физические границы. Требуется проверка переноса; автоматически подменять победителя по недоступным меткам нельзя.</p>")
        table = metrics.loc[metrics.case_id.eq(case_id) & metrics.output.eq("bounded")].sort_values("pearson_squared", ascending=False)
        parts.append(table[["model", "n", "pearson_r", "pearson_squared", "mae_mm", "bias_mm", "recall_thin_10pct"]].to_html(index=False, float_format=lambda x: f"{x:.4f}", na_rep="undefined"))
    parts.append("<p>Сырые прогнозы, метрики до отсечки, OOF всех ячеек, фолды, параметры и диапазоны сохранены в CSV/JSON. Смеси имеют фиксированные веса. Единого конкурсного балла без официального scorer нет.</p></html>")
    (output / "report.html").write_text("\n".join(parts), encoding="utf-8")
