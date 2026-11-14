# Синтетические проверки контракта 1.0.0

```powershell
python -m pytest src/pipetwin/agent2_physics/tests -q -W error
python -m pipetwin.agent2_physics.synthetic_check --output artifacts/agent2_physics/synthetic-check-001
```

Тесты проверяют обязательные модели и дополнительный hybrid_scale на канонических
синтетических frame/exposure. Второй запуск сохраняет 16 отдельных fit/replay
(4 модели, 2 сценария, 2 missing_policy), исходные синтетические таблицы,
предсказания, параметры, время и joblib. Каталог должен быть новым.

Реальные CSV не читаются, общие фолды и scorer не заменяются. Эти проверки
не являются оценкой качества официальных кейсов или переноса на другую трубу.

Второй этап добавляет проверки сильного сдвига режимных признаков и номинала
8 -> 4.5 мм, train-only защиты, отключённого исходного режима, совместимости
старых joblib-объектов, инвариантности общих ключей/маски после сохранения CSV.
Финальная общая проверка: 211 тестов, из них 96 в этом пакете.

Реальное сравнение запускается отдельно после общей prepare-cases:

```powershell
python -m pipetwin.agent2_physics.shared_comparison --prepared artifacts/agent2_physics/shared-inputs-d443e4f --output artifacts/agent2_physics/shared-comparison-v2
```

Wrapper вызывает общий evaluator без изменения фолдов или scorer. Данные для
этого запуска и ограничения результатов описаны в RESULTS.md.
