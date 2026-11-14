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
