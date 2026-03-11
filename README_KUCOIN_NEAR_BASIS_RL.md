# KuCoin NEAR Basis RL Bot (from scratch)

Этот модуль реализует бота для basis-trading между:
- `NEAR-USDT` (spot, KuCoin)
- `NEARUSDTM` (futures perpetual, KuCoin)

## Что внутри
- Публичный сбор данных со свечей KuCoin spot/futures.
- Фичи среды: объемы, волатильность, z-score базиса, momentum базиса.
- Baseline-алгоритм (z-score mean reversion).
- RL-обучение (tabular Q-learning), где baseline встроен как:
  - imitation exploration;
  - bonus к reward за совпадение действий с baseline.
- Live/paper loop для принятия решений и отправки хеджированных ордеров.

## Источник API KuCoin
Для реального исполнения ордеров используется официальный SDK из GitHub-организации KuCoin:
- https://github.com/Kucoin/kucoin-python-sdk
- https://github.com/Kucoin/kucoin-futures-python-sdk

Публичные market-data берутся через официальные REST endpoints KuCoin.

## Установка
```powershell
pip install --upgrade pip wheel "setuptools<81"
pip install -r requirements.kucoin_near_basis_rl.txt
```

## Переменные окружения (для live-торговли)
```powershell
$env:KUCOIN_API_KEY="..."
$env:KUCOIN_API_SECRET="..."
$env:KUCOIN_API_PASSPHRASE="..."
```

## Обучение
```powershell
$env:PYTHONPATH="src"
python -m kucoin_near_basis_rl.train `
  --config config/kucoin_near_basis_rl.json `
  --model-out models/near_basis_qlearning.json `
  --features-out reports/near_basis_features.csv
```

Опционально можно тренировать по своему CSV:
```powershell
python -m kucoin_near_basis_rl.train `
  --config config/kucoin_near_basis_rl.json `
  --model-out models/near_basis_qlearning.json `
  --source-csv data/near_basis_history.csv
```

## Paper/live запуск
Paper (по умолчанию):
```powershell
$env:PYTHONPATH="src"
python -m kucoin_near_basis_rl.live `
  --config config/kucoin_near_basis_rl.json `
  --model models/near_basis_qlearning.json
```

Один тик (удобно для проверки):
```powershell
python -m kucoin_near_basis_rl.live `
  --config config/kucoin_near_basis_rl.json `
  --model models/near_basis_qlearning.json `
  --once
```

Live-торговля:
```powershell
python -m kucoin_near_basis_rl.live `
  --config config/kucoin_near_basis_rl.json `
  --model models/near_basis_qlearning.json `
  --live
```

## Важно по рискам
- Basis-trading не безрисковая: возможны расширения базиса, ликвидации и проблемы с ликвидностью.
- Перед live-режимом проверь размер контракта `futures_contract_multiplier` и комиссии для вашего аккаунта.
- Начинайте с paper-режима и маленького notional.
