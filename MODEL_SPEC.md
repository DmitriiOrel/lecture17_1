# NEAR Basis RL Model Spec (KuCoin Spot + Futures)

## 1) Instruments and cadence
- Spot: `NEAR-USDT`
- Futures perpetual: `NEARUSDTM`
- Default cycle: `1m`
- Alternative profile: `15m` in `config/micro_near_v1.json`

## 2) Feature space (state)
From synchronized spot/futures candles:
- `basis = (F - S) / S`
- `basis_zscore`: rolling z-score of basis
- `spot_volatility`: rolling std of spot log-return
- `futures_volatility`: rolling std of futures log-return
- `volume_imbalance`: normalized futures-vs-spot rolling volume diff
- `basis_momentum`: lagged basis diff
- `position`: current position in `{-1, 0, +1}`

## 3) Baseline policy (teacher)
Mean-reversion baseline by z-score:
- if `z >= enter_z` -> short basis (`position = -1`)
- if `z <= -enter_z` -> long basis (`position = +1`)
- if `|z| <= exit_z` -> flat (`position = 0`)

## 4) RL policy
Tabular Q-learning over discretized feature bins.

Actions:
- `0 -> position -1`
- `1 -> position 0`
- `2 -> position +1`

Training enhancements:
- imitation exploration (follow baseline with decaying probability)
- imitation reward bonus when RL action matches baseline action

## 5) Reward
For step `t -> t+1`:
- `pnl = target_position * (basis_{t+1} - basis_t)`
- `rebalance_cost = fee_rate * |target_position - previous_position|`
- `risk_cost = risk_penalty * |target_position| * |zscore_t|`
- `reward = pnl - rebalance_cost - risk_cost + baseline_bonus_if_matched`

## 6) Execution model
Position semantics:
- `position +1`: long basis (`BUY futures`, `SELL spot`)
- `position -1`: short basis (`SELL futures`, `BUY spot`)
- `position 0`: close both legs

Sizing:
- spot size from quote notional: `spot_qty = quote_notional_usdt / spot_price`
- futures contracts from multiplier: `fut_contracts = round(spot_qty / contract_multiplier)`

## 7) Risk notes
- Start in `shadow` mode only.
- Use small quote notional.
- Validate contract multiplier and fees for your KuCoin account tier.
- Use API keys with IP whitelist in production.
