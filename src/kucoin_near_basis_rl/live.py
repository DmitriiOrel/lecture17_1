from __future__ import annotations

import argparse
import time

import numpy as np

from .baseline import ACTION_TO_POSITION, POSITION_TO_ACTION, BaselinePolicy
from .config import load_config
from .features import FEATURE_COLUMNS, build_feature_frame
from .kucoin_api import KuCoinExecutionClient, KuCoinPublicDataClient
from .qlearning import load_model_artifact


def run_live(config_path: str, model_path: str, paper: bool, once: bool) -> None:
    cfg = load_config(config_path)
    agent, discretizer, obs_columns, _ = load_model_artifact(model_path)

    data_client = KuCoinPublicDataClient(cfg.api)
    execution_client = KuCoinExecutionClient(cfg.api, dry_run=paper)
    baseline = BaselinePolicy(
        enter_zscore=cfg.baseline.enter_zscore,
        exit_zscore=cfg.baseline.exit_zscore,
    )

    current_position = 0
    while True:
        start_dt, end_dt = data_client.utc_lookback(cfg.data.lookback_minutes)
        raw = data_client.fetch_merged_candles(cfg.data, start_dt=start_dt, end_dt=end_dt)
        feature_frame = build_feature_frame(raw, cfg.features)
        if feature_frame.empty:
            raise RuntimeError("Feature frame is empty in live loop.")

        row = feature_frame.iloc[-1]
        obs = row[obs_columns].to_numpy(dtype=float).tolist()
        obs.append(float(current_position))
        observation = np.array(obs, dtype=float)

        state = discretizer.transform(observation)
        baseline_position = baseline.decide_position(float(row["basis_zscore"]), current_position)
        baseline_action = POSITION_TO_ACTION[baseline_position]

        if agent.has_state(state):
            action = agent.greedy_action(state)
        else:
            action = baseline_action

        target_position = ACTION_TO_POSITION[action]
        prices = data_client.fetch_price_snapshot(
            spot_symbol=cfg.data.spot_symbol,
            futures_symbol=cfg.data.futures_symbol,
        )
        spot_size = max(0.0, cfg.execution.quote_notional_usdt / prices.spot_price)
        futures_size = max(1, int(round(spot_size / cfg.execution.futures_contract_multiplier)))

        result = execution_client.rebalance_basis_position(
            current_position=current_position,
            target_position=target_position,
            spot_symbol=cfg.data.spot_symbol,
            futures_symbol=cfg.data.futures_symbol,
            spot_size=spot_size,
            futures_size=futures_size,
            leverage=cfg.execution.leverage,
        )
        if result["changed"]:
            current_position = target_position

        print(
            "tick",
            {
                "timestamp": str(row["timestamp"]),
                "basis": float(row["basis"]),
                "zscore": float(row["basis_zscore"]),
                "action": action,
                "target_position": target_position,
                "paper_mode": paper,
                "orders": result["orders"],
            },
        )

        if once:
            return
        time.sleep(cfg.execution.poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live/paper execution for NEAR basis RL bot on KuCoin.")
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--model", required=True, help="Path to trained model artifact.")
    parser.add_argument("--live", action="store_true", help="If set, sends real orders.")
    parser.add_argument("--once", action="store_true", help="Run single decision cycle and exit.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paper_mode = not args.live
    if args.live and cfg.execution.default_paper_mode:
        print("WARNING: config has default_paper_mode=true, but --live was passed. Sending live orders.")
    run_live(args.config, args.model, paper=paper_mode, once=args.once)


if __name__ == "__main__":
    main()
