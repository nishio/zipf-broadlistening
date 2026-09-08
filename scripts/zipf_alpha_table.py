#!/usr/bin/env python3
"""Generate the Zipf alpha comparison table used in README.md.

Model:
- n_r = C / r^alpha
- Real clusters are ranks where n_r >= 1
- M_eff is the largest rank that satisfies n_r >= 1
- C is chosen so that sum_{r=1..M_eff} n_r = N
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Row:
    alpha: float
    m_eff: int
    p1: float
    rank_near_target: int
    rank_max_ge_target: int


def find_m_eff_and_c(n_total: int, alpha: float, max_rank: int) -> tuple[int, float]:
    if n_total < 1:
        raise ValueError("N must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    harmonic = 0.0
    for m in range(1, max_rank + 1):
        harmonic += m ** (-alpha)
        c = n_total / harmonic
        n_m = c / (m ** alpha)
        n_next = c / ((m + 1) ** alpha)
        if n_m >= 1.0 and n_next < 1.0:
            return m, c

    raise RuntimeError(f"Could not find M_eff within max_rank={max_rank}")


def nearest_rank_for_share(m_eff: int, c: float, n_total: int, alpha: float, target_p: float) -> int:
    # n_r / N = c / (N * r^alpha), monotonic in r.
    r_float = (c / (n_total * target_p)) ** (1.0 / alpha)
    r1 = max(1, min(m_eff, math.floor(r_float)))
    r2 = max(1, min(m_eff, math.ceil(r_float)))

    def p(r: int) -> float:
        return c / (n_total * (r**alpha))

    return r1 if abs(p(r1) - target_p) <= abs(p(r2) - target_p) else r2


def max_rank_ge_share(m_eff: int, c: float, n_total: int, alpha: float, target_p: float) -> int:
    if target_p <= 0:
        return m_eff

    r = int((c / (n_total * target_p)) ** (1.0 / alpha))
    r = max(0, min(m_eff, r))
    # Correct rounding drift.
    while r > 0 and c / (n_total * (r**alpha)) < target_p:
        r -= 1
    while r < m_eff and c / (n_total * ((r + 1) ** alpha)) >= target_p:
        r += 1
    return r


def generate_rows(n_total: int, alphas: Iterable[float], target_p: float, max_rank: int) -> List[Row]:
    rows: List[Row] = []
    for alpha in alphas:
        m_eff, c = find_m_eff_and_c(n_total=n_total, alpha=alpha, max_rank=max_rank)
        p1 = c / n_total
        rank_near = nearest_rank_for_share(
            m_eff=m_eff, c=c, n_total=n_total, alpha=alpha, target_p=target_p
        )
        rank_max = max_rank_ge_share(
            m_eff=m_eff, c=c, n_total=n_total, alpha=alpha, target_p=target_p
        )
        rows.append(
            Row(
                alpha=alpha,
                m_eff=m_eff,
                p1=p1,
                rank_near_target=rank_near,
                rank_max_ge_target=rank_max,
            )
        )
    return rows


def print_markdown(rows: List[Row], target_percent: float) -> None:
    print("| α | M_eff | 1位シェア p_1 | {} の順位(付近) | {}以上の最大順位 |".format(
        f"{target_percent:g}%", f"{target_percent:g}%"
    ))
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row.alpha:.1f} | {row.m_eff} | {row.p1 * 100:.2f}% | "
            f"{row.rank_near_target}位 | {row.rank_max_ge_target}位 |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Zipf alpha summary table.")
    parser.add_argument("--N", type=int, default=10000, help="Total count N (default: 10000)")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.8, 1.0, 1.2, 1.4],
        help="Alpha values (default: 0.8 1.0 1.2 1.4)",
    )
    parser.add_argument(
        "--target-percent",
        type=float,
        default=1.0,
        help="Target share percent for rank columns (default: 1.0)",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=200000,
        help="Search upper bound for M_eff (default: 200000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_p = args.target_percent / 100.0
    rows = generate_rows(n_total=args.N, alphas=args.alphas, target_p=target_p, max_rank=args.max_rank)
    print_markdown(rows=rows, target_percent=args.target_percent)


if __name__ == "__main__":
    main()

