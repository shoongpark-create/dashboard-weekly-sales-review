#!/usr/bin/env python3
"""Build a decision-ready weekly sales dashboard MVP from processed marts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DECISION_COLUMNS = [
    "sales_period_ty",
    "sales_period_ly",
    "sales_period_prev_week_ty",
    "yoy_sales_diff",
    "yoy_sales_pct",
    "wow_sales_diff",
    "wow_sales_pct",
    "qty_period_ty",
    "qty_period_ly",
    "qty_period_prev_week_ty",
    "yoy_qty_diff",
    "yoy_qty_pct",
    "wow_qty_diff",
    "wow_qty_pct",
]

PERIOD_COMPARE_CODES = ["period_ty", "period_ly", "period_prev_week_ty"]
CUMULATIVE_COMPARE_CODES = ["cum_ty", "cum_ly"]


def safe_pct(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def classify_action(yoy: float | None, wow: float | None) -> str:
    yoy = yoy or 0
    wow = wow or 0

    if yoy >= 0.12 and wow >= 0.05:
        return "투자 확대"
    if yoy <= -0.08 and wow <= -0.05:
        return "긴급 점검"
    if wow <= -0.10:
        return "단기 회복 액션"
    if yoy <= -0.08 and wow >= 0.05:
        return "반등 검증"
    return "유지/모니터링"


def decision_score(frame: pd.DataFrame) -> pd.Series:
    sales_share = frame["sales_period_ty"].div(frame["sales_period_ty"].sum())
    yoy_abs = frame["yoy_sales_pct"].fillna(0).abs()
    wow_abs = frame["wow_sales_pct"].fillna(0).abs()
    return sales_share * 70 + yoy_abs * 20 + wow_abs * 10


def enrich_decision_table(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    out["priority_score"] = decision_score(out)
    out["priority_rank"] = (
        out["priority_score"].rank(method="dense", ascending=False).astype(int)
    )
    out["recommended_action"] = out.apply(
        lambda row: classify_action(row.get("yoy_sales_pct"), row.get("wow_sales_pct")),
        axis=1,
    )

    ordered = [
        *key_columns,
        "priority_rank",
        "priority_score",
        "recommended_action",
        *DECISION_COLUMNS,
    ]
    existing = [column for column in ordered if column in out.columns]
    return out[existing].sort_values("priority_rank").reset_index(drop=True)


def format_metric(value: float | int | None, as_pct: bool = False) -> str:
    if value is None or pd.isna(value):
        return "-"
    if as_pct:
        return f"{value * 100:.1f}%"
    return f"{value:,.0f}"


def find_opportunity_and_risk(
    frame: pd.DataFrame, key_columns: list[str]
) -> tuple[str, str]:
    scored = frame.copy()
    scored["trend_score"] = scored["yoy_sales_pct"].fillna(0) + scored[
        "wow_sales_pct"
    ].fillna(0)

    target_col = " / ".join(key_columns)
    scored[target_col] = scored[key_columns].astype(str).agg(" / ".join, axis=1)

    opportunity = scored.nlargest(1, "trend_score")[target_col].iloc[0]
    risk = scored.nsmallest(1, "trend_score")[target_col].iloc[0]
    return opportunity, risk


def write_markdown_brief(
    path: Path,
    generated_at: str,
    kpi: dict,
    roadmap: pd.DataFrame,
    brand_table: pd.DataFrame,
    channel_table: pd.DataFrame,
) -> None:
    lines = [
        "# Weekly Sales Dashboard MVP Brief",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Period TY Net Sales: `{format_metric(kpi['sales_period_ty'])}`",
        f"- YoY: `{format_metric(kpi['yoy_sales_diff'])}` ({format_metric(kpi['yoy_sales_pct'], as_pct=True)})",
        f"- WoW: `{format_metric(kpi['wow_sales_diff'])}` ({format_metric(kpi['wow_sales_pct'], as_pct=True)})",
        "",
        "## Step-by-step Decision Roadmap",
        "",
    ]

    for _, row in roadmap.iterrows():
        lines.extend(
            [
                f"### Step {int(row['step'])}. {row['focus']}",
                f"- Objective: {row['objective']}",
                f"- Opportunity: {row['opportunity_target']}",
                f"- Risk: {row['risk_target']}",
                f"- Action: {row['recommended_action']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Brand Priorities",
            "",
            brand_table.head(10).to_csv(index=False),
            "",
            "## Top Channel Priorities",
            "",
            channel_table.head(10).to_csv(index=False),
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _display_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        if column.endswith("_pct"):
            out[column] = out[column].apply(
                lambda value: format_metric(value, as_pct=True)
            )
        elif pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].apply(format_metric)
    return out


def _top_rows_per_brand(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "brand" not in frame.columns:
        return frame.head(limit)

    if "sales_period_ty" in frame.columns:
        sorted_frame = frame.sort_values("sales_period_ty", ascending=False)
    elif "priority_rank" in frame.columns:
        sorted_frame = frame.sort_values("priority_rank", ascending=True)
    else:
        sorted_frame = frame.copy()

    return (
        sorted_frame.groupby("brand", dropna=False, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _cap_rows_per_brand(
    frame: pd.DataFrame,
    limit: int,
    sort_candidates: list[str] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    if limit <= 0:
        return frame.head(0).reset_index(drop=True)
    if "brand" not in frame.columns:
        return frame.head(limit).reset_index(drop=True)

    sort_candidates = sort_candidates or ["sales_period_ty", "qty_period_ty", "rank"]
    sort_columns = [col for col in sort_candidates if col in frame.columns]

    if not sort_columns:
        return (
            frame.groupby("brand", dropna=False, group_keys=False)
            .head(limit)
            .reset_index(drop=True)
        )

    out = frame.copy()
    for col in sort_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return (
        out.sort_values(
            ["brand", *sort_columns], ascending=[True, *([False] * len(sort_columns))]
        )
        .groupby("brand", dropna=False, group_keys=False)
        .head(limit)
        .reset_index(drop=True)
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator.div(denominator)
    return ratio.where(denominator.ne(0), pd.NA)


def _align_ly_year_for_keys(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "year" not in keys:
        return frame
    if "year" not in frame.columns or "timeframe_code" not in frame.columns:
        return frame

    out = frame.copy()
    ly_mask = out["timeframe_code"].isin(["period_ly", "cum_ly"])
    if not ly_mask.any():
        return out

    year_raw = out.loc[ly_mask, "year"].astype("string")
    year_num = pd.to_numeric(year_raw, errors="coerce")
    year_shifted = (year_num + 1).astype("Int64").astype("string")
    out.loc[ly_mask, "year"] = year_shifted.where(year_num.notna(), year_raw)
    return out


def _pivot_timeframe(
    frame: pd.DataFrame,
    keys: list[str],
    value_col: str,
    prefix: str,
    codes: list[str],
) -> pd.DataFrame:
    pivot = (
        frame.pivot_table(
            index=keys,
            columns="timeframe_code",
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .copy()
    )

    for code in codes:
        if code not in pivot.columns:
            pivot[code] = 0

    out = pivot[keys + codes].rename(
        columns={code: f"{prefix}_{code}" for code in codes}
    )
    return out


def _build_order_style_totals(
    fact: pd.DataFrame, reference_scope: pd.DataFrame
) -> pd.DataFrame:
    expected_cols = ["brand", "style_no", "order_qty_total", "order_amt_total"]

    order_scope = fact[fact["timeframe_code"].eq("order")].copy()
    if order_scope.empty:
        return pd.DataFrame(columns=expected_cols)

    if "style_no" in order_scope.columns:
        order_scope["style_no"] = (
            order_scope["style_no"].fillna("").astype(str).str.strip()
        )
    else:
        order_scope["style_no"] = ""

    if "basic_item_code" in order_scope.columns:
        fallback_style = (
            order_scope["basic_item_code"].fillna("").astype(str).str.strip()
        )
        missing_style = order_scope["style_no"].eq("")
        order_scope.loc[missing_style, "style_no"] = fallback_style[missing_style]

    for col in ["order_qty", "order_amt"]:
        if col in order_scope.columns:
            order_scope[col] = pd.to_numeric(order_scope[col], errors="coerce")

    valid_order_rows = order_scope[order_scope["style_no"].ne("")].copy()
    if valid_order_rows.empty:
        return pd.DataFrame(columns=expected_cols)

    agg_cols = [
        col for col in ["order_qty", "order_amt"] if col in valid_order_rows.columns
    ]
    if not agg_cols:
        return pd.DataFrame(columns=expected_cols)

    order_agg = (
        valid_order_rows.groupby(["style_no"], dropna=False)[agg_cols]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                "order_qty": "order_qty_total",
                "order_amt": "order_amt_total",
            }
        )
    )

    if "brand" in reference_scope.columns and "style_no" in reference_scope.columns:
        brand_map = (
            reference_scope[["brand", "style_no"]]
            .copy()
            .assign(
                brand=lambda x: x["brand"].fillna("").astype(str).str.strip(),
                style_no=lambda x: x["style_no"].fillna("").astype(str).str.strip(),
            )
        )
        brand_map = brand_map[
            brand_map["brand"].ne("") & brand_map["style_no"].ne("")
        ].drop_duplicates(subset=["style_no"], keep="first")
        order_agg = order_agg.merge(brand_map, on="style_no", how="left")
    else:
        order_agg["brand"] = pd.NA

    if "order_qty_total" not in order_agg.columns:
        order_agg["order_qty_total"] = pd.NA
    if "order_amt_total" not in order_agg.columns:
        order_agg["order_amt_total"] = pd.NA

    return order_agg[expected_cols]


def build_style_period_compare(
    fact: pd.DataFrame, keys: list[str] | None = None
) -> pd.DataFrame:
    keys = keys or ["brand", "style_no", "style_name", "category"]
    scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()
    cumulative_scope = fact[
        fact["timeframe_code"].isin(CUMULATIVE_COMPARE_CODES)
    ].copy()

    for frame in (scope, cumulative_scope):
        for key in ["year", "new_old"]:
            if key in frame.columns:
                frame[key] = frame[key].fillna("").astype(str).str.strip()

    scope = _align_ly_year_for_keys(scope, keys)
    cumulative_scope = _align_ly_year_for_keys(cumulative_scope, keys)

    sales = _pivot_timeframe(
        scope, keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    qty = _pivot_timeframe(scope, keys, "sales_qty", "qty", PERIOD_COMPARE_CODES)
    cost = _pivot_timeframe(
        scope, keys, "cost_amt_v_incl", "cost", PERIOD_COMPARE_CODES
    )
    tag = _pivot_timeframe(
        scope, keys, "sales_amt_tag_v_excl", "tag", PERIOD_COMPARE_CODES
    )
    cumulative_sales = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_amt_net_v_excl",
        "sales",
        CUMULATIVE_COMPARE_CODES,
    )
    cumulative_tag = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_amt_tag_v_excl",
        "tag",
        CUMULATIVE_COMPARE_CODES,
    )
    cumulative_qty = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_qty",
        "qty",
        CUMULATIVE_COMPARE_CODES,
    )

    order_scope = fact[fact["timeframe_code"].eq("order")].copy()
    if "style_no" in order_scope.columns:
        order_scope["style_no"] = (
            order_scope["style_no"].fillna("").astype(str).str.strip()
        )
    else:
        order_scope["style_no"] = ""

    if "basic_item_code" in order_scope.columns:
        fallback_style = (
            order_scope["basic_item_code"].fillna("").astype(str).str.strip()
        )
        style_missing = order_scope["style_no"].eq("")
        order_scope.loc[style_missing, "style_no"] = fallback_style[style_missing]

    for col in ["order_qty", "order_amt"]:
        if col in order_scope.columns:
            order_scope[col] = pd.to_numeric(order_scope[col], errors="coerce")

    order_agg = pd.DataFrame(
        columns=["style_no", "brand", "order_qty_total", "order_amt_total"]
    )
    valid_order_rows = order_scope[order_scope["style_no"].ne("")].copy()
    if not valid_order_rows.empty:
        agg_cols = [
            col for col in ["order_qty", "order_amt"] if col in valid_order_rows.columns
        ]
        if agg_cols:
            order_agg = (
                valid_order_rows.groupby(["style_no"], dropna=False)[agg_cols]
                .sum(min_count=1)
                .reset_index()
                .rename(
                    columns={
                        "order_qty": "order_qty_total",
                        "order_amt": "order_amt_total",
                    }
                )
            )

    if not order_agg.empty and "style_no" in scope.columns and "brand" in scope.columns:
        brand_map = (
            scope[["brand", "style_no"]]
            .copy()
            .assign(
                brand=lambda x: x["brand"].fillna("").astype(str).str.strip(),
                style_no=lambda x: x["style_no"].fillna("").astype(str).str.strip(),
            )
        )
        brand_map = brand_map[
            brand_map["brand"].ne("") & brand_map["style_no"].ne("")
        ].drop_duplicates(subset=["style_no"], keep="first")
        order_agg = order_agg.merge(brand_map, on="style_no", how="left")

    period_ty_scope = scope[scope["timeframe_code"].eq("period_ty")].copy()
    style_26_new = (
        period_ty_scope[
            period_ty_scope["year"].eq("26")
            & ~period_ty_scope["new_old"].str.contains("이월", na=False)
        ]
        .groupby(keys, dropna=False)
        .size()
        .reset_index(name="_count")
    )
    style_26_new["is_26_new"] = style_26_new["_count"] > 0
    style_26_new = style_26_new[keys + ["is_26_new"]]

    out = (
        sales.merge(qty, on=keys, how="outer")
        .merge(cost, on=keys, how="outer")
        .merge(tag, on=keys, how="outer")
        .merge(cumulative_sales, on=keys, how="left")
        .merge(cumulative_tag, on=keys, how="left")
        .merge(cumulative_qty, on=keys, how="left")
        .merge(style_26_new, on=keys, how="left")
    )

    if not order_agg.empty:
        order_join_cols = [
            col
            for col in ["brand", "style_no"]
            if col in keys and col in order_agg.columns
        ]
        if order_join_cols:
            out = out.merge(
                order_agg[order_join_cols + ["order_qty_total", "order_amt_total"]],
                on=order_join_cols,
                how="left",
            )

    for col in [
        "sales_cum_ty",
        "sales_cum_ly",
        "tag_cum_ty",
        "tag_cum_ly",
        "qty_cum_ty",
        "qty_cum_ly",
    ]:
        if col not in out.columns:
            out[col] = 0

    for col in ["order_qty_total", "order_amt_total"]:
        if col not in out.columns:
            out[col] = pd.NA
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["is_26_new"] = out["is_26_new"].astype("boolean").fillna(False).astype(bool)

    out["yoy_sales_diff"] = out["sales_period_ty"] - out["sales_period_ly"]
    out["yoy_sales_pct"] = _safe_ratio(out["yoy_sales_diff"], out["sales_period_ly"])
    out["wow_sales_diff"] = out["sales_period_ty"] - out["sales_period_prev_week_ty"]
    out["wow_sales_pct"] = _safe_ratio(
        out["wow_sales_diff"], out["sales_period_prev_week_ty"]
    )

    out["gross_margin_amt_ty"] = out["sales_period_ty"] - out["cost_period_ty"]
    out["gross_margin_rate_ty"] = _safe_ratio(
        out["gross_margin_amt_ty"], out["sales_period_ty"]
    )
    out["discount_rate_ty"] = 1 - _safe_ratio(
        out["sales_period_ty"], out["tag_period_ty"]
    )
    out["discount_rate_ty"] = out["discount_rate_ty"].where(
        out["tag_period_ty"].ne(0), pd.NA
    )
    out["season_sales_rate_ty"] = _safe_ratio(out["sales_cum_ty"], out["tag_cum_ty"])
    out["season_sales_rate_ty"] = out["season_sales_rate_ty"].where(
        out["tag_cum_ty"].ne(0), pd.NA
    )
    out["cumulative_sell_through_ty"] = _safe_ratio(
        out["qty_cum_ty"], out["order_qty_total"]
    )
    out["cumulative_sell_through_ty"] = out["cumulative_sell_through_ty"].where(
        out["order_qty_total"].gt(0), pd.NA
    )

    out = out.sort_values("sales_period_ty", ascending=False).reset_index(drop=True)
    out["rank_period_ty"] = out.index + 1
    return out


def build_season_period_compare(fact: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "year_season", "season", "season_half", "new_old"]
    period_scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()
    cumulative_scope = fact[
        fact["timeframe_code"].isin(CUMULATIVE_COMPARE_CODES)
    ].copy()

    for scope in (period_scope, cumulative_scope):
        for key in ["year_season", "season", "season_half", "new_old"]:
            if key in scope.columns:
                scope[key] = scope[key].fillna("").astype(str).str.strip()
        if "new_old" in scope.columns:
            scope.loc[scope["new_old"].eq(""), "new_old"] = "미분류"

    sales = _pivot_timeframe(
        period_scope, keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    qty = _pivot_timeframe(period_scope, keys, "sales_qty", "qty", PERIOD_COMPARE_CODES)
    tag = _pivot_timeframe(
        period_scope, keys, "sales_amt_tag_v_excl", "tag", PERIOD_COMPARE_CODES
    )
    cumulative = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_amt_net_v_excl",
        "sales",
        CUMULATIVE_COMPARE_CODES,
    )
    cumulative_qty = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_qty",
        "qty",
        CUMULATIVE_COMPARE_CODES,
    )

    order_agg = _build_order_style_totals(fact, period_scope)
    season_order_qty = pd.DataFrame(columns=[*keys, "order_qty_total"])
    if not order_agg.empty:
        cum_ty_scope = cumulative_scope[
            cumulative_scope["timeframe_code"].eq("cum_ty")
        ].copy()
        if "style_no" in cum_ty_scope.columns:
            cum_ty_scope["style_no"] = (
                cum_ty_scope["style_no"].fillna("").astype(str).str.strip()
            )
            season_style_keys = cum_ty_scope[cum_ty_scope["style_no"].ne("")][
                [*keys, "style_no"]
            ].drop_duplicates()
            if not season_style_keys.empty:
                season_style_keys = season_style_keys.merge(
                    order_agg[["brand", "style_no", "order_qty_total"]],
                    on=["brand", "style_no"],
                    how="left",
                )
                season_order_qty = (
                    season_style_keys.groupby(keys, dropna=False)["order_qty_total"]
                    .sum(min_count=1)
                    .reset_index()
                )

    out = (
        sales.merge(qty, on=keys, how="outer")
        .merge(tag, on=keys, how="outer")
        .merge(cumulative, on=keys, how="left")
        .merge(cumulative_qty, on=keys, how="left")
        .merge(season_order_qty, on=keys, how="left")
    )
    for col in ["sales_cum_ty", "sales_cum_ly", "qty_cum_ty", "qty_cum_ly"]:
        if col not in out.columns:
            out[col] = 0

    if "order_qty_total" not in out.columns:
        out["order_qty_total"] = pd.NA
    out["order_qty_total"] = pd.to_numeric(out["order_qty_total"], errors="coerce")

    out["yoy_sales_diff"] = out["sales_period_ty"] - out["sales_period_ly"]
    out["yoy_sales_pct"] = _safe_ratio(out["yoy_sales_diff"], out["sales_period_ly"])
    out["wow_sales_diff"] = out["sales_period_ty"] - out["sales_period_prev_week_ty"]
    out["wow_sales_pct"] = _safe_ratio(
        out["wow_sales_diff"], out["sales_period_prev_week_ty"]
    )
    out["discount_rate_ty"] = 1 - _safe_ratio(
        out["sales_period_ty"], out["tag_period_ty"]
    )
    out["discount_rate_ly"] = 1 - _safe_ratio(
        out["sales_period_ly"], out["tag_period_ly"]
    )
    out["discount_rate_prev_week"] = 1 - _safe_ratio(
        out["sales_period_prev_week_ty"], out["tag_period_prev_week_ty"]
    )
    out["discount_rate_wow_p"] = (
        out["discount_rate_ty"] - out["discount_rate_prev_week"]
    )
    out["discount_rate_yoy_p"] = out["discount_rate_ty"] - out["discount_rate_ly"]
    out["cumulative_sell_through_ty"] = _safe_ratio(
        out["qty_cum_ty"], out["order_qty_total"]
    )
    out["cumulative_sell_through_ty"] = out["cumulative_sell_through_ty"].where(
        out["order_qty_total"].gt(0), pd.NA
    )
    return out.sort_values("sales_period_ty", ascending=False).reset_index(drop=True)


def build_category_detail_period(fact: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "category", "item", "style_no", "style_name"]
    period_scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()
    cumulative_scope = fact[
        fact["timeframe_code"].isin(CUMULATIVE_COMPARE_CODES)
    ].copy()

    sales = _pivot_timeframe(
        period_scope, keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    qty = _pivot_timeframe(period_scope, keys, "sales_qty", "qty", PERIOD_COMPARE_CODES)
    tag = _pivot_timeframe(
        period_scope, keys, "sales_amt_tag_v_excl", "tag", PERIOD_COMPARE_CODES
    )
    cumulative = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_amt_net_v_excl",
        "sales",
        CUMULATIVE_COMPARE_CODES,
    )
    cumulative_qty = _pivot_timeframe(
        cumulative_scope,
        keys,
        "sales_qty",
        "qty",
        CUMULATIVE_COMPARE_CODES,
    )
    order_agg = _build_order_style_totals(fact, period_scope)

    out = (
        sales.merge(qty, on=keys, how="outer")
        .merge(tag, on=keys, how="outer")
        .merge(cumulative, on=keys, how="left")
        .merge(cumulative_qty, on=keys, how="left")
    )
    if not order_agg.empty:
        out = out.merge(
            order_agg[["brand", "style_no", "order_qty_total"]],
            on=["brand", "style_no"],
            how="left",
        )

    for col in ["sales_cum_ty", "sales_cum_ly", "qty_cum_ty", "qty_cum_ly"]:
        if col not in out.columns:
            out[col] = 0

    if "order_qty_total" not in out.columns:
        out["order_qty_total"] = pd.NA
    out["order_qty_total"] = pd.to_numeric(out["order_qty_total"], errors="coerce")

    out["yoy_sales_diff"] = out["sales_period_ty"] - out["sales_period_ly"]
    out["yoy_sales_pct"] = _safe_ratio(out["yoy_sales_diff"], out["sales_period_ly"])
    out["wow_sales_diff"] = out["sales_period_ty"] - out["sales_period_prev_week_ty"]
    out["wow_sales_pct"] = _safe_ratio(
        out["wow_sales_diff"], out["sales_period_prev_week_ty"]
    )
    out["discount_rate_ty"] = 1 - _safe_ratio(
        out["sales_period_ty"], out["tag_period_ty"]
    )
    out["discount_rate_ly"] = 1 - _safe_ratio(
        out["sales_period_ly"], out["tag_period_ly"]
    )
    out["discount_rate_prev_week"] = 1 - _safe_ratio(
        out["sales_period_prev_week_ty"], out["tag_period_prev_week_ty"]
    )
    out["discount_rate_wow_p"] = (
        out["discount_rate_ty"] - out["discount_rate_prev_week"]
    )
    out["discount_rate_yoy_p"] = out["discount_rate_ty"] - out["discount_rate_ly"]
    out["cumulative_sell_through_ty"] = _safe_ratio(
        out["qty_cum_ty"], out["order_qty_total"]
    )
    out["cumulative_sell_through_ty"] = out["cumulative_sell_through_ty"].where(
        out["order_qty_total"].gt(0), pd.NA
    )
    return out.sort_values("sales_period_ty", ascending=False).reset_index(drop=True)


def build_item_store_period(fact: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "category", "item", "store_name", "channel"]
    scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()

    sales = _pivot_timeframe(
        scope, keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    qty = _pivot_timeframe(scope, keys, "sales_qty", "qty", PERIOD_COMPARE_CODES)

    out = sales.merge(qty, on=keys, how="outer")
    out["yoy_sales_diff"] = out["sales_period_ty"] - out["sales_period_ly"]
    out["yoy_sales_pct"] = _safe_ratio(out["yoy_sales_diff"], out["sales_period_ly"])
    out["wow_sales_diff"] = out["sales_period_ty"] - out["sales_period_prev_week_ty"]
    out["wow_sales_pct"] = _safe_ratio(
        out["wow_sales_diff"], out["sales_period_prev_week_ty"]
    )

    return out.sort_values("sales_period_ty", ascending=False).reset_index(drop=True)


def build_item_scope_period(fact: pd.DataFrame) -> pd.DataFrame:
    keys = ["brand", "year", "season", "category", "item"]
    scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()

    for key in ["year", "season", "category", "item"]:
        if key in scope.columns:
            scope[key] = scope[key].fillna("").astype(str).str.strip()

    scope = _align_ly_year_for_keys(scope, keys)

    sales = _pivot_timeframe(
        scope, keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    qty = _pivot_timeframe(scope, keys, "sales_qty", "qty", PERIOD_COMPARE_CODES)
    out = sales.merge(qty, on=keys, how="outer")

    out["product_type"] = out.apply(
        lambda row: classify_product_type(row.get("item"), row.get("category")), axis=1
    )
    out["yoy_sales_diff"] = out["sales_period_ty"] - out["sales_period_ly"]
    out["yoy_sales_pct"] = _safe_ratio(out["yoy_sales_diff"], out["sales_period_ly"])
    out["wow_sales_diff"] = out["sales_period_ty"] - out["sales_period_prev_week_ty"]
    out["wow_sales_pct"] = _safe_ratio(
        out["wow_sales_diff"], out["sales_period_prev_week_ty"]
    )

    return out.sort_values("sales_period_ty", ascending=False).reset_index(drop=True)


def build_style_channel_store_period(
    fact: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()
    cumulative_scope = fact[
        fact["timeframe_code"].isin(CUMULATIVE_COMPARE_CODES)
    ].copy()
    order_scope = fact[fact["timeframe_code"].eq("order")].copy()

    for frame in (scope, cumulative_scope):
        if "style_no" in frame.columns:
            frame["style_no"] = frame["style_no"].fillna("").astype(str).str.strip()

    if "style_no" in order_scope.columns:
        order_scope["style_no"] = (
            order_scope["style_no"].fillna("").astype(str).str.strip()
        )
    else:
        order_scope["style_no"] = ""

    if "basic_item_code" in order_scope.columns:
        fallback_style = (
            order_scope["basic_item_code"].fillna("").astype(str).str.strip()
        )
        missing_style = order_scope["style_no"].eq("")
        order_scope.loc[missing_style, "style_no"] = fallback_style[missing_style]

    for col in ["order_qty", "order_amt"]:
        if col in order_scope.columns:
            order_scope[col] = pd.to_numeric(order_scope[col], errors="coerce")

    channel_keys = ["brand", "year", "season_half", "season", "style_no", "channel"]
    store_keys = [
        "brand",
        "year",
        "season_half",
        "season",
        "style_no",
        "store_name",
        "channel",
    ]

    scope = _align_ly_year_for_keys(scope, channel_keys)
    cumulative_scope = _align_ly_year_for_keys(cumulative_scope, channel_keys)

    style_name_map = (
        scope.loc[
            scope["style_name"].notna()
            & scope["style_name"].astype(str).str.strip().ne(""),
            ["brand", "style_no", "style_name"],
        ]
        .drop_duplicates(subset=["brand", "style_no"])
        .copy()
    )

    order_agg = pd.DataFrame(
        columns=["brand", "style_no", "order_qty_total", "order_amt_total"]
    )
    valid_order_rows = order_scope[order_scope["style_no"].ne("")].copy()
    if not valid_order_rows.empty:
        agg_cols = [
            col for col in ["order_qty", "order_amt"] if col in valid_order_rows.columns
        ]
        if agg_cols:
            order_agg = (
                valid_order_rows.groupby(["style_no"], dropna=False)[agg_cols]
                .sum(min_count=1)
                .reset_index()
                .rename(
                    columns={
                        "order_qty": "order_qty_total",
                        "order_amt": "order_amt_total",
                    }
                )
            )

    if not order_agg.empty:
        brand_map = (
            scope[["brand", "style_no"]]
            .copy()
            .assign(
                brand=lambda x: x["brand"].fillna("").astype(str).str.strip(),
                style_no=lambda x: x["style_no"].fillna("").astype(str).str.strip(),
            )
        )
        brand_map = brand_map[
            brand_map["brand"].ne("") & brand_map["style_no"].ne("")
        ].drop_duplicates(subset=["style_no"], keep="first")
        order_agg = order_agg.merge(brand_map, on="style_no", how="left")

    channel_cum_qty = _pivot_timeframe(
        cumulative_scope,
        channel_keys,
        "sales_qty",
        "qty",
        CUMULATIVE_COMPARE_CODES,
    )

    channel_sales = _pivot_timeframe(
        scope, channel_keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    channel_qty = _pivot_timeframe(
        scope, channel_keys, "sales_qty", "qty", PERIOD_COMPARE_CODES
    )
    style_channel = channel_sales.merge(
        channel_qty, on=channel_keys, how="outer"
    ).merge(channel_cum_qty, on=channel_keys, how="left")
    for col in ["qty_cum_ty", "qty_cum_ly"]:
        if col not in style_channel.columns:
            style_channel[col] = 0
    style_channel["yoy_sales_diff"] = (
        style_channel["sales_period_ty"] - style_channel["sales_period_ly"]
    )
    style_channel["yoy_sales_pct"] = _safe_ratio(
        style_channel["yoy_sales_diff"], style_channel["sales_period_ly"]
    )
    style_channel["wow_sales_diff"] = (
        style_channel["sales_period_ty"] - style_channel["sales_period_prev_week_ty"]
    )
    style_channel["wow_sales_pct"] = _safe_ratio(
        style_channel["wow_sales_diff"], style_channel["sales_period_prev_week_ty"]
    )
    style_channel = style_channel.merge(
        style_name_map, on=["brand", "style_no"], how="left"
    )
    if not order_agg.empty:
        style_channel = style_channel.merge(
            order_agg[["brand", "style_no", "order_qty_total", "order_amt_total"]],
            on=["brand", "style_no"],
            how="left",
        )
    else:
        style_channel["order_qty_total"] = pd.NA
        style_channel["order_amt_total"] = pd.NA

    style_channel["order_qty_total"] = pd.to_numeric(
        style_channel["order_qty_total"], errors="coerce"
    )
    style_channel["cumulative_sell_through_ty"] = _safe_ratio(
        style_channel["qty_cum_ty"], style_channel["order_qty_total"]
    )
    style_channel["cumulative_sell_through_ty"] = style_channel[
        "cumulative_sell_through_ty"
    ].where(style_channel["order_qty_total"].gt(0), pd.NA)

    store_sales = _pivot_timeframe(
        scope, store_keys, "sales_amt_net_v_excl", "sales", PERIOD_COMPARE_CODES
    )
    store_qty = _pivot_timeframe(
        scope, store_keys, "sales_qty", "qty", PERIOD_COMPARE_CODES
    )
    store_cum_qty = _pivot_timeframe(
        cumulative_scope,
        store_keys,
        "sales_qty",
        "qty",
        CUMULATIVE_COMPARE_CODES,
    )

    style_store = store_sales.merge(store_qty, on=store_keys, how="outer").merge(
        store_cum_qty, on=store_keys, how="left"
    )
    for col in ["qty_cum_ty", "qty_cum_ly"]:
        if col not in style_store.columns:
            style_store[col] = 0
    style_store["yoy_sales_diff"] = (
        style_store["sales_period_ty"] - style_store["sales_period_ly"]
    )
    style_store["yoy_sales_pct"] = _safe_ratio(
        style_store["yoy_sales_diff"], style_store["sales_period_ly"]
    )
    style_store["wow_sales_diff"] = (
        style_store["sales_period_ty"] - style_store["sales_period_prev_week_ty"]
    )
    style_store["wow_sales_pct"] = _safe_ratio(
        style_store["wow_sales_diff"], style_store["sales_period_prev_week_ty"]
    )
    style_store = style_store.merge(
        style_name_map, on=["brand", "style_no"], how="left"
    )
    if not order_agg.empty:
        style_store = style_store.merge(
            order_agg[["brand", "style_no", "order_qty_total", "order_amt_total"]],
            on=["brand", "style_no"],
            how="left",
        )
    else:
        style_store["order_qty_total"] = pd.NA
        style_store["order_amt_total"] = pd.NA

    style_store["order_qty_total"] = pd.to_numeric(
        style_store["order_qty_total"], errors="coerce"
    )
    style_store["cumulative_sell_through_ty"] = _safe_ratio(
        style_store["qty_cum_ty"], style_store["order_qty_total"]
    )
    style_store["cumulative_sell_through_ty"] = style_store[
        "cumulative_sell_through_ty"
    ].where(style_store["order_qty_total"].gt(0), pd.NA)

    style_channel = style_channel.sort_values(
        "sales_period_ty", ascending=False
    ).reset_index(drop=True)
    style_store = style_store.sort_values(
        "sales_period_ty", ascending=False
    ).reset_index(drop=True)
    return style_channel, style_store


def classify_product_type(item: object, category: object) -> str:
    item_text = str(item or "").strip().upper()
    category_text = str(category or "").strip()
    category_code = category_text.upper()

    goods_item_codes = {
        "BG",
        "CA",
        "GV",
        "SO",
        "SS",
        "SH",
        "SC",
        "ACC",
    }
    goods_keywords = ["용품", "잡화", "가방", "모자", "벨트", "양말", "슈즈"]

    if item_text in goods_item_codes:
        return "용품"
    if category_code in goods_item_codes:
        return "용품"
    if any(keyword in category_text for keyword in goods_keywords):
        return "용품"
    return "의류"


def build_category_mix_breakdowns(
    fact: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = fact[fact["timeframe_code"].isin(["period_ty", "period_ly"])].copy()

    segment_mix = (
        scope.groupby(["brand", "segment", "timeframe_code"], dropna=False)[
            "sales_amt_net_v_excl"
        ]
        .sum(min_count=1)
        .reset_index()
        .pivot_table(
            index=["brand", "segment"],
            columns="timeframe_code",
            values="sales_amt_net_v_excl",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    if "period_ty" not in segment_mix.columns:
        segment_mix["period_ty"] = 0
    if "period_ly" not in segment_mix.columns:
        segment_mix["period_ly"] = 0
    segment_mix = segment_mix.rename(
        columns={"period_ty": "sales_period_ty", "period_ly": "sales_period_ly"}
    )

    scope["product_type"] = scope.apply(
        lambda row: classify_product_type(row.get("item"), row.get("category")), axis=1
    )
    product_mix = (
        scope.groupby(["brand", "product_type", "timeframe_code"], dropna=False)[
            "sales_amt_net_v_excl"
        ]
        .sum(min_count=1)
        .reset_index()
        .pivot_table(
            index=["brand", "product_type"],
            columns="timeframe_code",
            values="sales_amt_net_v_excl",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    if "period_ty" not in product_mix.columns:
        product_mix["period_ty"] = 0
    if "period_ly" not in product_mix.columns:
        product_mix["period_ly"] = 0
    product_mix = product_mix.rename(
        columns={"period_ty": "sales_period_ty", "period_ly": "sales_period_ly"}
    )

    return segment_mix, product_mix


def build_store_deep_dive_records(fact: pd.DataFrame) -> list[dict]:
    key_cols = ["brand", "store_code", "store_name", "channel"]
    metrics = [
        "sales_amt_net_v_excl",
        "sales_qty",
        "sales_amt_tag_v_excl",
        "cost_amt_v_incl",
    ]

    grouped = (
        fact[
            fact["timeframe_code"].isin(
                [*PERIOD_COMPARE_CODES, *CUMULATIVE_COMPARE_CODES]
            )
        ]
        .groupby([*key_cols, "timeframe_code"], dropna=False)[metrics]
        .sum(min_count=1)
        .reset_index()
    )

    records: list[dict] = []
    for key, sub in grouped.groupby(key_cols, dropna=False):
        base = {
            "brand": key[0],
            "store_code": key[1],
            "store_name": key[2],
            "channel": key[3],
        }

        pivot = sub.set_index("timeframe_code")

        def val(code: str, metric: str) -> float:
            if code not in pivot.index:
                return 0.0
            raw = pivot.loc[code, metric]
            if isinstance(raw, pd.Series):
                return float(raw.iloc[0] or 0)
            return float(raw or 0)

        tw_sales = val("period_ty", "sales_amt_net_v_excl")
        lw_sales = val("period_prev_week_ty", "sales_amt_net_v_excl")
        ly_sales = val("period_ly", "sales_amt_net_v_excl")
        tw_qty = val("period_ty", "sales_qty")
        tw_tag = val("period_ty", "sales_amt_tag_v_excl")
        tw_cost = val("period_ty", "cost_amt_v_incl")
        ytd_sales = val("cum_ty", "sales_amt_net_v_excl")
        ytd_ly_sales = val("cum_ly", "sales_amt_net_v_excl")

        base.update(
            {
                "sales_period_ty": tw_sales,
                "sales_period_prev_week_ty": lw_sales,
                "sales_period_ly": ly_sales,
                "qty_period_ty": tw_qty,
                "sales_cum_ty": ytd_sales,
                "sales_cum_ly": ytd_ly_sales,
                "yoy_sales_pct": (tw_sales - ly_sales) / ly_sales if ly_sales else None,
                "wow_sales_pct": (tw_sales - lw_sales) / lw_sales if lw_sales else None,
                "ytd_yoy_pct": (ytd_sales - ytd_ly_sales) / ytd_ly_sales
                if ytd_ly_sales
                else None,
                "discount_rate_ty": 1 - (tw_sales / tw_tag) if tw_tag else None,
                "gross_margin_rate_ty": (tw_sales - tw_cost) / tw_sales
                if tw_sales
                else None,
            }
        )

        records.append(base)

    return records


def write_html_dashboard(
    path: Path,
    generated_at: str,
    kpi: dict,
    brand_table: pd.DataFrame,
    channel_table: pd.DataFrame,
    category_table: pd.DataFrame,
    item_table: pd.DataFrame,
    store_table: pd.DataFrame,
    extra_data: dict[str, list[dict]],
) -> None:
    cards_html = "".join(
        [
            "<article class='card panel'><h3>Period TY Net Sales</h3><p id='kpiSalesTy' class='kpi-value'>-</p><span id='kpiSalesTySub' class='kpi-sub'>-</span></article>",
            "<article class='card panel'><h3>Period TY Qty</h3><p id='kpiQtyTy' class='kpi-value'>-</p><span id='kpiQtyTySub' class='kpi-sub'>-</span></article>",
            "<article class='card panel'><h3>ASP</h3><p id='kpiAsp' class='kpi-value'>-</p><span id='kpiAspSub' class='kpi-sub'>-</span></article>",
            "<article class='card panel'><h3>YoY Growth</h3><p id='kpiYoY' class='kpi-value'>-</p><span id='kpiYoYSub' class='kpi-sub'>-</span></article>",
            "<article class='card panel'><h3>WoW Growth</h3><p id='kpiWoW' class='kpi-value'>-</p><span id='kpiWoWSub' class='kpi-sub'>-</span></article>",
        ]
    )

    dataset = {
        "brand": brand_table.to_dict(orient="records"),
        "channel": _top_rows_per_brand(channel_table, 30).to_dict(orient="records"),
        "category": _top_rows_per_brand(category_table, 20).to_dict(orient="records"),
        "item": item_table.to_dict(orient="records"),
        "store": _top_rows_per_brand(store_table, 40).to_dict(orient="records"),
        **extra_data,
    }
    dataset_json = json.dumps(dataset, ensure_ascii=False)

    template = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Weekly Sales Dashboard MVP</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js"></script>
  <style>
    :root {
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #132033;
      --muted: #5f6b7a;
      --line: #dfe5ee;
      --accent: #0d6e6e;
      --shadow: 0 10px 25px rgba(9, 30, 66, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      font-family: "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
      color: var(--text);
      background: radial-gradient(circle at 0 0, #f0f5ff, var(--bg) 45%);
    }
    h1 { margin: 0 0 6px; font-size: 28px; }
    h2 { margin: 0 0 10px; font-size: 18px; }
    .panel-head {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 8px;
      margin-bottom: 10px;
    }
    .panel-head h2 { margin: 0; }
    .unit-ref {
      margin-left: auto;
      font-size: 11px;
      color: var(--muted);
      background: #f3f7fd;
      border: 1px solid #dbe6f3;
      border-radius: 999px;
      padding: 3px 8px;
      white-space: nowrap;
    }
    p.meta { margin: 0 0 18px; color: var(--muted); }
    .panel {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      padding: 14px;
    }
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .card h3 { margin: 0; font-size: 13px; color: var(--muted); }
    .card p { margin: 8px 0 0; font-size: 24px; font-weight: 800; }
    .kpi-value { font-variant-numeric: tabular-nums; letter-spacing: 0.2px; }
    .kpi-sub { display: block; margin-top: 6px; font-size: 12px; color: var(--muted); }
    .trend-up { color: #1f8f4d; }
    .trend-down { color: #c23838; }
    .trend-flat { color: #4e5d73; }
    .control-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    .period-banner {
      margin-bottom: 16px;
      background: linear-gradient(115deg, #0d6e6e, #1b7f93);
      color: #fff;
    }
    .period-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
    }
    .period-block {
      background: rgba(255, 255, 255, 0.1);
      border: 1px solid rgba(255, 255, 255, 0.18);
      border-radius: 10px;
      padding: 10px;
    }
    .period-label { font-size: 11px; opacity: 0.85; margin-bottom: 4px; }
    .period-value { font-size: 20px; font-weight: 800; line-height: 1.1; }
    .period-sub { font-size: 12px; margin-top: 3px; opacity: 0.95; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .mix-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .mix-grid-3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .summary-item { border: 1px solid var(--line); border-radius: 10px; padding: 10px; background: #f9fbff; }
    .summary-title { font-size: 12px; color: var(--muted); margin-bottom: 4px; }
    .summary-value { font-size: 18px; font-weight: 700; }
    .channel-cards {
      display: grid;
      gap: 8px;
      max-height: 260px;
      overflow: auto;
    }
    .channel-card {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9fbff;
      padding: 10px;
      display: grid;
      gap: 3px;
    }
    .channel-card-title { font-size: 12px; color: var(--muted); }
    .channel-card-value { font-size: 17px; font-weight: 800; }
    .channel-card-sub { font-size: 12px; color: var(--muted); }
    .channel-group {
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #f9fbff;
      padding: 8px;
    }
    .channel-group-title {
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 6px;
      color: #203046;
    }
    .channel-row {
      display: grid;
      grid-template-columns: minmax(130px, 1fr) minmax(120px, 1.6fr) auto;
      gap: 8px;
      font-size: 12px;
      padding: 3px 0;
      border-bottom: 1px dashed #e4e9f1;
      align-items: center;
    }
    .channel-row:last-child { border-bottom: none; }
    .channel-row-label {
      color: #25364c;
      line-height: 1.35;
      word-break: break-word;
    }
    .channel-row-track {
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: #e7eef7;
      overflow: hidden;
    }
    .channel-row-fill {
      height: 100%;
      border-radius: 999px;
      background: linear-gradient(90deg, #5ac2b8, #0d6e6e);
    }
    .channel-row-fill.muted {
      background: linear-gradient(90deg, #b6c3d6, #8d99ae);
    }
    .channel-row-fill.up {
      background: linear-gradient(90deg, #89d4a6, #1f8f4d);
    }
    .channel-row-fill.down {
      background: linear-gradient(90deg, #f8b0b0, #c23838);
    }
    .channel-row-value {
      text-align: right;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
      min-width: 80px;
    }
    .channel-subtotal {
      margin-top: 6px;
      padding-top: 6px;
      border-top: 1px solid #d9e2ef;
      display: block;
    }
    .channel-subtotal-highlight {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
      border: 1px solid #c9d7e8;
      border-radius: 8px;
      padding: 8px 10px;
      background: linear-gradient(90deg, rgba(13, 110, 110, 0.08), rgba(13, 110, 110, 0.02));
    }
    .channel-subtotal-highlight .label {
      display: inline-flex;
      align-items: baseline;
      gap: 8px;
      color: #23405f;
      font-size: 12px;
      font-weight: 700;
    }
    .channel-subtotal-highlight .value {
      color: #0d6e6e;
      font-size: 18px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }
    .channel-subtotal-ref {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      font-size: 12px;
      color: #42566f;
    }
    .control-row .left {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .control-row .right {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    select {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fff;
      font-size: 13px;
    }
    .control-row input[type="file"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 5px 8px;
      background: #fff;
      font-size: 12px;
      max-width: 240px;
    }
    .control-row button {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fff;
      font-size: 12px;
      cursor: pointer;
    }
    .control-row button:hover {
      background: #f5f8fc;
    }
    .control-row input[type="search"] {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 10px;
      background: #fff;
      font-size: 13px;
      min-width: 180px;
    }
    .inline-filter-row {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .inline-filter-row label {
      font-size: 12px;
      color: var(--muted);
    }
    .tag {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      color: #1c2e44;
      background: #eef4fb;
      border: 1px solid #dbe6f3;
    }
    .tag.status-processing {
      color: #7a4b00;
      background: #fff7e8;
      border-color: #f8ddb4;
    }
    .tag.status-success {
      color: #166534;
      background: #ecfdf3;
      border-color: #b7ebc6;
    }
    .tag.status-error {
      color: #991b1b;
      background: #fef2f2;
      border-color: #fecaca;
    }
    .viz-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }
    .bar-list {
      display: grid;
      gap: 8px;
      margin-top: 8px;
    }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) 2fr minmax(88px, auto);
      gap: 8px;
      align-items: center;
      font-size: 12px;
    }
    .bar-track {
      width: 100%;
      height: 10px;
      background: #edf2f8;
      border-radius: 999px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: linear-gradient(90deg, #5ac2b8, #0d6e6e);
      border-radius: 999px;
    }
    .bar-value {
      text-align: right;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }
    .quad {
      position: relative;
      height: 320px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background:
        linear-gradient(to right, transparent 49.5%, #d4dde9 49.5%, #d4dde9 50.5%, transparent 50.5%),
        linear-gradient(to top, transparent 49.5%, #d4dde9 49.5%, #d4dde9 50.5%, transparent 50.5%),
        linear-gradient(180deg, #f8fbff, #f3f8f7);
      overflow: hidden;
    }
    .quad-label {
      position: absolute;
      font-size: 11px;
      color: var(--muted);
      background: rgba(255,255,255,0.85);
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid #e6edf6;
    }
    .point {
      position: absolute;
      border-radius: 999px;
      background: rgba(13, 110, 110, 0.72);
      border: 1px solid rgba(13, 110, 110, 1);
      transform: translate(-50%, -50%);
      cursor: default;
    }
    .point span {
      position: absolute;
      left: 50%;
      top: -18px;
      transform: translateX(-50%);
      white-space: nowrap;
      font-size: 11px;
      color: #123;
      background: rgba(255,255,255,0.9);
      border: 1px solid #dde6f2;
      border-radius: 999px;
      padding: 1px 6px;
    }
    .table-section { margin-top: 12px; }
    .table-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(390px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .top-style-insight-grid {
      grid-template-columns: repeat(3, minmax(0, 1fr));
      align-items: stretch;
    }
    .top-style-filter-panel {
      grid-column: 1 / -1;
      padding-top: 12px;
      padding-bottom: 12px;
    }
    .top-style-filter-panel .control-row {
      margin-bottom: 0;
    }
    .top-style-filter-panel .left {
      row-gap: 8px;
    }
    .table-wrap {
      overflow: auto;
      max-height: 360px;
      border: 1px solid var(--line);
      border-radius: 10px;
    }
    .table-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      margin-bottom: 8px;
      flex-wrap: wrap;
    }
    .table-toolbar input {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px 8px;
      min-width: 180px;
      font-size: 12px;
    }
    .table-pagination {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
    .table-pagination button {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 4px 8px;
      cursor: pointer;
    }
    .table-pagination button:disabled {
      opacity: 0.45;
      cursor: not-allowed;
    }
    th.sortable {
      cursor: pointer;
      user-select: none;
    }
    th.sortable::after {
      content: "  ↕";
      color: #94a3b8;
      font-size: 11px;
    }
    .chart-box {
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }
    .chart-box canvas {
      width: 100%;
      height: 260px;
      max-height: 260px;
    }
    .segment-legend {
      margin-top: 10px;
      display: grid;
      gap: 6px;
    }
    .segment-legend-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
      font-size: 12px;
      color: #334155;
    }
    .segment-legend-row .name {
      font-weight: 700;
      color: #1f2f46;
    }
    .segment-legend-swatches {
      display: inline-flex;
      align-items: center;
      gap: 10px;
    }
    .segment-legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      color: #475569;
      font-size: 11px;
    }
    .segment-legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      border: 1px solid rgba(15, 23, 42, 0.2);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      min-width: 620px;
    }
    table.fit-table {
      min-width: 0;
      table-layout: fixed;
    }
    table.fit-table th,
    table.fit-table td {
      font-size: 11px;
      padding: 7px 8px;
    }
    th, td {
      padding: 8px 12px;
      border-bottom: 1px solid #eef2f7;
      text-align: left;
      vertical-align: top;
      line-height: 1.35;
    }
    th {
      background: #f7fafc;
      position: sticky;
      top: 0;
      z-index: 1;
      color: #334155;
      font-weight: 700;
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: keep-all;
    }
    th.th-num,
    td.cell-num {
      text-align: right;
      white-space: nowrap;
      word-break: normal;
      overflow-wrap: normal;
      font-variant-numeric: tabular-nums;
    }
    td.cell-center {
      text-align: center;
      white-space: nowrap;
    }
    td.cell-text {
      white-space: normal;
      overflow-wrap: anywhere;
      word-break: keep-all;
    }
    .action-list {
      display: grid;
      gap: 8px;
      margin-top: 10px;
    }
    .action-pill {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 7px 9px;
      font-size: 12px;
      background: #f9fcff;
    }
    .empty-note {
      color: var(--muted);
      font-size: 12px;
      padding: 8px 0;
    }
    .roadmap table { min-width: 100%; }
    @media (max-width: 1280px) {
      .top-style-insight-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .top-style-filter-panel {
        grid-column: 1 / -1;
      }
    }
    @media (max-width: 740px) {
      body {
        padding: 12px;
        overflow-x: hidden;
      }
      table {
        min-width: 0;
      }
      .panel,
      .chart-box,
      .segment-legend,
      .channel-subtotal-highlight,
      .channel-subtotal-ref {
        width: 100%;
        max-width: 100%;
        min-width: 0;
        box-sizing: border-box;
      }
      .bar-row { grid-template-columns: 1fr; }
      .bar-value { text-align: left; }
      .table-grid { grid-template-columns: 1fr; }
      .mix-grid-3 { grid-template-columns: 1fr; }
      .top-style-filter-panel .control-row {
        flex-direction: column;
        align-items: stretch;
        gap: 10px;
      }
      .top-style-filter-panel .left {
        width: 100%;
        display: grid;
        grid-template-columns: minmax(70px, auto) minmax(0, 1fr);
        column-gap: 8px;
        row-gap: 6px;
        align-items: center;
      }
      .top-style-filter-panel .left label {
        margin: 0;
      }
      .top-style-filter-panel .left select {
        width: 100%;
        min-width: 0;
      }
      .top-style-filter-panel .control-row .tag {
        align-self: flex-start;
        max-width: 100%;
      }
      .segment-legend-row {
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 4px;
        width: 100%;
      }
      .segment-legend-swatches {
        width: 100%;
        max-width: 100%;
        display: grid;
        grid-template-columns: repeat(2, minmax(0, max-content));
        gap: 6px 10px;
        justify-content: flex-start;
      }
      .segment-legend-chip {
        min-width: 0;
        white-space: nowrap;
      }
      .channel-subtotal-highlight {
        flex-direction: column;
        align-items: flex-start;
      }
      .channel-subtotal-highlight .label {
        width: 100%;
        justify-content: space-between;
      }
      .channel-subtotal-highlight .value {
        font-size: 16px;
      }
      .channel-subtotal-ref {
        width: 100%;
        justify-content: space-between;
        gap: 6px;
      }
      .channel-subtotal-ref span {
        white-space: nowrap;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Weekly Sales Dashboard MVP</h1>
    <p class="meta">Generated at __GENERATED_AT__</p>
  </header>

  <section class="kpi-grid">__CARDS_HTML__</section>

  <section class="panel control-row">
    <div class="left">
      <label for="brandFilter"><strong>브랜드 필터</strong></label>
      <select id="brandFilter"></select>
      <label for="excelUploadInput"><strong>엑셀 업로드</strong></label>
      <input id="excelUploadInput" type="file" accept=".xlsx,.xls" />
      <button id="resetUploadBtn" type="button">기본 데이터</button>
      <span class="tag" id="scopeTag">현재 범위: 전체</span>
    </div>
    <div class="right">
      <div class="tag" id="uploadStatusTag">데이터: 기본</div>
      <div class="tag" id="recordsTag">레코드 계산 중</div>
    </div>
  </section>

  <section class="panel period-banner">
    <div class="period-grid">
      <div class="period-block">
        <div class="period-label">Period TY Sales</div>
        <div id="bannerTy" class="period-value">-</div>
        <div id="bannerTySub" class="period-sub">-</div>
      </div>
      <div class="period-block">
        <div class="period-label">Period LY Sales</div>
        <div id="bannerLy" class="period-value">-</div>
        <div id="bannerLySub" class="period-sub">-</div>
      </div>
      <div class="period-block">
        <div class="period-label">Previous Week Sales</div>
        <div id="bannerLw" class="period-value">-</div>
        <div id="bannerLwSub" class="period-sub">-</div>
      </div>
      <div class="period-block">
        <div class="period-label">Growth Snapshot</div>
        <div id="bannerGrowth" class="period-value">-</div>
        <div id="bannerGrowthSub" class="period-sub">-</div>
      </div>
    </div>
  </section>

  <section class="mix-grid-3">
    <article class="panel">
      <h2>채널 믹스 (Period TY)</h2>
      <div class="chart-box"><canvas id="chartChannelMixCompare"></canvas></div>
    </article>
    <article class="panel">
      <h2>채널 비중 (Period TY)</h2>
      <div class="chart-box"><canvas id="chartChannelMixShare"></canvas></div>
    </article>
    <article class="panel">
      <h2>채널 매출 상세</h2>
      <div id="channelSalesDetailCards" class="channel-cards"></div>
    </article>
  </section>

  <section class="mix-grid-3">
    <article class="panel">
      <h2>카테고리 믹스 (Period TY)</h2>
      <div class="chart-box"><canvas id="chartCategoryMixCompare"></canvas></div>
    </article>
    <article class="panel">
      <h2>세그먼트 믹스 (TY vs LY)</h2>
      <div class="chart-box"><canvas id="chartCombinedMixCompare"></canvas></div>
      <div id="segmentMixLegend" class="segment-legend"></div>
    </article>
    <article class="panel">
      <div class="panel-head">
        <h2>Top 20 매장 (Period TY)</h2>
        <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
      </div>
      <div id="topStoreTable" class="table-wrap"></div>
    </article>
  </section>

  <section class="panel table-section">
    <div class="panel-head">
      <h2>연도/시즌 TY-LY 매출/비중/성장율</h2>
      <span class="unit-ref">단위: 금액 백만원 | 비중 %</span>
    </div>
    <div class="tag" id="itemSalesScopeTag">전체 매출 대비 연도/시즌 비중</div>
    <div class="chart-box"><canvas id="chartItemSalesShare"></canvas></div>
  </section>

  <section class="table-grid table-section top-style-insight-grid">
    <article class="panel top-style-filter-panel">
      <div class="control-row">
        <div class="left">
          <label for="topStyleYearFilter"><strong>연도</strong></label>
          <select id="topStyleYearFilter"></select>
          <label for="topStyleSeasonFilter"><strong>시즌</strong></label>
          <select id="topStyleSeasonFilter"></select>
          <label for="topStyleCategoryFilter"><strong>카테고리</strong></label>
          <select id="topStyleCategoryFilter"></select>
          <label for="itemProductTypeFilter"><strong>상품구분</strong></label>
          <select id="itemProductTypeFilter">
            <option value="ALL">전체</option>
            <option value="의류">의류</option>
            <option value="용품">용품</option>
          </select>
        </div>
        <div class="tag" id="topStyleScopeTag">스타일 필터: 전체</div>
      </div>
    </article>
    <article class="panel">
      <div class="panel-head">
        <h2>아이템 매출 비교 (TY vs LY)</h2>
        <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
      </div>
      <div class="chart-box"><canvas id="chartItemYoYCompare"></canvas></div>
    </article>
    <article class="panel">
      <div class="panel-head">
        <h2>Best Selling Styles</h2>
        <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
      </div>
      <div id="bestStyleTable" class="table-wrap"></div>
    </article>
    <article class="panel">
      <div class="panel-head">
        <h2>Worst Selling Styles</h2>
        <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
      </div>
      <div id="worstStyleTable" class="table-wrap"></div>
    </article>
  </section>

  <section class="panel table-section">
    <div class="panel-head">
      <h2>스타일 번호 검색</h2>
      <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
    </div>
    <div class="control-row">
      <div class="left">
        <label for="styleNoYearFilter"><strong>Year</strong></label>
        <select id="styleNoYearFilter"></select>
        <label for="styleNoHalfFilter"><strong>SS/FW</strong></label>
        <select id="styleNoHalfFilter"></select>
        <label for="styleNoSeasonFilter"><strong>Season</strong></label>
        <select id="styleNoSeasonFilter"></select>
        <label for="styleNoSearch"><strong>스타일 번호</strong></label>
        <input id="styleNoSearch" type="search" placeholder="스타일 번호/스타일명 검색" />
        <select id="styleNoLookup"></select>
      </div>
      <div class="tag" id="styleNoScopeTag">스타일 번호를 선택하면 채널/매장 베스트가 표시됩니다.</div>
    </div>
    <div class="table-grid">
      <article class="panel">
        <div class="panel-head">
          <h2>잘 팔린 채널</h2>
          <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
        </div>
        <div id="styleNoTopChannelTable" class="table-wrap"></div>
      </article>
      <article class="panel">
        <div class="panel-head">
          <h2>잘 팔린 매장</h2>
          <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
        </div>
        <div id="styleNoTopStoreTable" class="table-wrap"></div>
      </article>
    </div>
  </section>

  <section class="panel table-section">
    <h2>Store Deep Dive</h2>
    <div class="control-row">
      <div class="left">
        <label for="storeLookup"><strong>매장 선택</strong></label>
        <input id="storeSearch" type="search" placeholder="매장명/채널 검색" />
        <select id="storeLookup"></select>
      </div>
      <div class="tag" id="storeScopeTag">매장을 선택하면 상세 분석이 표시됩니다.</div>
    </div>
    <div class="summary-grid">
      <article class="summary-item panel"><div class="summary-title">Weekly Sales (TW)</div><div id="storeWeeklySales" class="summary-value">-</div></article>
      <article class="summary-item panel"><div class="summary-title">WoW</div><div id="storeWeeklyGrowth" class="summary-value">-</div></article>
      <article class="summary-item panel"><div class="summary-title">YoY</div><div id="storeYoYGrowth" class="summary-value">-</div></article>
      <article class="summary-item panel"><div class="summary-title">YTD YoY</div><div id="storeYTDGrowth" class="summary-value">-</div></article>
      <article class="summary-item panel"><div class="summary-title">Avg Discount</div><div id="storeAvgDisc" class="summary-value">-</div></article>
      <article class="summary-item panel"><div class="summary-title">Gross Margin Rate</div><div id="storeMarginRate" class="summary-value">-</div></article>
    </div>
    <div class="table-grid">
      <article class="panel"><h2>Store Category Mix (Donut)</h2><div class="chart-box"><canvas id="chartStoreCategoryMix"></canvas></div></article>
      <article class="panel"><h2>Top 10 Styles (Bar)</h2><div class="chart-box"><canvas id="chartStoreTopStyles"></canvas></div></article>
      <article class="panel">
        <div class="panel-head">
          <h2>Top 10 Styles in Store</h2>
          <span class="unit-ref">단위: 금액 백만원 | 수량 EA</span>
        </div>
        <div id="storeTopStylesTable" class="table-wrap"></div>
      </article>
    </div>
  </section>

  <section class="panel table-section">
    <div class="panel-head">
      <h2>Sales by Season</h2>
      <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
    </div>
    <div class="control-row">
      <div class="left">
        <label for="seasonFilter"><strong>시즌 필터</strong></label>
        <select id="seasonFilter"></select>
      </div>
    </div>
    <div id="seasonTable" class="table-wrap"></div>
  </section>

  <section class="panel table-section">
    <h2>Profitability Analysis (손익분석)</h2>
    <div class="mix-grid-3">
      <article class="panel"><h2>Top Margin Styles (백만원)</h2><div class="chart-box"><canvas id="chartProfitTopStyles"></canvas></div></article>
      <article class="panel"><h2>Category Margin/Discount (%)</h2><div class="chart-box"><canvas id="chartProfitCategoryRate"></canvas></div></article>
      <article class="panel"><h2>Profit Map (할인율 vs 마진율)</h2><div class="chart-box"><canvas id="chartProfitScatter"></canvas></div></article>
    </div>
  </section>

  <section class="panel table-section">
    <div class="panel-head">
      <h2>Sales by Category Detail (Interactive)</h2>
      <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
    </div>
    <div class="control-row">
      <div class="left">
        <label for="categoryFilter"><strong>카테고리</strong></label>
        <select id="categoryFilter"></select>
        <label for="itemFilter"><strong>아이템</strong></label>
        <select id="itemFilter"></select>
        <label for="categoryStyleSearch"><strong>스타일 검색</strong></label>
        <input id="categoryStyleSearch" type="search" placeholder="아이템 선택 후 스타일 검색" />
        <label for="categoryStyleFilter"><strong>스타일</strong></label>
        <select id="categoryStyleFilter"></select>
      </div>
    </div>
    <div id="categoryDetailTable" class="table-wrap"></div>
    <div class="panel-head" style="margin-top: 12px;">
      <h2>판매 베스트 매장 (선택 아이템)</h2>
      <span class="unit-ref">단위: 금액 백만원 | 비율 %</span>
    </div>
    <div id="categoryItemBestStoreHint" class="empty-note">아이템을 선택하면 판매 베스트 매장이 표시됩니다.</div>
    <div id="categoryItemBestStoreTable" class="table-wrap" style="display: none;"></div>
  </section>

  <script>
    const defaultDataset = __DATASET_JSON__;
    let dataset = JSON.parse(JSON.stringify(defaultDataset));
    const numberFmt = new Intl.NumberFormat("ko-KR");

    const brandFilter = document.getElementById("brandFilter");
    const scopeTag = document.getElementById("scopeTag");
    const recordsTag = document.getElementById("recordsTag");
    const itemProductTypeFilter = document.getElementById("itemProductTypeFilter");
    const excelUploadInput = document.getElementById("excelUploadInput");
    const resetUploadBtn = document.getElementById("resetUploadBtn");
    const uploadStatusTag = document.getElementById("uploadStatusTag");

    const PERIOD_COMPARE_CODES = ["period_ty", "period_ly", "period_prev_week_ty"];
    const CUMULATIVE_COMPARE_CODES = ["cum_ty", "cum_ly"];
    const SHEET_TIMEFRAME_MAP = {
      cum_TY: "cum_ty",
      cum_LY: "cum_ly",
      period_TY: "period_ty",
      "period-1w(7d)_TY": "period_prev_week_ty",
      period_LY: "period_ly",
      order: "order",
    };
    const TIMEFRAME_GROUP_MAP = {
      cum_ty: "cumulative",
      cum_ly: "cumulative",
      period_ty: "period",
      period_prev_week_ty: "period",
      period_ly: "period",
      order: "order",
    };
    const COLUMN_RENAME_MAP = {
      "Sales Month": "sales_month",
      Brand: "brand",
      Item: "item",
      "Uni/Wms/Kids": "segment",
      Year: "year",
      Season: "season",
      "SS/FW": "season_half",
      "Year+Season": "year_season",
      "New/Old": "new_old",
      Channel: "channel",
      Store: "store_code",
      "Store Name": "store_name",
      "Style No.": "style_no",
      "Style Name": "style_name",
      "TAG Price": "tag_price",
      "Cost per pcs(+V)": "cost_per_pcs_v_incl",
      "Sales Qty": "sales_qty",
      "Sales Amt_Net": "sales_amt_net",
      "Sales Amt_TAG": "sales_amt_tag",
      "Cost Amt(V+)": "cost_amt_v_incl",
      "Channel type": "channel_type",
      Category: "category",
      "Sales Amt_TAG(V-)": "sales_amt_tag_v_excl",
      "Sales Amt_Net(V-)": "sales_amt_net_v_excl",
      basic_item_code: "style_no",
    };
    const STRING_COLUMNS = [
      "sales_month",
      "brand",
      "item",
      "segment",
      "year",
      "season",
      "season_half",
      "year_season",
      "new_old",
      "channel",
      "store_code",
      "store_name",
      "style_no",
      "style_name",
      "channel_type",
      "category",
    ];
    const NUMERIC_COLUMNS = [
      "tag_price",
      "cost_per_pcs_v_incl",
      "sales_qty",
      "sales_amt_net",
      "sales_amt_tag",
      "cost_amt_v_incl",
      "sales_amt_tag_v_excl",
      "sales_amt_net_v_excl",
      "order_qty",
      "order_amt",
    ];

    const isMissing = (value) => value === null || value === undefined || Number.isNaN(value);
    const fmtNum = (value) => isMissing(value) ? "-" : numberFmt.format(Math.round(value));
    const fmtShortAmount = (value) => {
      if (isMissing(value)) return "-";
      const abs = Math.abs(value);
      if (abs >= 100000000) return `${(value / 100000000).toFixed(1)}억`;
      if (abs >= 10000) return `${(value / 10000).toFixed(1)}만`;
      return fmtNum(value);
    };
    const fmtSignedAmount = (value) => {
      if (isMissing(value)) return "-";
      const sign = value > 0 ? "+" : "";
      return `${sign}${fmtShortAmount(value)}`;
    };
    const fmtMillion = (value) => {
      if (isMissing(value)) return "-";
      return (value / 1000000).toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    };
    const fmtPct = (value) => {
      if (isMissing(value)) return "-";
      const sign = value > 0 ? "+" : "";
      return `${sign}${(value * 100).toFixed(1)}%`;
    };
    const fmtPctPlain = (value) => {
      if (isMissing(value)) return "-";
      return `${(value * 100).toFixed(1)}%`;
    };
    const fmtPctPointPlain = (value) => {
      if (isMissing(value)) return "-";
      return `${(value * 100).toFixed(1)}%p`;
    };
    const pctWidth = (value, max) => max <= 0 ? 0 : Math.max(0, Math.min(100, (value / max) * 100));
    const safePct = (num, den) => den ? num / den : null;

    function trendClass(value) {
      if (isMissing(value)) return "trend-flat";
      if (value > 0) return "trend-up";
      if (value < 0) return "trend-down";
      return "trend-flat";
    }

    function classifyProductType(item, category) {
      const itemText = String(item || "").trim().toUpperCase();
      const categoryText = String(category || "").trim();
      const categoryCode = categoryText.toUpperCase();

      const goodsItemCodes = new Set(["BG", "CA", "GV", "SO", "SS", "SH", "SC", "ACC"]);
      const goodsKeywords = ["용품", "잡화", "가방", "모자", "벨트", "양말", "슈즈"];

      if (goodsItemCodes.has(itemText)) return "용품";
      if (goodsItemCodes.has(categoryCode)) return "용품";
      if (goodsKeywords.some((keyword) => categoryText.includes(keyword))) return "용품";
      return "의류";
    }

    function sumBy(rows, key) {
      return rows.reduce((acc, row) => acc + (Number(row[key]) || 0), 0);
    }

    function normalizeHeader(value) {
      return String(value || "").trim().replace(/\s+/g, " ");
    }

    function toSnakeCase(value) {
      return normalizeHeader(value)
        .replace(/[^a-zA-Z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "")
        .toLowerCase();
    }

    function normalizeString(value) {
      return String(value ?? "").trim();
    }

    function normalizeBrand(value) {
      return normalizeString(value).replace(/\s+/g, " ");
    }

    function cleanNumeric(value) {
      if (value === null || value === undefined) return null;
      if (typeof value === "number") {
        return Number.isFinite(value) ? value : null;
      }
      const text = String(value).trim().replace(/,/g, "");
      if (!text || text === "-" || text.toLowerCase() === "nan") {
        return null;
      }
      const parsed = Number(text);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function toFiniteNumber(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    }

    function makeKey(row, keys) {
      return keys.map((k) => normalizeString(row[k])).join("||");
    }

    function initMetricRow(row, keys) {
      const out = {};
      keys.forEach((k) => {
        out[k] = normalizeString(row[k]);
      });

      out.sales_period_ty = 0;
      out.sales_period_ly = 0;
      out.sales_period_prev_week_ty = 0;
      out.qty_period_ty = 0;
      out.qty_period_ly = 0;
      out.qty_period_prev_week_ty = 0;
      out.cost_period_ty = 0;
      out.cost_period_ly = 0;
      out.cost_period_prev_week_ty = 0;
      out.tag_period_ty = 0;
      out.tag_period_ly = 0;
      out.tag_period_prev_week_ty = 0;
      out.sales_cum_ty = 0;
      out.sales_cum_ly = 0;
      out.qty_cum_ty = 0;
      out.qty_cum_ly = 0;
      out.tag_cum_ty = 0;
      out.tag_cum_ly = 0;
      return out;
    }

    function finalizeMetricRows(rows) {
      return rows.map((row) => {
        row.yoy_sales_diff = row.sales_period_ty - row.sales_period_ly;
        row.yoy_sales_pct = safePct(row.yoy_sales_diff, row.sales_period_ly);
        row.wow_sales_diff = row.sales_period_ty - row.sales_period_prev_week_ty;
        row.wow_sales_pct = safePct(row.wow_sales_diff, row.sales_period_prev_week_ty);
        row.yoy_qty_diff = row.qty_period_ty - row.qty_period_ly;
        row.yoy_qty_pct = safePct(row.yoy_qty_diff, row.qty_period_ly);
        row.wow_qty_diff = row.qty_period_ty - row.qty_period_prev_week_ty;
        row.wow_qty_pct = safePct(row.wow_qty_diff, row.qty_period_prev_week_ty);

        row.gross_margin_amt_ty = row.sales_period_ty - row.cost_period_ty;
        row.gross_margin_rate_ty = safePct(row.gross_margin_amt_ty, row.sales_period_ty);
        row.discount_rate_ty = row.tag_period_ty ? 1 - row.sales_period_ty / row.tag_period_ty : null;
        row.discount_rate_ly = row.tag_period_ly ? 1 - row.sales_period_ly / row.tag_period_ly : null;
        row.discount_rate_prev_week = row.tag_period_prev_week_ty
          ? 1 - row.sales_period_prev_week_ty / row.tag_period_prev_week_ty
          : null;
        row.discount_rate_wow_p =
          row.discount_rate_ty == null || row.discount_rate_prev_week == null
            ? null
            : row.discount_rate_ty - row.discount_rate_prev_week;
        row.discount_rate_yoy_p =
          row.discount_rate_ty == null || row.discount_rate_ly == null
            ? null
            : row.discount_rate_ty - row.discount_rate_ly;
        row.season_sales_rate_ty = safePct(row.sales_cum_ty, row.tag_cum_ty);
        return row;
      });
    }

    function aggregateCompareRows(factRows, keys) {
      const map = new Map();

      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        const isPeriod = PERIOD_COMPARE_CODES.includes(code);
        const isCum = CUMULATIVE_COMPARE_CODES.includes(code);
        if (!isPeriod && !isCum) return;

        const rowForKey = { ...row };
        if (keys.includes("year") && ["period_ly", "cum_ly"].includes(code)) {
          const parsedYear = Number(normalizeString(row.year));
          if (Number.isFinite(parsedYear)) {
            rowForKey.year = String(Math.trunc(parsedYear) + 1);
          }
        }

        const rowKey = makeKey(rowForKey, keys);
        if (!map.has(rowKey)) {
          map.set(rowKey, initMetricRow(rowForKey, keys));
        }
        const target = map.get(rowKey);

        const sales = toFiniteNumber(row.sales_amt_net_v_excl);
        const qty = toFiniteNumber(row.sales_qty);
        const tag = toFiniteNumber(row.sales_amt_tag_v_excl);
        const cost = toFiniteNumber(row.cost_amt_v_incl);

        if (code === "period_ty") {
          target.sales_period_ty += sales;
          target.qty_period_ty += qty;
          target.tag_period_ty += tag;
          target.cost_period_ty += cost;
        } else if (code === "period_ly") {
          target.sales_period_ly += sales;
          target.qty_period_ly += qty;
          target.tag_period_ly += tag;
          target.cost_period_ly += cost;
        } else if (code === "period_prev_week_ty") {
          target.sales_period_prev_week_ty += sales;
          target.qty_period_prev_week_ty += qty;
          target.tag_period_prev_week_ty += tag;
          target.cost_period_prev_week_ty += cost;
        } else if (code === "cum_ty") {
          target.sales_cum_ty += sales;
          target.qty_cum_ty += qty;
          target.tag_cum_ty += tag;
        } else if (code === "cum_ly") {
          target.sales_cum_ly += sales;
          target.qty_cum_ly += qty;
          target.tag_cum_ly += tag;
        }
      });

      return finalizeMetricRows(Array.from(map.values())).sort(
        (a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0),
      );
    }

    function buildStyleNameMap(factRows) {
      const map = new Map();
      factRows.forEach((row) => {
        const brand = normalizeString(row.brand);
        const styleNo = normalizeString(row.style_no);
        const styleName = normalizeString(row.style_name);
        if (!brand || !styleNo || !styleName) return;
        const key = `${brand}||${styleNo}`;
        if (!map.has(key)) {
          map.set(key, styleName);
        }
      });
      return map;
    }

    function buildOrderByBrandStyle(factRows) {
      const brandByStyle = new Map();
      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        if (code === "order") return;
        const brand = normalizeString(row.brand);
        const styleNo = normalizeString(row.style_no || row.basic_item_code);
        if (!brand || !styleNo || brandByStyle.has(styleNo)) return;
        brandByStyle.set(styleNo, brand);
      });

      const orderMap = new Map();
      factRows.forEach((row) => {
        if (normalizeString(row.timeframe_code) !== "order") return;
        const styleNo = normalizeString(row.style_no || row.basic_item_code);
        if (!styleNo) return;
        const brand = brandByStyle.get(styleNo) || "";
        const key = `${brand}||${styleNo}`;
        if (!orderMap.has(key)) {
          orderMap.set(key, { order_qty_total: 0, order_amt_total: 0 });
        }
        const target = orderMap.get(key);
        target.order_qty_total += toFiniteNumber(row.order_qty);
        target.order_amt_total += toFiniteNumber(row.order_amt);
      });

      return orderMap;
    }

    function attachOrderMetrics(rows, orderByBrandStyle) {
      rows.forEach((row) => {
        const key = `${normalizeString(row.brand)}||${normalizeString(row.style_no)}`;
        const order = orderByBrandStyle.get(key);
        row.order_qty_total = order ? order.order_qty_total : null;
        row.order_amt_total = order ? order.order_amt_total : null;
        row.cumulative_sell_through_ty = row.order_qty_total
          ? safePct(toFiniteNumber(row.qty_cum_ty), row.order_qty_total)
          : null;
      });
      return rows;
    }

    function buildStyle26NewSet(factRows) {
      const set = new Set();
      factRows.forEach((row) => {
        if (normalizeString(row.timeframe_code) !== "period_ty") return;
        if (normalizeString(row.year) !== "26") return;
        if (normalizeString(row.new_old).includes("이월")) return;
        const brand = normalizeString(row.brand);
        const styleNo = normalizeString(row.style_no);
        if (!brand || !styleNo) return;
        set.add(`${brand}||${styleNo}`);
      });
      return set;
    }

    function buildSeasonOrderQtyMap(factRows, orderByBrandStyle) {
      const styleSets = new Map();
      factRows.forEach((row) => {
        if (normalizeString(row.timeframe_code) !== "cum_ty") return;
        const styleNo = normalizeString(row.style_no);
        const brand = normalizeString(row.brand);
        if (!styleNo || !brand) return;
        const seasonKey = makeKey(row, ["brand", "year_season", "season", "season_half", "new_old"]);
        if (!styleSets.has(seasonKey)) {
          styleSets.set(seasonKey, new Set());
        }
        styleSets.get(seasonKey).add(`${brand}||${styleNo}`);
      });

      const orderBySeason = new Map();
      styleSets.forEach((styleSet, seasonKey) => {
        let total = 0;
        styleSet.forEach((styleKey) => {
          const order = orderByBrandStyle.get(styleKey);
          if (order) total += toFiniteNumber(order.order_qty_total);
        });
        orderBySeason.set(seasonKey, total > 0 ? total : null);
      });
      return orderBySeason;
    }

    function buildStoreDeepDiveRows(factRows) {
      const keys = ["brand", "store_code", "store_name", "channel"];
      const map = new Map();

      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        if (![...PERIOD_COMPARE_CODES, ...CUMULATIVE_COMPARE_CODES].includes(code)) return;
        const rowKey = makeKey(row, keys);
        if (!map.has(rowKey)) {
          const base = {
            brand: normalizeString(row.brand),
            store_code: normalizeString(row.store_code),
            store_name: normalizeString(row.store_name),
            channel: normalizeString(row.channel),
            sales_period_ty: 0,
            sales_period_prev_week_ty: 0,
            sales_period_ly: 0,
            qty_period_ty: 0,
            sales_cum_ty: 0,
            sales_cum_ly: 0,
            tag_period_ty: 0,
            cost_period_ty: 0,
          };
          map.set(rowKey, base);
        }
        const target = map.get(rowKey);
        const sales = toFiniteNumber(row.sales_amt_net_v_excl);
        const qty = toFiniteNumber(row.sales_qty);
        const tag = toFiniteNumber(row.sales_amt_tag_v_excl);
        const cost = toFiniteNumber(row.cost_amt_v_incl);

        if (code === "period_ty") {
          target.sales_period_ty += sales;
          target.qty_period_ty += qty;
          target.tag_period_ty += tag;
          target.cost_period_ty += cost;
        } else if (code === "period_prev_week_ty") {
          target.sales_period_prev_week_ty += sales;
        } else if (code === "period_ly") {
          target.sales_period_ly += sales;
        } else if (code === "cum_ty") {
          target.sales_cum_ty += sales;
        } else if (code === "cum_ly") {
          target.sales_cum_ly += sales;
        }
      });

      return Array.from(map.values())
        .map((row) => {
          row.yoy_sales_pct = safePct(
            row.sales_period_ty - row.sales_period_ly,
            row.sales_period_ly,
          );
          row.wow_sales_pct = safePct(
            row.sales_period_ty - row.sales_period_prev_week_ty,
            row.sales_period_prev_week_ty,
          );
          row.ytd_yoy_pct = safePct(
            row.sales_cum_ty - row.sales_cum_ly,
            row.sales_cum_ly,
          );
          row.discount_rate_ty = row.tag_period_ty
            ? 1 - row.sales_period_ty / row.tag_period_ty
            : null;
          row.gross_margin_rate_ty = row.sales_period_ty
            ? (row.sales_period_ty - row.cost_period_ty) / row.sales_period_ty
            : null;
          return row;
        })
        .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0));
    }

    function buildStoreStyleRows(factRows) {
      const map = new Map();
      factRows.forEach((row) => {
        if (normalizeString(row.timeframe_code) !== "period_ty") return;
        const keys = ["brand", "store_name", "style_no", "style_name", "category"];
        const rowKey = makeKey(row, keys);
        if (!map.has(rowKey)) {
          map.set(rowKey, {
            brand: normalizeString(row.brand),
            store_name: normalizeString(row.store_name),
            style_no: normalizeString(row.style_no),
            style_name: normalizeString(row.style_name),
            category: normalizeString(row.category),
            sales_period_ty: 0,
            sales_qty_period_ty: 0,
          });
        }
        const target = map.get(rowKey);
        target.sales_period_ty += toFiniteNumber(row.sales_amt_net_v_excl);
        target.sales_qty_period_ty += toFiniteNumber(row.sales_qty);
      });
      return Array.from(map.values()).sort(
        (a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0),
      );
    }

    function buildStoreCategoryMixRows(factRows) {
      const map = new Map();
      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        if (!["period_ty", "period_ly"].includes(code)) return;
        const key = makeKey(row, ["brand", "store_name", "category"]);
        if (!map.has(key)) {
          map.set(key, {
            brand: normalizeString(row.brand),
            store_name: normalizeString(row.store_name),
            category: normalizeString(row.category),
            sales_period_ty: 0,
            sales_period_ly: 0,
          });
        }
        const target = map.get(key);
        if (code === "period_ty") {
          target.sales_period_ty += toFiniteNumber(row.sales_amt_net_v_excl);
        } else {
          target.sales_period_ly += toFiniteNumber(row.sales_amt_net_v_excl);
        }
      });
      return Array.from(map.values());
    }

    function buildSegmentMixRows(factRows) {
      const map = new Map();
      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        if (!["period_ty", "period_ly"].includes(code)) return;
        const key = makeKey(row, ["brand", "segment"]);
        if (!map.has(key)) {
          map.set(key, {
            brand: normalizeString(row.brand),
            segment: normalizeString(row.segment),
            sales_period_ty: 0,
            sales_period_ly: 0,
          });
        }
        const target = map.get(key);
        if (code === "period_ty") {
          target.sales_period_ty += toFiniteNumber(row.sales_amt_net_v_excl);
        } else {
          target.sales_period_ly += toFiniteNumber(row.sales_amt_net_v_excl);
        }
      });
      return Array.from(map.values());
    }

    function buildProductMixRows(factRows) {
      const map = new Map();
      factRows.forEach((row) => {
        const code = normalizeString(row.timeframe_code);
        if (!["period_ty", "period_ly"].includes(code)) return;
        const productType = classifyProductType(row.item, row.category);
        const key = `${normalizeString(row.brand)}||${productType}`;
        if (!map.has(key)) {
          map.set(key, {
            brand: normalizeString(row.brand),
            product_type: productType,
            sales_period_ty: 0,
            sales_period_ly: 0,
          });
        }
        const target = map.get(key);
        if (code === "period_ty") {
          target.sales_period_ty += toFiniteNumber(row.sales_amt_net_v_excl);
        } else {
          target.sales_period_ly += toFiniteNumber(row.sales_amt_net_v_excl);
        }
      });
      return Array.from(map.values());
    }

    function buildItemScopeRows(factRows) {
      const rows = aggregateCompareRows(
        factRows,
        ["brand", "year", "season", "category", "item"],
      );
      return rows.map((row) => ({
        ...row,
        product_type: classifyProductType(row.item, row.category),
      }));
    }

    function buildProfitabilityStyleRows(styleRows) {
      return [...styleRows]
        .sort((a, b) => (Number(b.gross_margin_amt_ty) || 0) - (Number(a.gross_margin_amt_ty) || 0))
        .map((row, index) => ({
          brand: normalizeString(row.brand),
          style_no: normalizeString(row.style_no),
          style_name: normalizeString(row.style_name),
          category: normalizeString(row.category),
          sales_period_ty: Number(row.sales_period_ty) || 0,
          gross_margin_amt_ty: Number(row.gross_margin_amt_ty) || 0,
          gross_margin_rate_ty: row.gross_margin_rate_ty ?? null,
          discount_rate_ty: row.discount_rate_ty ?? null,
          rank: index + 1,
        }));
    }

    function buildProfitabilityCategoryRows(categoryRows) {
      return [...categoryRows].map((row) => ({
        brand: normalizeString(row.brand),
        category: normalizeString(row.category),
        sales_period_ty: Number(row.sales_period_ty) || 0,
        cost_amt_v_incl: Number(row.cost_period_ty) || 0,
        sales_amt_tag_v_excl: Number(row.tag_period_ty) || 0,
        gross_margin_amt_ty: Number(row.gross_margin_amt_ty) || 0,
        gross_margin_rate_ty: row.gross_margin_rate_ty ?? null,
        discount_rate_ty: row.discount_rate_ty ?? null,
      }));
    }

    function buildDatasetFromFactRows(factRows) {
      const styleNameMap = buildStyleNameMap(factRows);
      const orderByBrandStyle = buildOrderByBrandStyle(factRows);
      const style26NewSet = buildStyle26NewSet(factRows);
      const styleProductTypeMap = new Map();
      factRows.forEach((row) => {
        const styleNo = normalizeString(row.style_no);
        if (!styleNo) return;
        const key = makeKey(row, ["brand", "style_no", "category"]);
        if (!styleProductTypeMap.has(key)) {
          styleProductTypeMap.set(key, classifyProductType(row.item, row.category));
        }
      });

      const resolveStyleProductType = (row) => {
        const key = makeKey(row, ["brand", "style_no", "category"]);
        return styleProductTypeMap.get(key) || classifyProductType(row.item, row.category);
      };

      const brandRows = aggregateCompareRows(factRows, ["brand"]);
      const channelRows = aggregateCompareRows(factRows, ["brand", "channel", "channel_type"]);
      const categoryRows = aggregateCompareRows(factRows, ["brand", "category"]);
      const itemRows = aggregateCompareRows(factRows, ["brand", "item", "category"]);
      const itemScopeRows = buildItemScopeRows(factRows);
      const storeRows = aggregateCompareRows(factRows, ["brand", "store_code", "store_name", "channel"]);

      const styleRows = attachOrderMetrics(
        aggregateCompareRows(factRows, ["brand", "style_no", "style_name", "category"]),
        orderByBrandStyle,
      )
        .map((row, index) => ({
          ...row,
          product_type: resolveStyleProductType(row),
          is_26_new: style26NewSet.has(`${normalizeString(row.brand)}||${normalizeString(row.style_no)}`),
          rank_period_ty: index + 1,
        }))
        .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0));

      const styleScopeRows = attachOrderMetrics(
        aggregateCompareRows(factRows, ["brand", "year", "season", "style_no", "style_name", "category"]),
        orderByBrandStyle,
      )
        .map((row, index) => ({
          ...row,
          product_type: resolveStyleProductType(row),
          is_26_new: style26NewSet.has(`${normalizeString(row.brand)}||${normalizeString(row.style_no)}`),
          rank_period_ty: index + 1,
        }))
        .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0));

      const seasonRows = aggregateCompareRows(
        factRows,
        ["brand", "year_season", "season", "season_half", "new_old"],
      );
      const seasonOrderQty = buildSeasonOrderQtyMap(factRows, orderByBrandStyle);
      seasonRows.forEach((row) => {
        const key = makeKey(row, ["brand", "year_season", "season", "season_half", "new_old"]);
        row.order_qty_total = seasonOrderQty.get(key) ?? null;
        row.cumulative_sell_through_ty = row.order_qty_total
          ? safePct(toFiniteNumber(row.qty_cum_ty), row.order_qty_total)
          : null;
      });

      const categoryDetailRows = attachOrderMetrics(
        aggregateCompareRows(factRows, ["brand", "category", "item", "style_no", "style_name"]),
        orderByBrandStyle,
      );

      const itemStoreRows = aggregateCompareRows(
        factRows,
        ["brand", "category", "item", "store_name", "channel"],
      );

      const styleChannelRows = attachOrderMetrics(
        aggregateCompareRows(factRows, ["brand", "year", "season_half", "season", "style_no", "channel"]),
        orderByBrandStyle,
      ).map((row) => {
        const styleKey = `${normalizeString(row.brand)}||${normalizeString(row.style_no)}`;
        return {
          ...row,
          style_name: styleNameMap.get(styleKey) || "",
        };
      });

      const styleStoreRows = attachOrderMetrics(
        aggregateCompareRows(
          factRows,
          ["brand", "year", "season_half", "season", "style_no", "store_name", "channel"],
        ),
        orderByBrandStyle,
      ).map((row) => {
        const styleKey = `${normalizeString(row.brand)}||${normalizeString(row.style_no)}`;
        return {
          ...row,
          style_name: styleNameMap.get(styleKey) || "",
        };
      });

      const profitabilityStyleRows = buildProfitabilityStyleRows(styleRows);
      const profitabilityCategoryRows = buildProfitabilityCategoryRows(categoryRows);
      const profitabilityStyleSeasonRows = aggregateCompareRows(
        factRows,
        ["brand", "year_season", "style_no", "style_name", "category"],
      ).map((row) => ({
        brand: normalizeString(row.brand),
        year_season: normalizeString(row.year_season),
        style_no: normalizeString(row.style_no),
        style_name: normalizeString(row.style_name),
        category: normalizeString(row.category),
        sales_period_ty: Number(row.sales_period_ty) || 0,
        cost_amt_v_incl: Number(row.cost_period_ty) || 0,
        sales_amt_tag_v_excl: Number(row.tag_period_ty) || 0,
        gross_margin_amt_ty: Number(row.gross_margin_amt_ty) || 0,
        gross_margin_rate_ty: row.gross_margin_rate_ty ?? null,
        discount_rate_ty: row.discount_rate_ty ?? null,
      }));

      const profitabilityCategorySeasonRows = aggregateCompareRows(
        factRows,
        ["brand", "year_season", "category"],
      ).map((row) => ({
        brand: normalizeString(row.brand),
        year_season: normalizeString(row.year_season),
        category: normalizeString(row.category),
        sales_period_ty: Number(row.sales_period_ty) || 0,
        cost_amt_v_incl: Number(row.cost_period_ty) || 0,
        sales_amt_tag_v_excl: Number(row.tag_period_ty) || 0,
        gross_margin_amt_ty: Number(row.gross_margin_amt_ty) || 0,
        gross_margin_rate_ty: row.gross_margin_rate_ty ?? null,
        discount_rate_ty: row.discount_rate_ty ?? null,
      }));

      return {
        brand: brandRows,
        channel: channelRows,
        category: categoryRows,
        item: itemRows,
        item_scope: itemScopeRows,
        store: storeRows,
        style: styleRows,
        style_scope: styleScopeRows,
        season: seasonRows,
        category_detail: categoryDetailRows,
        item_store_period: itemStoreRows,
        style_channel_period: styleChannelRows,
        style_store_period: styleStoreRows,
        store_deep_dive: buildStoreDeepDiveRows(factRows),
        store_style: buildStoreStyleRows(factRows),
        store_category_mix: buildStoreCategoryMixRows(factRows),
        segment_mix: buildSegmentMixRows(factRows),
        product_mix: buildProductMixRows(factRows),
        profitability_style: profitabilityStyleRows,
        profitability_category: profitabilityCategoryRows,
        profitability_style_season: profitabilityStyleSeasonRows,
        profitability_category_season: profitabilityCategorySeasonRows,
      };
    }

    async function parseWorkbookToFactRows(file) {
      if (typeof XLSX === "undefined") {
        throw new Error("XLSX parser is not loaded.");
      }

      const buffer = await file.arrayBuffer();
      const workbook = XLSX.read(buffer, { type: "array" });
      const parsedRows = [];

      workbook.SheetNames.forEach((sheetName) => {
        const timeframeCode = SHEET_TIMEFRAME_MAP[sheetName];
        if (!timeframeCode) return;

        const worksheet = workbook.Sheets[sheetName];
        const rows = XLSX.utils.sheet_to_json(worksheet, { defval: null, raw: false });

        rows.forEach((rawRow) => {
          const row = {};
          Object.entries(rawRow).forEach(([header, value]) => {
            const normalized = normalizeHeader(header);
            const renamed = COLUMN_RENAME_MAP[normalized] || toSnakeCase(normalized);
            row[renamed] = value;
          });

          if (!row.style_no && row.basic_item_code) {
            row.style_no = row.basic_item_code;
          }

          STRING_COLUMNS.forEach((col) => {
            row[col] = normalizeString(row[col]);
          });
          NUMERIC_COLUMNS.forEach((col) => {
            row[col] = cleanNumeric(row[col]);
          });

          row.source_sheet = sheetName;
          row.timeframe_code = timeframeCode;
          row.timeframe_group = TIMEFRAME_GROUP_MAP[timeframeCode] || "unknown";
          parsedRows.push(row);
        });
      });

      return parsedRows;
    }

    function setUploadStatus(type, message) {
      if (!uploadStatusTag) return;
      uploadStatusTag.classList.remove("status-processing", "status-success", "status-error");
      if (type === "processing") {
        uploadStatusTag.classList.add("status-processing");
      } else if (type === "success") {
        uploadStatusTag.classList.add("status-success");
      } else if (type === "error") {
        uploadStatusTag.classList.add("status-error");
      }
      uploadStatusTag.textContent = message;
    }

    function applyUploadedDataset(nextDataset, sourceLabel, statusType = "default") {
      const previousBrand = brandFilter.value || "ALL";
      dataset = nextDataset;
      setupInteractiveFilters();
      setupBrandFilter();

      const hasPrevBrand = Array.from(brandFilter.options).some(
        (opt) => opt.value === previousBrand,
      );
      brandFilter.value = hasPrevBrand ? previousBrand : "ALL";
      updateDashboard();

      if (statusType === "success") {
        setUploadStatus("success", `데이터: 업로드 성공 (${sourceLabel})`);
      } else {
        setUploadStatus("default", `데이터: ${sourceLabel}`);
      }
    }

    async function handleExcelUploadChange(event) {
      const file = event.target.files && event.target.files[0];
      if (!file) return;

      try {
        setUploadStatus("processing", `분석 중: ${file.name}`);
        const factRows = await parseWorkbookToFactRows(file);
        if (!factRows.length) {
          throw new Error("유효한 시트 데이터가 없습니다.");
        }

        const rebuiltDataset = buildDatasetFromFactRows(factRows);
        applyUploadedDataset(rebuiltDataset, file.name, "success");
      } catch (error) {
        console.error(error);
        setUploadStatus("error", "데이터: 업로드 실패");
        alert("엑셀 분석 중 오류가 발생했습니다. 파일 형식과 시트명을 확인해 주세요.");
      }
    }

    function resetToDefaultDataset() {
      if (excelUploadInput) {
        excelUploadInput.value = "";
      }
      applyUploadedDataset(JSON.parse(JSON.stringify(defaultDataset)), "기본");
    }

    const chartRegistry = {};

    function upsertChart(canvasId, config) {
      const canvas = document.getElementById(canvasId);
      if (!canvas || typeof Chart === "undefined") return;
      if (chartRegistry[canvasId]) {
        chartRegistry[canvasId].destroy();
      }
      chartRegistry[canvasId] = new Chart(canvas, config);
    }

    function setupBrandFilter() {
      const brands = Array.from(
        new Set(dataset.brand.map((row) => normalizeBrand(row.brand)).filter(Boolean)),
      ).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      brandFilter.innerHTML = "";

      const allOpt = document.createElement("option");
      allOpt.value = "ALL";
      allOpt.textContent = "전체";
      brandFilter.appendChild(allOpt);

      brands.forEach((brand) => {
        const opt = document.createElement("option");
        opt.value = brand;
        opt.textContent = brand;
        brandFilter.appendChild(opt);
      });
    }

    function filteredRows(rows) {
      const selected = normalizeBrand(brandFilter.value);
      if (!selected || selected === "ALL") {
        return rows;
      }
      return rows.filter((row) => normalizeBrand(row.brand) === selected);
    }

    function normalizeDimensionLabel(dimensionKey, value) {
      const text = normalizeString(value);
      if (!text || text === "-") return "Unknown";
      if (dimensionKey === "category") {
        const upper = text.toUpperCase();
        const categoryMap = {
          TOP: "TOP",
          BOTTOM: "BOTTOM",
          OUTER: "OUTER",
          ACC: "ACC",
          ACCESSORY: "ACC",
          ACCESSORIES: "ACC",
          GOODS: "ACC",
        };
        return categoryMap[upper] || upper;
      }
      return text;
    }

    function renderMixBars(containerId, rows, dimensionKey, limit = 12) {
      const container = document.getElementById(containerId);
      container.innerHTML = "";

      const grouped = {};
      rows.forEach((row) => {
        const key = normalizeDimensionLabel(dimensionKey, row[dimensionKey]);
        if (!grouped[key]) {
          grouped[key] = { ty: 0, ly: 0 };
        }
        grouped[key].ty += Number(row.sales_period_ty) || 0;
        grouped[key].ly += Number(row.sales_period_ly) || 0;
      });

      let sorted = Object.entries(grouped)
        .map(([label, value]) => ({
          label,
          ty: value.ty,
          ly: value.ly,
          yoyPct: value.ly ? (value.ty - value.ly) / value.ly : null,
        }))
        .sort((a, b) => b.ty - a.ty)
        .slice(0, limit);

      if (dimensionKey === "category") {
        const knownCategories = new Set(["TOP", "BOTTOM", "OUTER", "ACC"]);
        const hasKnownWithValue = sorted.some(
          (row) => knownCategories.has(row.label) && ((Number(row.ty) || 0) !== 0 || (Number(row.ly) || 0) !== 0),
        );
        if (hasKnownWithValue) {
          sorted = sorted.filter((row) => row.label !== "Unknown");
        }
      }

      if (!sorted.length) {
        container.innerHTML = "<div class='empty-note'>데이터 없음</div>";
        return;
      }

      const max = sorted.reduce((acc, row) => Math.max(acc, row.ty || 0), 0);
      const total = sorted.reduce((acc, row) => acc + (row.ty || 0), 0);

      sorted.forEach((row) => {
        const rowEl = document.createElement("div");
        rowEl.className = "bar-row";

        const labelEl = document.createElement("div");
        labelEl.textContent = row.label;

        const track = document.createElement("div");
        track.className = "bar-track";
        const fill = document.createElement("div");
        fill.className = "bar-fill";
        fill.style.width = `${pctWidth(row.ty || 0, max)}%`;
        track.appendChild(fill);

        const valueEl = document.createElement("div");
        valueEl.className = "bar-value";
        const share = total ? ((row.ty / total) * 100).toFixed(1) : "0.0";
        valueEl.textContent = `TY ${fmtShortAmount(row.ty)} | LY ${fmtShortAmount(row.ly)} | YoY ${fmtPct(row.yoyPct)} | ${share}%`;

        rowEl.appendChild(labelEl);
        rowEl.appendChild(track);
        rowEl.appendChild(valueEl);
        container.appendChild(rowEl);
      });
    }

    function channelBucket(channel) {
      const value = String(channel || "").trim();
      const offline = new Set(["백화점", "쇼핑몰", "대리점", "위탁사", "아울렛", "면세점", "직영점"]);
      const online = new Set(["자사몰", "무신사", "위탁몰", "외부몰"]);

      if (["기타", "홀세일", "해외몰"].includes(value) || value.includes("해외")) return "기타";
      if (offline.has(value)) return "오프라인";
      if (online.has(value)) return "온라인";
      return "기타";
    }

    function renderChannelMixCharts(channelRows) {
      const groups = {
        "오프라인": { ty: 0, ly: 0, channels: [] },
        "온라인": { ty: 0, ly: 0, channels: [] },
        "기타": { ty: 0, ly: 0, channels: [] },
      };

      channelRows.forEach((row) => {
        const bucket = channelBucket(row.channel);
        groups[bucket].ty += Number(row.sales_period_ty) || 0;
        groups[bucket].ly += Number(row.sales_period_ly) || 0;
        groups[bucket].channels.push({
          brand: row.brand || "-",
          channel: row.channel || "Unknown",
          ty: Number(row.sales_period_ty) || 0,
          ly: Number(row.sales_period_ly) || 0,
        });
      });

      const labels = ["오프라인", "온라인", "기타"];
      const tyValues = labels.map((k) => groups[k].ty);
      const lyValues = labels.map((k) => groups[k].ly);
      const yoyValues = labels.map((k) => groups[k].ly ? (groups[k].ty - groups[k].ly) / groups[k].ly : null);

      const cardBox = document.getElementById("channelSalesDetailCards");
      if (cardBox) {
        cardBox.innerHTML = "";
        labels.forEach((label) => {
          const bucket = groups[label];
          const ty = bucket.ty;
          const ly = bucket.ly;
          const yoy = ly ? (ty - ly) / ly : null;
          const channels = [...bucket.channels].sort((a, b) => b.ty - a.ty);
          const maxTy = channels.reduce((acc, entry) => Math.max(acc, entry.ty), 0);

          const rowsHtml = channels
            .map(
              (entry) => {
                const width = maxTy > 0 ? pctWidth(entry.ty, maxTy) : 0;
                return `<div class="channel-row"><span class="channel-row-label">${entry.brand} | ${entry.channel}</span><div class="channel-row-track"><div class="channel-row-fill" style="width:${width}%"></div></div><strong class="channel-row-value">${fmtShortAmount(entry.ty)}</strong></div>`;
              },
            )
            .join("");

          const yoyClass = trendClass(yoy);

          const card = document.createElement("article");
          card.className = "channel-group";
          card.innerHTML = `
            <div class="channel-group-title">${label}</div>
            ${rowsHtml || "<div class='empty-note'>세부 채널 없음</div>"}
            <div class="channel-subtotal">
              <div class="channel-subtotal-highlight">
                <span class="label">합계 TY <strong class="value">${fmtShortAmount(ty)}</strong></span>
                <span class="channel-subtotal-ref">
                  <span>LY ${fmtShortAmount(ly)}</span>
                  <span class="${yoyClass}">YoY ${fmtPct(yoy)}</span>
                </span>
              </div>
            </div>
          `;
          cardBox.appendChild(card);
        });
      }

      upsertChart("chartChannelMixCompare", {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "TY",
              data: tyValues,
              backgroundColor: "#0d6e6e",
              borderRadius: 6,
            },
            {
              label: "LY",
              data: lyValues,
              backgroundColor: "#94a3b8",
              borderRadius: 6,
            },
            {
              type: "line",
              label: "YoY",
              data: yoyValues,
              yAxisID: "y1",
              borderColor: "#e76f51",
              backgroundColor: "#e76f51",
              tension: 0.35,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  if (ctx.dataset.label === "YoY") {
                    return `YoY: ${fmtPct(ctx.raw)}`;
                  }
                  return `${ctx.dataset.label}: ${fmtShortAmount(ctx.raw)}`;
                },
              },
            },
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { callback: (v) => fmtShortAmount(v) },
            },
            x: {
              grid: { display: false },
            },
            y1: {
              position: "right",
              grid: { drawOnChartArea: false },
              ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` },
            },
          },
        },
      });

      upsertChart("chartChannelMixShare", {
        type: "doughnut",
        data: {
          labels,
          datasets: [
            {
              data: tyValues,
              backgroundColor: ["#0d6e6e", "#2a9d8f", "#8d99ae"],
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "62%",
          plugins: {
            legend: { position: "bottom", labels: { boxWidth: 10 } },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const total = tyValues.reduce((acc, v) => acc + v, 0) || 1;
                  const share = ((ctx.raw / total) * 100).toFixed(1);
                  return `${ctx.label}: ${fmtShortAmount(ctx.raw)} (${share}%)`;
                },
              },
            },
          },
        },
      });
    }

    function renderDimensionMixChart(rows, dimensionKey, canvasId, limit = 12) {
      const grouped = {};
      rows.forEach((row) => {
        const key = normalizeDimensionLabel(dimensionKey, row[dimensionKey]);
        if (!grouped[key]) {
          grouped[key] = { ty: 0, ly: 0 };
        }
        grouped[key].ty += Number(row.sales_period_ty) || 0;
        grouped[key].ly += Number(row.sales_period_ly) || 0;
      });

      let materialized = Object.entries(grouped)
        .map(([label, value]) => ({
          label,
          ty: value.ty,
          ly: value.ly,
          yoy: value.ly ? (value.ty - value.ly) / value.ly : null,
        }))
        .sort((a, b) => b.ty - a.ty)
        .slice(0, limit);

      if (dimensionKey === "category") {
        const knownCategories = new Set(["TOP", "BOTTOM", "OUTER", "ACC"]);
        const hasKnownWithValue = materialized.some(
          (row) => knownCategories.has(row.label) && ((Number(row.ty) || 0) !== 0 || (Number(row.ly) || 0) !== 0),
        );
        if (hasKnownWithValue) {
          materialized = materialized.filter((row) => row.label !== "Unknown");
        }
      }

      const labels = materialized.map((r) => r.label);
      const tyValues = materialized.map((r) => r.ty);
      const lyValues = materialized.map((r) => r.ly);
      const yoyValues = materialized.map((r) => r.yoy);

      upsertChart(canvasId, {
        type: "bar",
        data: {
          labels,
          datasets: [
            { label: "TY", data: tyValues, backgroundColor: "#0d6e6e", borderRadius: 6 },
            { label: "LY", data: lyValues, backgroundColor: "#94a3b8", borderRadius: 6 },
            {
              type: "line",
              label: "YoY",
              data: yoyValues,
              yAxisID: "y1",
              borderColor: "#e76f51",
              backgroundColor: "#e76f51",
              tension: 0.35,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  if (ctx.dataset.label === "YoY") return `YoY: ${fmtPct(ctx.raw)}`;
                  return `${ctx.dataset.label}: ${fmtShortAmount(ctx.raw)}`;
                },
              },
            },
          },
          scales: {
            y: { beginAtZero: true, ticks: { callback: (v) => fmtShortAmount(v) } },
            y1: {
              position: "right",
              grid: { drawOnChartArea: false },
              ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` },
            },
          },
        },
      });
    }

    function renderSegmentMixAndItemYoY(segmentRows, itemRows, itemScopeRows) {
      const segmentOrder = ["Uni", "Wms", "Kids"];
      const selectedProductType = itemProductTypeFilter ? (itemProductTypeFilter.value || "ALL") : "ALL";
      const yearSelect = document.getElementById("topStyleYearFilter");
      const seasonSelect = document.getElementById("topStyleSeasonFilter");
      const categorySelect = document.getElementById("topStyleCategoryFilter");
      const selectedYear = yearSelect ? yearSelect.value || "ALL" : "ALL";
      const selectedSeason = seasonSelect ? seasonSelect.value || "ALL" : "ALL";
      const selectedCategory = categorySelect ? categorySelect.value || "ALL" : "ALL";
      const normalizeSeason = (value) => {
        const text = normalizeString(value);
        if (!text) return "";
        const numeric = Number(text);
        return Number.isFinite(numeric) ? String(numeric) : text;
      };

      const segmentMixLegendEl = document.getElementById("segmentMixLegend");

      const normalizeSegment = (v) => {
        const raw = normalizeString(v);
        const compact = raw.toLowerCase().replace(/\s+/g, "");
        if (!compact) return "Other";
        if (compact.includes("uni") || compact.includes("unisex") || compact.includes("유니")) return "Uni";
        if (
          compact.includes("wms")
          || compact.includes("women")
          || compact.includes("woman")
          || compact.includes("우먼")
          || compact.includes("여성")
        ) return "Wms";
        if (
          compact.includes("kids")
          || compact.includes("kid")
          || compact.includes("키즈")
          || compact.includes("아동")
        ) return "Kids";
        return "Other";
      };

      const segmentLabelMap = {
        Uni: "유니",
        Wms: "우먼스",
        Kids: "키즈",
        Other: "기타",
      };

      const segmentMap = {
        Uni: { ty: 0, ly: 0 },
        Wms: { ty: 0, ly: 0 },
        Kids: { ty: 0, ly: 0 },
        Other: { ty: 0, ly: 0 },
      };
      const unknownSegments = new Map();
      segmentRows.forEach((row) => {
        const key = normalizeSegment(row.segment);
        const target = segmentMap[key] || segmentMap.Other;
        target.ty += Number(row.sales_period_ty) || 0;
        target.ly += Number(row.sales_period_ly) || 0;
        if (key === "Other") {
          const raw = normalizeString(row.segment);
          if (raw) {
            unknownSegments.set(raw, (unknownSegments.get(raw) || 0) + 1);
          }
        }
      });

      if (unknownSegments.size > 0) {
        const preview = Array.from(unknownSegments.entries())
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([name, count]) => `${name}(${count})`)
          .join(", ");
        console.warn(`[segment-mix] Unmapped segment values detected: ${preview}`);
      }

      const labels = [...segmentOrder];
      if ((segmentMap.Other.ty || 0) > 0 || (segmentMap.Other.ly || 0) > 0) {
        labels.push("Other");
      }

      const segmentStats = labels.map((key) => ({
        key,
        name: segmentLabelMap[key] || key,
        ty: segmentMap[key] ? segmentMap[key].ty : 0,
        ly: segmentMap[key] ? segmentMap[key].ly : 0,
      }));

      const labelNames = segmentStats.map((entry) => entry.name);
      const tyValues = segmentStats.map((entry) => entry.ty);
      const lyValues = segmentStats.map((entry) => entry.ly);
      const tyTotal = tyValues.reduce((acc, value) => acc + (Number(value) || 0), 0);
      const lyTotal = lyValues.reduce((acc, value) => acc + (Number(value) || 0), 0);

      const toneMap = {
        Uni: { ty: "#0d6e6e", ly: "#8ccfcc" },
        Wms: { ty: "#c05621", ly: "#f4b79c" },
        Kids: { ty: "#2563eb", ly: "#a6c8ff" },
        Other: { ty: "#64748b", ly: "#cbd5e1" },
      };
      const tyColors = segmentStats.map((entry) => (toneMap[entry.key] ? toneMap[entry.key].ty : "#64748b"));
      const lyColors = segmentStats.map((entry) => (toneMap[entry.key] ? toneMap[entry.key].ly : "#cbd5e1"));

      upsertChart("chartCombinedMixCompare", {
        type: "doughnut",
        data: {
          labels: labelNames,
          datasets: [
            {
              label: "TY",
              data: tyValues,
              backgroundColor: tyColors,
              borderWidth: 1,
              borderColor: "#ffffff",
              hoverOffset: 8,
            },
            {
              label: "LY",
              data: lyValues,
              backgroundColor: lyColors,
              borderWidth: 1,
              borderColor: "#ffffff",
              hoverOffset: 8,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "52%",
          interaction: { mode: "nearest", intersect: true },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => (items && items.length ? String(items[0].label || "") : ""),
                label: (ctx) => {
                  const datasetValues = (ctx.dataset.data || []).map((v) => Number(v) || 0);
                  const datasetTotal = datasetValues.reduce((acc, value) => acc + value, 0) || 1;
                  const raw = Number(ctx.raw) || 0;
                  const share = (raw / datasetTotal) * 100;
                  return `${ctx.dataset.label}: ${fmtShortAmount(raw)} (${share.toFixed(1)}%)`;
                },
                afterLabel: (ctx) => {
                  const otherIndex = ctx.datasetIndex === 0 ? 1 : 0;
                  const otherDataset = ctx.chart.data.datasets[otherIndex];
                  if (!otherDataset) return "";
                  const otherValues = (otherDataset.data || []).map((v) => Number(v) || 0);
                  const otherTotal = otherValues.reduce((acc, value) => acc + value, 0) || 1;
                  const otherRaw = otherValues[ctx.dataIndex] || 0;
                  const otherShare = (otherRaw / otherTotal) * 100;
                  return `${otherDataset.label}: ${fmtShortAmount(otherRaw)} (${otherShare.toFixed(1)}%)`;
                },
                footer: (items) => {
                  if (!items || !items.length) return "";
                  const idx = items[0].dataIndex;
                  const tyRaw = Number(tyValues[idx]) || 0;
                  const lyRaw = Number(lyValues[idx]) || 0;
                  const yoy = lyRaw ? (tyRaw - lyRaw) / lyRaw : null;
                  const tyShare = tyTotal ? (tyRaw / tyTotal) * 100 : 0;
                  const lyShare = lyTotal ? (lyRaw / lyTotal) * 100 : 0;
                  const shareDelta = tyShare - lyShare;
                  const deltaPrefix = shareDelta >= 0 ? "+" : "";
                  return `YoY ${fmtPct(yoy)} | 비중 차이 ${deltaPrefix}${shareDelta.toFixed(1)}%p`;
                },
              },
            },
          },
        },
      });

      if (segmentMixLegendEl) {
        const legendRows = segmentStats.map((entry) => {
          const tones = toneMap[entry.key] || { ty: "#64748b", ly: "#cbd5e1" };
          const tyShare = tyTotal ? ((entry.ty / tyTotal) * 100).toFixed(1) : "0.0";
          const lyShare = lyTotal ? ((entry.ly / lyTotal) * 100).toFixed(1) : "0.0";
          return `
            <div class="segment-legend-row">
              <span class="name">${entry.name}</span>
              <span class="segment-legend-swatches">
                <span class="segment-legend-chip"><span class="segment-legend-dot" style="background:${tones.ty}"></span>TY ${tyShare}%</span>
                <span class="segment-legend-chip"><span class="segment-legend-dot" style="background:${tones.ly}"></span>LY ${lyShare}%</span>
              </span>
            </div>
          `;
        });
        segmentMixLegendEl.innerHTML = legendRows.join("");
      }

      const normalize = (value) => String(value || "").trim();
      const scopedSource = Array.isArray(itemScopeRows) && itemScopeRows.length ? itemScopeRows : itemRows;
      let scopedRowsNoYear = [...scopedSource];

      if (selectedSeason !== "ALL") {
        const seasonToken = normalizeSeason(selectedSeason);
        scopedRowsNoYear = scopedRowsNoYear.filter(
          (row) => normalizeSeason(row.season) === seasonToken,
        );
      }
      if (selectedCategory !== "ALL") {
        scopedRowsNoYear = scopedRowsNoYear.filter((row) => normalize(row.category) === selectedCategory);
      }
      if (selectedProductType !== "ALL") {
        scopedRowsNoYear = scopedRowsNoYear.filter((row) => {
          const productType = row.product_type || classifyProductType(row.item, row.category);
          return productType === selectedProductType;
        });
      }

      let scopedItemRows =
        selectedYear !== "ALL"
          ? scopedRowsNoYear.filter((row) => normalize(row.year) === selectedYear)
          : [...scopedRowsNoYear];

      const itemMap = {};
      scopedItemRows.forEach((row) => {
        const key = String(row.item || "Unknown");
        if (!itemMap[key]) itemMap[key] = { ty: 0, ly: 0 };
        itemMap[key].ty += Number(row.sales_period_ty) || 0;
        itemMap[key].ly += Number(row.sales_period_ly) || 0;
      });
      const itemMaterialized = Object.entries(itemMap)
        .map(([item, v]) => ({ item, ty: v.ty, ly: v.ly, yoy: v.ly ? (v.ty - v.ly) / v.ly : null }))
        .sort((a, b) => b.ty - a.ty);

      if (selectedYear !== "ALL") {
        const lyTotal = itemMaterialized.reduce((acc, row) => acc + (Number(row.ly) || 0), 0);
        const numericYear = Number(selectedYear);
        const prevYearToken = Number.isFinite(numericYear)
          ? String(numericYear - 1)
          : "";

        if (lyTotal <= 0 && prevYearToken) {
          const tyRows = scopedRowsNoYear.filter((row) => normalize(row.year) === selectedYear);
          const lyRows = scopedRowsNoYear.filter((row) => normalize(row.year) === prevYearToken);

          if (tyRows.length && lyRows.length) {
            const fallbackMap = {};
            tyRows.forEach((row) => {
              const key = String(row.item || "Unknown");
              if (!fallbackMap[key]) fallbackMap[key] = { ty: 0, ly: 0 };
              fallbackMap[key].ty += Number(row.sales_period_ty) || 0;
            });
            lyRows.forEach((row) => {
              const key = String(row.item || "Unknown");
              if (!fallbackMap[key]) fallbackMap[key] = { ty: 0, ly: 0 };
              fallbackMap[key].ly += Number(row.sales_period_ty) || 0;
            });

            const recovered = Object.entries(fallbackMap)
              .map(([item, v]) => ({ item, ty: v.ty, ly: v.ly, yoy: v.ly ? (v.ty - v.ly) / v.ly : null }))
              .sort((a, b) => b.ty - a.ty);
            const recoveredLyTotal = recovered.reduce((acc, row) => acc + (Number(row.ly) || 0), 0);
            if (recoveredLyTotal > 0) {
              console.warn(
                `[item-yoy] LY recovered from previous-year TY fallback (${selectedYear} -> ${prevYearToken})`,
              );
              itemMaterialized.splice(0, itemMaterialized.length, ...recovered);
            }
          }
        }
      }

      upsertChart("chartItemYoYCompare", {
        type: "bar",
        data: {
          labels: itemMaterialized.map((r) => r.item),
          datasets: [
            { label: "TY", data: itemMaterialized.map((r) => r.ty), backgroundColor: "#1f8f9f", borderRadius: 6 },
            { label: "LY", data: itemMaterialized.map((r) => r.ly), backgroundColor: "#8d99ae", borderRadius: 6 },
            {
              type: "line",
              label: "YoY",
              data: itemMaterialized.map((r) => r.yoy),
              yAxisID: "y1",
              borderColor: "#ef476f",
              backgroundColor: "#ef476f",
              tension: 0.35,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            tooltip: {
              callbacks: {
                label: (ctx) => (ctx.dataset.label === "YoY" ? `YoY: ${fmtPct(ctx.raw)}` : `${ctx.dataset.label}: ${fmtShortAmount(ctx.raw)}`),
              },
            },
          },
          scales: {
            y: { beginAtZero: true, ticks: { callback: (v) => fmtShortAmount(v) } },
            y1: { position: "right", grid: { drawOnChartArea: false }, ticks: { callback: (v) => `${(v * 100).toFixed(0)}%` } },
          },
        },
      });
    }

    function renderDataTable(containerId, columns, rows, limit, options = {}) {
      const container = document.getElementById(containerId);
      container.innerHTML = "";

      if (!rows.length) {
        container.innerHTML = "<div class='empty-note'>데이터 없음</div>";
        return;
      }

      const cfg = {
        sortable: options.sortable ?? false,
        searchable: options.searchable ?? false,
        paginated: options.paginated ?? false,
        pageSize: options.pageSize ?? 12,
        fit: options.fit ?? false,
      };

      const sourceRows = rows.slice(0, limit);
      const wrapper = document.createElement("div");

      let query = "";
      let sortKey = "";
      let sortAsc = false;
      let page = 1;

      const toolbar = document.createElement("div");
      toolbar.className = "table-toolbar";
      const left = document.createElement("div");
      const right = document.createElement("div");
      right.className = "table-pagination";

      if (cfg.searchable) {
        const input = document.createElement("input");
        input.type = "search";
        input.placeholder = "검색 (전체 컬럼)";
        input.oninput = () => {
          query = input.value.trim().toLowerCase();
          page = 1;
          drawBody();
        };
        left.appendChild(input);
      }

      const prevBtn = document.createElement("button");
      prevBtn.textContent = "이전";
      const nextBtn = document.createElement("button");
      nextBtn.textContent = "다음";
      const pageInfo = document.createElement("span");

      if (cfg.paginated) {
        prevBtn.onclick = () => {
          page = Math.max(1, page - 1);
          drawBody();
        };
        nextBtn.onclick = () => {
          page += 1;
          drawBody();
        };
        right.appendChild(prevBtn);
        right.appendChild(pageInfo);
        right.appendChild(nextBtn);
      }

      if (cfg.searchable || cfg.paginated) {
        toolbar.appendChild(left);
        toolbar.appendChild(right);
        wrapper.appendChild(toolbar);
      }

      const table = document.createElement("table");
      if (cfg.fit) {
        table.classList.add("fit-table");
      }

      const isRateColumn = (col) =>
        col.format === "pct" ||
        col.key.endsWith("_pct") ||
        col.key.endsWith("_p") ||
        col.key.includes("rate");

      const isNumericColumn = (col) =>
        col.format === "million" ||
        col.format === "pct_plain" ||
        isRateColumn(col) ||
        col.key.includes("qty") ||
        col.key.includes("sales") ||
        col.key.includes("margin") ||
        col.key.includes("cost") ||
        col.key.includes("amount") ||
        col.key.includes("amt") ||
        col.key.includes("rank");

      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col.label;
        if (isNumericColumn(col)) {
          th.classList.add("th-num");
        }
        if (cfg.sortable) {
          th.classList.add("sortable");
          th.onclick = () => {
            if (sortKey === col.key) {
              sortAsc = !sortAsc;
            } else {
              sortKey = col.key;
              sortAsc = false;
            }
            page = 1;
            drawBody();
          };
        }
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      table.appendChild(tbody);
      wrapper.appendChild(table);
      container.appendChild(wrapper);

      function filteredAndSortedRows() {
        let out = sourceRows;
        if (query) {
          out = out.filter((row) =>
            columns.some((col) => String(row[col.key] ?? "").toLowerCase().includes(query))
          );
        }

        if (cfg.sortable && sortKey) {
          out = [...out].sort((a, b) => {
            const va = a[sortKey];
            const vb = b[sortKey];
            if (va == null && vb == null) return 0;
            if (va == null) return 1;
            if (vb == null) return -1;
            if (typeof va === "number" && typeof vb === "number") {
              return sortAsc ? va - vb : vb - va;
            }
            const sa = String(va);
            const sb = String(vb);
            return sortAsc ? sa.localeCompare(sb) : sb.localeCompare(sa);
          });
        }
        return out;
      }

      function drawBody() {
        const materialized = filteredAndSortedRows();
        const totalPages = cfg.paginated ? Math.max(1, Math.ceil(materialized.length / cfg.pageSize)) : 1;
        if (page > totalPages) page = totalPages;

        const pageRows = cfg.paginated
          ? materialized.slice((page - 1) * cfg.pageSize, page * cfg.pageSize)
          : materialized;

        tbody.innerHTML = "";
        pageRows.forEach((row) => {
          const tr = document.createElement("tr");
          columns.forEach((col) => {
            const td = document.createElement("td");
            const value = row[col.key];
            const numericCell = isNumericColumn(col);

            if (col.key === "rank") {
              td.classList.add("cell-center");
            } else {
              td.classList.add(numericCell ? "cell-num" : "cell-text");
            }

            if (col.format === "million") {
              td.textContent = fmtMillion(value);
            } else if (col.format === "pct_plain") {
              td.textContent = fmtPctPlain(value);
            } else if (isRateColumn(col)) {
              td.textContent = fmtPct(value);
              td.classList.add(trendClass(value));
            } else if (col.key.includes("qty")) {
              td.textContent = fmtNum(value);
            } else if (col.key.includes("sales") || col.key.includes("margin") || col.key.includes("cost")) {
              td.textContent = fmtShortAmount(value);
            } else if (typeof value === "number") {
              td.textContent = fmtNum(value);
            } else {
              td.textContent = value ?? "-";
            }
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        });

        if (cfg.paginated) {
          pageInfo.textContent = `${page} / ${totalPages} (${materialized.length} rows)`;
          prevBtn.disabled = page <= 1;
          nextBtn.disabled = page >= totalPages;
        }
      }

      drawBody();
    }

    function setText(id, value) {
      const el = document.getElementById(id);
      if (el) {
        el.textContent = value;
      }
    }

    function updateNumericPanels(brandRows, channelRows, categoryRows, itemRows, storeRows) {
      const salesTy = sumBy(brandRows, "sales_period_ty");
      const salesLy = sumBy(brandRows, "sales_period_ly");
      const salesLw = sumBy(brandRows, "sales_period_prev_week_ty");
      const qtyTy = sumBy(brandRows, "qty_period_ty");
      const qtyLy = sumBy(brandRows, "qty_period_ly");
      const qtyLw = sumBy(brandRows, "qty_period_prev_week_ty");

      const yoyDiff = salesTy - salesLy;
      const wowDiff = salesTy - salesLw;
      const yoyPct = safePct(yoyDiff, salesLy);
      const wowPct = safePct(wowDiff, salesLw);
      const asp = safePct(salesTy, qtyTy);

      setText("kpiSalesTy", fmtShortAmount(salesTy));
      setText("kpiSalesTySub", `LY ${fmtShortAmount(salesLy)} / LW ${fmtShortAmount(salesLw)}`);
      setText("kpiQtyTy", fmtNum(qtyTy));
      setText("kpiQtyTySub", `LY ${fmtNum(qtyLy)} / LW ${fmtNum(qtyLw)}`);
      setText("kpiAsp", fmtNum(asp));
      setText("kpiAspSub", `총액 ${fmtNum(salesTy)} / 수량 ${fmtNum(qtyTy)}`);

      const kpiYoY = document.getElementById("kpiYoY");
      if (kpiYoY) {
        kpiYoY.textContent = fmtPct(yoyPct);
        kpiYoY.className = `kpi-value ${trendClass(yoyPct)}`;
      }
      setText("kpiYoYSub", `${fmtSignedAmount(yoyDiff)} vs LY`);

      const kpiWoW = document.getElementById("kpiWoW");
      if (kpiWoW) {
        kpiWoW.textContent = fmtPct(wowPct);
        kpiWoW.className = `kpi-value ${trendClass(wowPct)}`;
      }
      setText("kpiWoWSub", `${fmtSignedAmount(wowDiff)} vs LW`);

      setText("bannerTy", fmtShortAmount(salesTy));
      setText("bannerTySub", `Qty ${fmtNum(qtyTy)}`);
      setText("bannerLy", fmtShortAmount(salesLy));
      setText("bannerLySub", `YoY Diff ${fmtSignedAmount(yoyDiff)}`);
      setText("bannerLw", fmtShortAmount(salesLw));
      setText("bannerLwSub", `WoW Diff ${fmtSignedAmount(wowDiff)}`);
      setText("bannerGrowth", `${fmtPct(yoyPct)} / ${fmtPct(wowPct)}`);
      setText("bannerGrowthSub", "YoY / WoW");
    }

    function renderTopInsights(storeRows, styleRows, styleScopeRows, onScopeChange = null) {
      const topStores = [...storeRows]
        .sort((a, b) => (b.sales_period_ty || 0) - (a.sales_period_ty || 0))
        .slice(0, 20)
        .map((row, index) => ({ ...row, rank: index + 1 }));

      const yearSelect = document.getElementById("topStyleYearFilter");
      const seasonSelect = document.getElementById("topStyleSeasonFilter");
      const categorySelect = document.getElementById("topStyleCategoryFilter");
      const productTypeSelect = document.getElementById("itemProductTypeFilter");
      const scopeTagEl = document.getElementById("topStyleScopeTag");

      const normalize = (value) => String(value || "").trim();
      const normalizeSeason = (value) => {
        const text = normalize(value);
        if (!text) return "";
        const numeric = Number(text);
        return Number.isFinite(numeric) ? String(numeric) : text;
      };
      const scopedRows = Array.isArray(styleScopeRows) ? styleScopeRows : [];
      const hasScopedRows = scopedRows.length > 0;
      const styleFallbackByCode = new Map();

      (Array.isArray(styleRows) ? styleRows : []).forEach((row) => {
        const styleNo = normalize(row.style_no);
        if (!styleNo || styleFallbackByCode.has(styleNo)) return;
        styleFallbackByCode.set(styleNo, {
          category: normalize(row.category),
          wow_sales_pct: row.wow_sales_pct,
          cumulative_sell_through_ty: row.cumulative_sell_through_ty,
          season_sales_rate_ty: row.season_sales_rate_ty,
          order_qty_total: row.order_qty_total,
          qty_cum_ty: row.qty_cum_ty,
        });
      });

      const uniqueValues = (rows, key) =>
        Array.from(new Set(rows.map((row) => normalize(row[key])).filter(Boolean))).sort(
          (a, b) => a.localeCompare(b, undefined, { numeric: true }),
        );

      const aggregateStyleRows = (rows) => {
        const grouped = new Map();
        rows.forEach((row) => {
          const styleNo = normalize(row.style_no);
          const styleName = normalize(row.style_name);
          const category = normalize(row.category);
          const key = `${styleNo}||${styleName}||${category}`;
            if (!grouped.has(key)) {
              grouped.set(key, {
                style_no: styleNo,
                style_name: styleName || styleNo,
                category: category || "-",
              sales_period_ty: 0,
              sales_period_ly: 0,
              sales_period_prev_week_ty: 0,
              sales_cum_ty: 0,
              tag_cum_ty: 0,
              qty_cum_ty: 0,
              order_qty_total: 0,
              order_amt_total: 0,
              is_26_new: false,
            });
          }

          const target = grouped.get(key);
          target.sales_period_ty += Number(row.sales_period_ty) || 0;
          target.sales_period_ly += Number(row.sales_period_ly) || 0;
          target.sales_period_prev_week_ty += Number(row.sales_period_prev_week_ty) || 0;
          target.sales_cum_ty += Number(row.sales_cum_ty) || 0;
          target.tag_cum_ty += Number(row.tag_cum_ty) || 0;
          target.qty_cum_ty += Number(row.qty_cum_ty) || 0;
          target.order_qty_total += Number(row.order_qty_total) || 0;
          target.order_amt_total += Number(row.order_amt_total) || 0;
          target.is_26_new = target.is_26_new || Boolean(row.is_26_new);
        });

        return Array.from(grouped.values()).map((row) => {
          row.yoy_sales_diff = row.sales_period_ty - row.sales_period_ly;
          row.yoy_sales_pct = safePct(row.yoy_sales_diff, row.sales_period_ly);
          row.wow_sales_diff = row.sales_period_ty - row.sales_period_prev_week_ty;
          row.wow_sales_pct = safePct(row.wow_sales_diff, row.sales_period_prev_week_ty);
          row.season_sales_rate_ty = safePct(row.sales_cum_ty, row.tag_cum_ty);
          row.cumulative_sell_through_ty = safePct(row.qty_cum_ty, row.order_qty_total);

          const fallback = styleFallbackByCode.get(normalize(row.style_no));
          if (fallback) {
            if ((!row.category || row.category === "-") && fallback.category) {
              row.category = fallback.category;
            }
            if (isMissing(row.wow_sales_pct) && !isMissing(fallback.wow_sales_pct)) {
              row.wow_sales_pct = fallback.wow_sales_pct;
            }
            if (
              (isMissing(row.order_qty_total) || Number(row.order_qty_total) <= 0) &&
              !isMissing(fallback.order_qty_total)
            ) {
              row.order_qty_total = Number(fallback.order_qty_total) || 0;
            }
            if (
              (isMissing(row.qty_cum_ty) || Number(row.qty_cum_ty) <= 0) &&
              !isMissing(fallback.qty_cum_ty)
            ) {
              row.qty_cum_ty = Number(fallback.qty_cum_ty) || 0;
            }
            if (
              isMissing(row.cumulative_sell_through_ty) &&
              !isMissing(fallback.cumulative_sell_through_ty)
            ) {
              row.cumulative_sell_through_ty = fallback.cumulative_sell_through_ty;
            }
            if (isMissing(row.season_sales_rate_ty) && !isMissing(fallback.season_sales_rate_ty)) {
              row.season_sales_rate_ty = fallback.season_sales_rate_ty;
            }
          }

          if (isMissing(row.cumulative_sell_through_ty)) {
            row.cumulative_sell_through_ty = safePct(row.qty_cum_ty, row.order_qty_total);
          }
          return row;
        });
      };

      const setAllOption = (selectEl) => {
        selectEl.innerHTML = "";
        const allOpt = document.createElement("option");
        allOpt.value = "ALL";
        allOpt.textContent = "전체";
        selectEl.appendChild(allOpt);
      };

      const populateYears = (preferredYear) => {
        if (!yearSelect) return;
        const currentYear = preferredYear || yearSelect.value || "ALL";
        setAllOption(yearSelect);
        if (!hasScopedRows) {
          yearSelect.value = "ALL";
          return;
        }

        const years = uniqueValues(scopedRows, "year");
        years.forEach((year) => {
          const opt = document.createElement("option");
          opt.value = year;
          opt.textContent = year;
          yearSelect.appendChild(opt);
        });
        yearSelect.value = years.includes(currentYear) ? currentYear : "ALL";
      };

      const populateSeasons = (preferredSeason) => {
        if (!seasonSelect) return;
        const currentSeason = preferredSeason || seasonSelect.value || "ALL";
        setAllOption(seasonSelect);
        if (!hasScopedRows) {
          seasonSelect.value = "ALL";
          return;
        }

        const selectedYear = yearSelect ? yearSelect.value || "ALL" : "ALL";
        const seasonBase =
          selectedYear === "ALL"
            ? scopedRows
            : scopedRows.filter((row) => normalize(row.year) === selectedYear);
        const seasons = Array.from(
          new Set(seasonBase.map((row) => normalizeSeason(row.season)).filter(Boolean)),
        ).sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
        seasons.forEach((season) => {
          const opt = document.createElement("option");
          opt.value = season;
          opt.textContent = season;
          seasonSelect.appendChild(opt);
        });
        const currentSeasonToken = normalizeSeason(currentSeason);
        seasonSelect.value = seasons.includes(currentSeasonToken)
          ? currentSeasonToken
          : "ALL";
      };

      const populateCategories = (preferredCategory) => {
        if (!categorySelect) return;
        const currentCategory = preferredCategory || categorySelect.value || "ALL";
        setAllOption(categorySelect);

        if (!hasScopedRows) {
          const fallbackCategories = uniqueValues(styleRows, "category");
          fallbackCategories.forEach((category) => {
            const opt = document.createElement("option");
            opt.value = category;
            opt.textContent = category;
            categorySelect.appendChild(opt);
          });
          categorySelect.value = fallbackCategories.includes(currentCategory)
            ? currentCategory
            : "ALL";
          return;
        }

        const selectedYear = yearSelect ? yearSelect.value || "ALL" : "ALL";
        const selectedSeason = seasonSelect ? seasonSelect.value || "ALL" : "ALL";

        let categoryBase = scopedRows;
        if (selectedYear !== "ALL") {
          categoryBase = categoryBase.filter((row) => normalize(row.year) === selectedYear);
        }
        if (selectedSeason !== "ALL") {
          const seasonToken = normalizeSeason(selectedSeason);
          categoryBase = categoryBase.filter(
            (row) => normalizeSeason(row.season) === seasonToken,
          );
        }

        const categories = uniqueValues(categoryBase, "category");
        categories.forEach((category) => {
          const opt = document.createElement("option");
          opt.value = category;
          opt.textContent = category;
          categorySelect.appendChild(opt);
        });
        categorySelect.value = categories.includes(currentCategory) ? currentCategory : "ALL";
      };

      const resolveScopedStyleRows = () => {
        const selectedCategory = categorySelect ? categorySelect.value || "ALL" : "ALL";
        const selectedProductType = productTypeSelect
          ? productTypeSelect.value || "ALL"
          : "ALL";
        const matchProductType = (row) => {
          if (selectedProductType === "ALL") return true;
          const resolvedType = row.product_type || classifyProductType(row.item, row.category);
          return resolvedType === selectedProductType;
        };

        if (!hasScopedRows || !yearSelect || !seasonSelect) {
          let fallback = [...styleRows];
          if (selectedCategory !== "ALL") {
            fallback = fallback.filter(
              (row) => normalize(row.category) === selectedCategory,
            );
          }
          fallback = fallback.filter(matchProductType);
          return fallback;
        }

        const selectedYear = yearSelect.value || "ALL";
        const selectedSeason = seasonSelect.value || "ALL";
        if (
          selectedYear === "ALL" &&
          selectedSeason === "ALL" &&
          selectedCategory === "ALL"
        ) {
          return [...styleRows].filter(matchProductType);
        }

        let filtered = scopedRows;
        if (selectedYear !== "ALL") {
          filtered = filtered.filter((row) => normalize(row.year) === selectedYear);
        }
        if (selectedSeason !== "ALL") {
          const seasonToken = normalizeSeason(selectedSeason);
          filtered = filtered.filter((row) => normalizeSeason(row.season) === seasonToken);
        }
        if (selectedCategory !== "ALL") {
          filtered = filtered.filter(
            (row) => normalize(row.category) === selectedCategory,
          );
        }
        filtered = filtered.filter(matchProductType);
        return aggregateStyleRows(filtered);
      };

      const updateScopeTag = () => {
        if (!scopeTagEl || !yearSelect || !seasonSelect || !categorySelect) return;
        const parts = [];
        if (yearSelect.value !== "ALL") parts.push(`연도 ${yearSelect.value}`);
        if (seasonSelect.value !== "ALL") parts.push(`시즌 ${seasonSelect.value}`);
        if (categorySelect.value !== "ALL") parts.push(`카테고리 ${categorySelect.value}`);
        if (productTypeSelect && productTypeSelect.value !== "ALL") {
          parts.push(`상품구분 ${productTypeSelect.value}`);
        }
        scopeTagEl.textContent = `스타일 필터: ${parts.length ? parts.join(" / ") : "전체"}`;
      };

      const drawStyleTables = () => {
        const rankedSource = resolveScopedStyleRows().map((row) => {
          const qtyCum = Number(row.qty_cum_ty) || 0;
          const orderQty = Number(row.order_qty_total) || 0;
          const fallbackCumulative = Number(row.cumulative_sell_through_ty);
          const fallbackSeasonRate = Number(row.season_sales_rate_ty);
          const cumulativeSellThrough = orderQty
            ? qtyCum / orderQty
            : Number.isFinite(fallbackCumulative)
              ? fallbackCumulative
              : Number.isFinite(fallbackSeasonRate)
                ? fallbackSeasonRate
                : null;
          return {
            ...row,
            cumulative_sell_through_ty: cumulativeSellThrough,
          };
        });

        const bestStyles = [...rankedSource]
          .sort((a, b) => (b.sales_period_ty || 0) - (a.sales_period_ty || 0))
          .slice(0, 10)
          .map((row, index) => ({ ...row, rank: index + 1 }));

        const worstStyles = [...rankedSource]
          .filter(
            (row) => !((Number(row.sales_period_ty) || 0) <= 0 && Boolean(row.is_26_new)),
          )
          .sort((a, b) => {
            const aWow = Number(a.wow_sales_pct);
            const bWow = Number(b.wow_sales_pct);

            const aDeclinePriority = Number.isFinite(aWow) && aWow < 0 ? 0 : 1;
            const bDeclinePriority = Number.isFinite(bWow) && bWow < 0 ? 0 : 1;
            if (aDeclinePriority !== bDeclinePriority) {
              return aDeclinePriority - bDeclinePriority;
            }

            const aSeasonRate = Number(a.cumulative_sell_through_ty);
            const bSeasonRate = Number(b.cumulative_sell_through_ty);
            const aSeasonPriority = Number.isFinite(aSeasonRate)
              ? aSeasonRate
              : Number.POSITIVE_INFINITY;
            const bSeasonPriority = Number.isFinite(bSeasonRate)
              ? bSeasonRate
              : Number.POSITIVE_INFINITY;
            if (aSeasonPriority !== bSeasonPriority) {
              return aSeasonPriority - bSeasonPriority;
            }

            const aWowPriority = Number.isFinite(aWow) ? aWow : Number.POSITIVE_INFINITY;
            const bWowPriority = Number.isFinite(bWow) ? bWow : Number.POSITIVE_INFINITY;
            if (aWowPriority !== bWowPriority) {
              return aWowPriority - bWowPriority;
            }

            const aSales = Number(a.sales_period_ty) || 0;
            const bSales = Number(b.sales_period_ty) || 0;
            if (aSales !== bSales) {
              return aSales - bSales;
            }

            return String(a.style_no || "").localeCompare(String(b.style_no || ""));
          })
          .slice(0, 10)
          .map((row, index) => ({ ...row, rank: index + 1 }));

        renderDataTable(
          "bestStyleTable",
          [
            { key: "rank", label: "순위" },
            { key: "style_name", label: "스타일" },
            { key: "style_no", label: "스타일코드" },
            { key: "category", label: "카테고리" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
          ],
          bestStyles,
          10,
          { sortable: true, searchable: false, paginated: false, fit: true },
        );

        renderDataTable(
          "worstStyleTable",
          [
            { key: "rank", label: "순위" },
            { key: "style_name", label: "스타일" },
            { key: "style_no", label: "스타일코드" },
            { key: "category", label: "카테고리" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
          ],
          worstStyles,
          10,
          { sortable: true, searchable: false, paginated: false, fit: true },
        );

        updateScopeTag();
      };

      const triggerScopeChange = () => {
        if (typeof onScopeChange === "function") {
          onScopeChange();
        }
      };

      renderDataTable(
        "topStoreTable",
        [
          { key: "rank", label: "순위" },
          { key: "store_name", label: "매장" },
          { key: "channel", label: "채널" },
          { key: "sales_period_ty", label: "당주 매출", format: "million" },
          { key: "yoy_sales_pct", label: "YoY" },
          { key: "wow_sales_pct", label: "WoW" },
        ],
        topStores,
        20,
        { sortable: true, searchable: false, paginated: false, fit: true },
      );

      if (yearSelect && seasonSelect && categorySelect) {
        const selectedYear = yearSelect.value || "ALL";
        const selectedSeason = seasonSelect.value || "ALL";
        const selectedCategory = categorySelect.value || "ALL";
        populateYears(selectedYear);
        populateSeasons(selectedSeason);
        populateCategories(selectedCategory);
        yearSelect.onchange = () => {
          populateSeasons("ALL");
          populateCategories("ALL");
          drawStyleTables();
          triggerScopeChange();
        };
        seasonSelect.onchange = () => {
          populateCategories("ALL");
          drawStyleTables();
          triggerScopeChange();
        };
        categorySelect.onchange = () => {
          drawStyleTables();
          triggerScopeChange();
        };
      }

      drawStyleTables();
    }

    function renderItemSalesShareSection(currentBrand) {
      const seasonRows = (dataset.season || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const itemScopeRows = (dataset.item_scope || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const scopeTagEl = document.getElementById("itemSalesScopeTag");
      if (!scopeTagEl) return;

      const normalize = (value) => String(value || "").trim();
      const parseYear = (value) => {
        const token = normalize(value);
        const matched = token.match(/([0-9]{2})/);
        return matched ? Number(matched[1]) : Number.NaN;
      };

      const parseSeason = (value) => {
        const token = normalize(value);
        const numeric = Number(token);
        return Number.isFinite(numeric) ? numeric : Number.NaN;
      };

      const halfOrder = (value) => {
        const token = normalize(value).toUpperCase();
        if (token === "SS") return 0;
        if (token === "FW") return 1;
        return 9;
      };

      const formatSeasonMetaKey = (yearNum, season) => `${yearNum}||${season}`;
      const formatBucketKey = (yearNum, seasonHalf, season) => `${yearNum}||${seasonHalf}||${season}`;
      const buildLabel = (yearNum, seasonHalf, season) => {
        const yy = String(yearNum).padStart(2, "0");
        if (seasonHalf) {
          return `${yy}${seasonHalf}-${season}`;
        }
        return `${yy}-${season}`;
      };

      const seasonMetaMap = new Map();
      seasonRows.forEach((row) => {
        const yearNum = parseYear(normalize(row.year_season) || normalize(row.year));
        const season = normalize(row.season);
        const seasonHalf = normalize(row.season_half).toUpperCase();
        if (!Number.isFinite(yearNum) || !season) {
          return;
        }
        const key = formatSeasonMetaKey(yearNum, season);
        if (!seasonMetaMap.has(key)) {
          seasonMetaMap.set(key, { season_half: seasonHalf });
        }
      });

      const bucketMap = new Map();
      if (itemScopeRows.length) {
        itemScopeRows.forEach((row) => {
          const yearNum = parseYear(normalize(row.year));
          const season = normalize(row.season);
          if (!Number.isFinite(yearNum) || !season) {
            return;
          }

          const seasonMeta = seasonMetaMap.get(formatSeasonMetaKey(yearNum, season));
          const seasonHalf = normalize(seasonMeta?.season_half).toUpperCase();
          const key = formatBucketKey(yearNum, seasonHalf, season);
          if (!bucketMap.has(key)) {
            bucketMap.set(key, {
              year_num: yearNum,
              season_half: seasonHalf,
              season,
              sales_period_ty: 0,
              sales_period_ly: 0,
            });
          }

          const target = bucketMap.get(key);
          target.sales_period_ty += Number(row.sales_period_ty) || 0;
          target.sales_period_ly += Number(row.sales_period_ly) || 0;
        });
      } else {
        seasonRows.forEach((row) => {
          const yearNum = parseYear(normalize(row.year_season) || normalize(row.year));
          const season = normalize(row.season);
          const seasonHalf = normalize(row.season_half).toUpperCase();
          if (!Number.isFinite(yearNum) || !season) {
            return;
          }

          const key = formatBucketKey(yearNum, seasonHalf, season);
          if (!bucketMap.has(key)) {
            bucketMap.set(key, {
              year_num: yearNum,
              season_half: seasonHalf,
              season,
              sales_period_ty: 0,
              sales_period_ly: 0,
            });
          }

          const target = bucketMap.get(key);
          target.sales_period_ty += Number(row.sales_period_ty) || 0;
          target.sales_period_ly += Number(row.sales_period_ly) || 0;
        });
      }

      const allBuckets = Array.from(bucketMap.values());
      const totalTySales = allBuckets.reduce(
        (acc, row) => acc + (Number(row.sales_period_ty) || 0),
        0,
      );
      const totalLySales = allBuckets.reduce(
        (acc, row) => acc + (Number(row.sales_period_ly) || 0),
        0,
      );

      if (!allBuckets.length || totalTySales <= 0) {
        scopeTagEl.textContent = "1:1 매칭 가능한 전년 데이터가 없습니다.";
        upsertChart("chartItemSalesShare", {
          type: "bar",
          data: { labels: [], datasets: [] },
          options: { responsive: true, maintainAspectRatio: false },
        });
        return;
      }

      const pairedRows = allBuckets
        .map((row) => {
          const tyAmount = Number(row.sales_period_ty) || 0;
          const lyAmount = Number(row.sales_period_ly) || 0;
          const tyShare = totalTySales ? tyAmount / totalTySales : null;
          const lyShare = totalLySales ? lyAmount / totalLySales : null;
          const growthRate = lyAmount > 0 ? (tyAmount - lyAmount) / lyAmount : null;

          return {
            year_num: row.year_num,
            season_half: row.season_half,
            season: row.season,
            ty_label: buildLabel(row.year_num, row.season_half, row.season),
            ly_label: buildLabel(row.year_num - 1, row.season_half, row.season),
            ty_amount: tyAmount,
            ly_amount: lyAmount,
            ty_share: tyShare,
            ly_share: lyShare,
            growth_rate: growthRate,
          };
        })
        .sort((a, b) => {
          const seasonDiff = parseSeason(a.season) - parseSeason(b.season);
          if (Number.isFinite(seasonDiff) && seasonDiff !== 0) return seasonDiff;

          const halfDiff = halfOrder(a.season_half) - halfOrder(b.season_half);
          if (halfDiff !== 0) return halfDiff;

          const yearDiff = Number(b.year_num) - Number(a.year_num);
          if (Number.isFinite(yearDiff) && yearDiff !== 0) return yearDiff;

          return a.ty_label.localeCompare(b.ty_label, undefined, { numeric: true });
        });

      const chartRows = pairedRows.filter((row) => {
        if ((row.ty_amount || 0) <= 0 || (row.ly_amount || 0) <= 0) {
          return false;
        }
        const tyShare = Number(row.ty_share) || 0;
        return tyShare >= 0.01;
      });

      const labels = chartRows.map((row) => row.ty_label);
      const tySalesValues = chartRows.map((row) => row.ty_amount);
      const lySalesValues = chartRows.map((row) => row.ly_amount);
      const tyShareValues = chartRows.map((row) => row.ty_share);
      const lyShareValues = chartRows.map((row) => row.ly_share);
      const growthValues = chartRows.map((row) => row.growth_rate);

      scopeTagEl.textContent = `TY시트 ${chartRows.length}개 시즌 vs LY시트 1:1 매칭 | TY ${fmtShortAmount(totalTySales)} / LY ${fmtShortAmount(totalLySales)} (TY 1% 미만 제외)`;

      upsertChart("chartItemSalesShare", {
        type: "bar",
        data: {
          labels,
          datasets: [
            {
              label: "TY 매출",
              data: tySalesValues,
              backgroundColor: "#1f8f9f",
              borderRadius: 6,
            },
            {
              label: "LY 매출",
              data: lySalesValues,
              backgroundColor: "#7f8ea3",
              borderRadius: 6,
            },
            {
              type: "line",
              label: "성장율",
              data: growthValues,
              yAxisID: "y1",
              borderColor: "#2a9d8f",
              backgroundColor: "#2a9d8f",
              tension: 0.3,
              pointRadius: 2,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: {
            mode: "index",
            intersect: false,
          },
            plugins: {
              tooltip: {
                callbacks: {
                  title: (items) => {
                    const idx = Number(items?.[0]?.dataIndex) || 0;
                    const row = chartRows[idx];
                    if (!row) return "";
                    return `${row.ty_label} vs ${row.ly_label}`;
                  },
                  label: (ctx) => {
                    if (ctx.dataset.label === "성장율") {
                      const idx = Number(ctx.dataIndex) || 0;
                      const shareDelta =
                        idx >= 0
                          ? (Number(tyShareValues[idx]) || 0) - (Number(lyShareValues[idx]) || 0)
                          : null;
                      return `${ctx.dataset.label}: ${fmtPct(ctx.raw)} | 비중변화 ${fmtPctPointPlain(shareDelta)}`;
                    }

                    const amount = Number(ctx.raw) || 0;
                    const idx = Number(ctx.dataIndex) || 0;
                    const share =
                      idx >= 0
                        ? (ctx.dataset.label === "TY 매출" ? tyShareValues[idx] : lyShareValues[idx])
                        : null;
                    return `${ctx.dataset.label}: ${fmtShortAmount(amount)} (매출비중 ${fmtPctPlain(share)})`;
                  },
                },
              },
            },
          scales: {
            x: {
              ticks: {
                autoSkip: true,
                maxRotation: 35,
                minRotation: 20,
              },
            },
            y: {
              beginAtZero: true,
              ticks: { callback: (value) => fmtShortAmount(value) },
            },
            y1: {
              position: "right",
              grid: { drawOnChartArea: false },
              ticks: { callback: (value) => fmtPctPlain(Number(value)) },
            },
          },
        },
      });
    }

    function renderStyleNoDrilldown(currentBrand) {
      const styleChannelRows = (dataset.style_channel_period || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const styleStoreRows = (dataset.style_store_period || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );

      const yearSelect = document.getElementById("styleNoYearFilter");
      const halfSelect = document.getElementById("styleNoHalfFilter");
      const seasonSelect = document.getElementById("styleNoSeasonFilter");
      const lookup = document.getElementById("styleNoLookup");
      const search = document.getElementById("styleNoSearch");
      const scopeTagEl = document.getElementById("styleNoScopeTag");
      if (!yearSelect || !halfSelect || !seasonSelect || !lookup || !search || !scopeTagEl) return;

      const normalize = (value) => String(value || "").trim();
      const allRows = [...styleChannelRows, ...styleStoreRows];

      const selectLabel = (value) => {
        const text = normalize(value);
        return text || "미분류";
      };

      const uniqueValues = (rows, key) =>
        Array.from(new Set(rows.map((row) => normalize(row[key])).filter(Boolean))).sort(
          (a, b) => a.localeCompare(b, undefined, { numeric: true }),
        );

      const filterByScope = (rows, override = {}) => {
        const yearValue = override.yearValue ?? (yearSelect.value || "ALL");
        const halfValue = override.halfValue ?? (halfSelect.value || "ALL");
        const seasonValue = override.seasonValue ?? (seasonSelect.value || "ALL");

        let scoped = rows;
        if (yearValue !== "ALL") {
          scoped = scoped.filter((row) => normalize(row.year) === yearValue);
        }
        if (halfValue !== "ALL") {
          scoped = scoped.filter((row) => normalize(row.season_half) === halfValue);
        }
        if (seasonValue !== "ALL") {
          scoped = scoped.filter((row) => normalize(row.season) === seasonValue);
        }
        return scoped;
      };

      const createStyleMap = (rows) => {
        const styleMap = new Map();
        rows.forEach((row) => {
          const styleNo = normalize(row.style_no);
          if (!styleNo) return;
          const styleName = normalize(row.style_name);
          if (!styleMap.has(styleNo)) {
            styleMap.set(styleNo, { style_no: styleNo, style_name: styleName });
            return;
          }
          if (!styleMap.get(styleNo).style_name && styleName) {
            styleMap.set(styleNo, { style_no: styleNo, style_name: styleName });
          }
        });
        return styleMap;
      };

      const setAllOption = (selectEl) => {
        selectEl.innerHTML = "";
        const allOpt = document.createElement("option");
        allOpt.value = "ALL";
        allOpt.textContent = "전체";
        selectEl.appendChild(allOpt);
        return allOpt;
      };

      const populateYears = (preferredYear) => {
        const currentYear = preferredYear || yearSelect.value || "ALL";
        setAllOption(yearSelect);
        const years = uniqueValues(allRows, "year");
        years.forEach((year) => {
          const opt = document.createElement("option");
          opt.value = year;
          opt.textContent = year;
          yearSelect.appendChild(opt);
        });
        yearSelect.value = years.includes(currentYear) ? currentYear : "ALL";
      };

      const populateHalves = (preferredHalf) => {
        const currentHalf = preferredHalf || halfSelect.value || "ALL";
        setAllOption(halfSelect);
        const scopedRows = filterByScope(allRows, {
          yearValue: yearSelect.value || "ALL",
          halfValue: "ALL",
          seasonValue: "ALL",
        });
        const halves = uniqueValues(scopedRows, "season_half");
        halves.forEach((half) => {
          const opt = document.createElement("option");
          opt.value = half;
          opt.textContent = half;
          halfSelect.appendChild(opt);
        });
        halfSelect.value = halves.includes(currentHalf) ? currentHalf : "ALL";
      };

      const populateSeasons = (preferredSeason) => {
        const currentSeason = preferredSeason || seasonSelect.value || "ALL";
        setAllOption(seasonSelect);
        const scopedRows = filterByScope(allRows, {
          yearValue: yearSelect.value || "ALL",
          halfValue: halfSelect.value || "ALL",
          seasonValue: "ALL",
        });
        const seasons = uniqueValues(scopedRows, "season");
        seasons.forEach((season) => {
          const opt = document.createElement("option");
          opt.value = season;
          opt.textContent = season;
          seasonSelect.appendChild(opt);
        });
        seasonSelect.value = seasons.includes(currentSeason) ? currentSeason : "ALL";
      };

      const populateStyles = (preferredStyleNo) => {
        const currentStyleNo = preferredStyleNo || lookup.value || "";
        const query = normalize(search.value).toLowerCase();
        const scopedRows = filterByScope(allRows);
        const styleMap = createStyleMap(scopedRows);
        let styleRows = Array.from(styleMap.values()).sort((a, b) =>
          a.style_no.localeCompare(b.style_no, undefined, { numeric: true }),
        );

        if (query) {
          styleRows = styleRows.filter((row) =>
            `${row.style_no} ${row.style_name}`.toLowerCase().includes(query),
          );
        }

        lookup.innerHTML = "";
        const def = document.createElement("option");
        def.value = "";
        def.textContent = styleRows.length ? "스타일 번호를 선택하세요" : "검색 결과 없음";
        lookup.appendChild(def);

        styleRows.forEach((row) => {
          const opt = document.createElement("option");
          opt.value = row.style_no;
          opt.textContent = row.style_name ? `${row.style_no} | ${row.style_name}` : row.style_no;
          lookup.appendChild(opt);
        });

        if (currentStyleNo && styleRows.some((row) => row.style_no === currentStyleNo)) {
          lookup.value = currentStyleNo;
        }
      };

      const aggregateRows = (rows, keyColumns) => {
        const grouped = new Map();
        rows.forEach((row) => {
          const key = keyColumns.map((column) => normalize(row[column])).join("||");
          if (!grouped.has(key)) {
            const base = {};
            keyColumns.forEach((column) => {
              base[column] = selectLabel(row[column]);
            });
            base.sales_period_ty = 0;
            base.sales_period_ly = 0;
            base.sales_period_prev_week_ty = 0;
            base.qty_period_ty = 0;
            base.qty_cum_ty = 0;
            base.order_qty_total = 0;
            grouped.set(key, base);
          }

          const target = grouped.get(key);
          target.sales_period_ty += Number(row.sales_period_ty) || 0;
          target.sales_period_ly += Number(row.sales_period_ly) || 0;
          target.sales_period_prev_week_ty += Number(row.sales_period_prev_week_ty) || 0;
          target.qty_period_ty += Number(row.qty_period_ty) || 0;
          target.qty_cum_ty += Number(row.qty_cum_ty) || 0;
          const orderQty = Number(row.order_qty_total) || 0;
          if (orderQty > target.order_qty_total) {
            target.order_qty_total = orderQty;
          }
        });

        return Array.from(grouped.values()).map((row) => {
          row.yoy_sales_diff = row.sales_period_ty - row.sales_period_ly;
          row.yoy_sales_pct = row.sales_period_ly
            ? row.yoy_sales_diff / row.sales_period_ly
            : null;
          row.wow_sales_diff = row.sales_period_ty - row.sales_period_prev_week_ty;
          row.wow_sales_pct = row.sales_period_prev_week_ty
            ? row.wow_sales_diff / row.sales_period_prev_week_ty
            : null;
          row.cumulative_sell_through_ty = row.order_qty_total
            ? row.qty_cum_ty / row.order_qty_total
            : null;
          return row;
        });
      };

      const scopeSummary = () => {
        const parts = [];
        if (yearSelect.value !== "ALL") parts.push(`Year ${yearSelect.value}`);
        if (halfSelect.value !== "ALL") parts.push(`SS/FW ${halfSelect.value}`);
        if (seasonSelect.value !== "ALL") parts.push(`Season ${seasonSelect.value}`);
        return parts.length ? parts.join(" / ") : "전체";
      };

      const applyStyle = () => {
        const selectedStyleNo = normalize(lookup.value);
        const scopedRows = filterByScope(allRows);
        const scopedStyleMap = createStyleMap(scopedRows);
        const selectedStyle = scopedStyleMap.get(selectedStyleNo);
        const scopedLabel = scopeSummary();

        if (!selectedStyleNo) {
          scopeTagEl.textContent = `필터: ${scopedLabel} | 스타일 번호를 선택하면 채널/매장 베스트가 표시됩니다.`;
          renderDataTable(
            "styleNoTopChannelTable",
            [{ key: "channel", label: "채널" }],
            [],
            10,
          );
          renderDataTable(
            "styleNoTopStoreTable",
            [{ key: "store_name", label: "매장" }],
            [],
            20,
          );
          return;
        }

        scopeTagEl.textContent = selectedStyle?.style_name
          ? `${selectedStyleNo} / ${selectedStyle.style_name} (${scopedLabel})`
          : `${selectedStyleNo} (${scopedLabel})`;

        const scopedChannelRows = filterByScope(styleChannelRows).filter(
          (row) => normalize(row.style_no) === selectedStyleNo,
        );
        const scopedStoreRows = filterByScope(styleStoreRows).filter(
          (row) => normalize(row.style_no) === selectedStyleNo,
        );

        const topChannels = aggregateRows(scopedChannelRows, ["channel"])
          .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
          .slice(0, 10)
          .map((row, index) => ({ ...row, rank: index + 1 }));

        const topStores = aggregateRows(scopedStoreRows, ["store_name", "channel"])
          .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
          .slice(0, 20)
          .map((row, index) => ({ ...row, rank: index + 1 }));

        renderDataTable(
          "styleNoTopChannelTable",
          [
            { key: "rank", label: "순위" },
            { key: "channel", label: "채널" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "sales_period_prev_week_ty", label: "전주 매출", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "qty_period_ty", label: "판매수량" },
            { key: "qty_cum_ty", label: "누적판매수량" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
          ],
          topChannels,
          10,
          { sortable: true, searchable: false, paginated: false, fit: true },
        );

        renderDataTable(
          "styleNoTopStoreTable",
          [
            { key: "rank", label: "순위" },
            { key: "store_name", label: "매장" },
            { key: "channel", label: "채널" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "sales_period_prev_week_ty", label: "전주 매출", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "qty_period_ty", label: "판매수량" },
            { key: "qty_cum_ty", label: "누적판매수량" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
          ],
          topStores,
          20,
          { sortable: true, searchable: false, paginated: false, fit: true },
        );
      };

      const selectedYear = yearSelect.value || "ALL";
      const selectedHalf = halfSelect.value || "ALL";
      const selectedSeason = seasonSelect.value || "ALL";
      const selectedStyleNo = lookup.value;

      populateYears(selectedYear);
      populateHalves(selectedHalf);
      populateSeasons(selectedSeason);
      populateStyles(selectedStyleNo);

      yearSelect.onchange = () => {
        populateHalves("ALL");
        populateSeasons("ALL");
        populateStyles("");
        applyStyle();
      };
      halfSelect.onchange = () => {
        populateSeasons("ALL");
        populateStyles("");
        applyStyle();
      };
      seasonSelect.onchange = () => {
        populateStyles("");
        applyStyle();
      };
      search.oninput = () => {
        populateStyles(lookup.value);
        applyStyle();
      };
      lookup.onchange = applyStyle;
      applyStyle();
    }

    function setupInteractiveFilters() {
      const storeLookup = document.getElementById("storeLookup");
      const storeSearch = document.getElementById("storeSearch");
      const styleNoLookup = document.getElementById("styleNoLookup");
      const styleNoSearch = document.getElementById("styleNoSearch");
      const itemSalesScopeTag = document.getElementById("itemSalesScopeTag");
      const topStyleYearFilter = document.getElementById("topStyleYearFilter");
      const topStyleSeasonFilter = document.getElementById("topStyleSeasonFilter");
      const topStyleCategoryFilter = document.getElementById("topStyleCategoryFilter");
      const topStyleScopeTag = document.getElementById("topStyleScopeTag");
      const seasonFilter = document.getElementById("seasonFilter");
      const categoryFilter = document.getElementById("categoryFilter");
      const itemFilter = document.getElementById("itemFilter");
      const itemTypeFilter = document.getElementById("itemProductTypeFilter");
      const categoryStyleSearch = document.getElementById("categoryStyleSearch");
      const categoryStyleFilter = document.getElementById("categoryStyleFilter");

      if (storeLookup) {
        storeLookup.innerHTML = "";
        const d = document.createElement("option");
        d.value = "";
        d.textContent = "매장을 선택하세요";
        storeLookup.appendChild(d);
      }

      if (storeSearch) {
        storeSearch.value = "";
      }

      if (styleNoLookup) {
        styleNoLookup.innerHTML = "";
        const d = document.createElement("option");
        d.value = "";
        d.textContent = "스타일 번호를 선택하세요";
        styleNoLookup.appendChild(d);
      }

      if (styleNoSearch) {
        styleNoSearch.value = "";
      }

      if (itemSalesScopeTag) {
        itemSalesScopeTag.textContent = "전체 매출 대비 연도/시즌 비중";
      }

      if (topStyleYearFilter) {
        topStyleYearFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        topStyleYearFilter.appendChild(d);
      }

      if (topStyleSeasonFilter) {
        topStyleSeasonFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        topStyleSeasonFilter.appendChild(d);
      }

      if (topStyleCategoryFilter) {
        topStyleCategoryFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        topStyleCategoryFilter.appendChild(d);
      }

      if (topStyleScopeTag) {
        topStyleScopeTag.textContent = "스타일 필터: 전체";
      }

      if (seasonFilter) {
        seasonFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        seasonFilter.appendChild(d);
      }

      if (categoryFilter) {
        categoryFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        categoryFilter.appendChild(d);
      }

      if (itemFilter) {
        itemFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        itemFilter.appendChild(d);
      }

      if (itemTypeFilter) {
        itemTypeFilter.value = itemTypeFilter.value || "ALL";
      }

      if (categoryStyleSearch) {
        categoryStyleSearch.value = "";
        categoryStyleSearch.placeholder = "아이템 선택 후 스타일 검색";
        categoryStyleSearch.disabled = true;
      }

      if (categoryStyleFilter) {
        categoryStyleFilter.innerHTML = "";
        const d = document.createElement("option");
        d.value = "ALL";
        d.textContent = "전체";
        categoryStyleFilter.appendChild(d);
        categoryStyleFilter.disabled = true;
      }
    }

    function renderStoreDeepDive(currentBrand) {
      const rows = (dataset.store_deep_dive || []).filter((row) => !currentBrand || normalizeBrand(row.brand) === currentBrand);
      const lookup = document.getElementById("storeLookup");
      const storeSearch = document.getElementById("storeSearch");
      if (!lookup) return;

      const makeStoreKey = (row) => {
        const code = String(row.store_code || "").trim();
        if (code) return code;
        return `${row.store_name || ""}__${row.channel || ""}`;
      };

      const uniqueStores = [];
      const seenStoreKeys = new Set();
      rows.forEach((row) => {
        const key = makeStoreKey(row);
        if (seenStoreKeys.has(key)) return;
        seenStoreKeys.add(key);
        uniqueStores.push(row);
      });

      const selected = lookup.value;
      const query = (storeSearch ? storeSearch.value : "").trim().toLowerCase();
      const optionRows = query
        ? uniqueStores.filter((row) => `${row.store_name || ""} ${row.channel || ""}`.toLowerCase().includes(query))
        : uniqueStores;

      lookup.innerHTML = "";
      const def = document.createElement("option");
      def.value = "";
      def.textContent = optionRows.length ? "매장을 선택하세요" : "검색 결과 없음";
      lookup.appendChild(def);

      optionRows.forEach((row) => {
        const opt = document.createElement("option");
        opt.value = makeStoreKey(row);
        opt.textContent = `${row.store_name} (${row.channel})`;
        lookup.appendChild(opt);
      });

      if (selected && optionRows.some((row) => makeStoreKey(row) === selected)) {
        lookup.value = selected;
      }

      const findStoreByKey = (key) => uniqueStores.find((row) => makeStoreKey(row) === key);

      const applyStore = () => {
        const storeRow = findStoreByKey(lookup.value);
        if (!storeRow) {
          setText("storeScopeTag", "매장을 선택하면 상세 분석이 표시됩니다.");
          ["storeWeeklySales", "storeWeeklyGrowth", "storeYoYGrowth", "storeYTDGrowth", "storeAvgDisc", "storeMarginRate"].forEach((id) => setText(id, "-"));
          renderDataTable("storeTopStylesTable", [{ key: "style_name", label: "스타일" }], [], 10);
          upsertChart("chartStoreCategoryMix", {
            type: "doughnut",
            data: { labels: ["데이터 없음"], datasets: [{ data: [1], backgroundColor: ["#e2e8f0"] }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "bottom" } } },
          });
          upsertChart("chartStoreTopStyles", {
            type: "bar",
            data: { labels: [], datasets: [{ data: [] }] },
            options: { responsive: true, maintainAspectRatio: false },
          });
          return;
        }

        setText("storeScopeTag", `${storeRow.store_name} / ${storeRow.channel}`);
        setText("storeWeeklySales", fmtShortAmount(storeRow.sales_period_ty));
        setText("storeWeeklyGrowth", fmtPct(storeRow.wow_sales_pct));
        setText("storeYoYGrowth", fmtPct(storeRow.yoy_sales_pct));
        setText("storeYTDGrowth", fmtPct(storeRow.ytd_yoy_pct));
        setText("storeAvgDisc", fmtPct(storeRow.discount_rate_ty));
        setText("storeMarginRate", fmtPct(storeRow.gross_margin_rate_ty));

        const styleRows = (dataset.store_style || []).filter((r) => r.store_name === storeRow.store_name && (!currentBrand || normalizeBrand(r.brand) === currentBrand));
        const topStyles = [...styleRows]
          .sort((a, b) => (b.sales_period_ty || 0) - (a.sales_period_ty || 0))
          .slice(0, 10)
          .map((r, i) => ({ ...r, rank: i + 1 }));

        renderDataTable(
          "storeTopStylesTable",
          [
            { key: "rank", label: "순위" },
            { key: "style_name", label: "스타일" },
            { key: "style_no", label: "코드" },
            { key: "sales_period_ty", label: "당주 매출" },
            { key: "sales_qty_period_ty", label: "수량" },
          ],
          topStyles,
          10,
        );

        const categoryRows = (dataset.store_category_mix || []).filter((r) => r.store_name === storeRow.store_name && (!currentBrand || normalizeBrand(r.brand) === currentBrand));

        const topCategory = [...categoryRows]
          .sort((a, b) => (b.sales_period_ty || 0) - (a.sales_period_ty || 0))
          .slice(0, 8);

        upsertChart("chartStoreCategoryMix", {
          type: "doughnut",
          data: {
            labels: topCategory.map((r) => r.category),
            datasets: [
              {
                data: topCategory.map((r) => Number(r.sales_period_ty) || 0),
                backgroundColor: ["#0d6e6e", "#2a9d8f", "#8ecae6", "#ffd166", "#ef476f", "#118ab2", "#6a4c93", "#8d99ae"],
              },
            ],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: { position: "bottom", labels: { boxWidth: 10 } },
              tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${fmtShortAmount(ctx.raw)}` } },
            },
          },
        });

        const styleLabels = topStyles.map((r) => r.style_name || r.style_no);
        const styleValues = topStyles.map((r) => Number(r.sales_period_ty) || 0);
        upsertChart("chartStoreTopStyles", {
          type: "bar",
          data: {
            labels: styleLabels,
            datasets: [{
              label: "Sales TY",
              data: styleValues,
              backgroundColor: "#1f8f9f",
            }],
          },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            plugins: {
              legend: { display: false },
              tooltip: { callbacks: { label: (ctx) => `매출: ${fmtShortAmount(ctx.raw)}` } },
            },
            scales: { x: { ticks: { callback: (v) => fmtShortAmount(v) } } },
          },
        });
      };

      lookup.onchange = applyStore;
      if (storeSearch) {
        storeSearch.oninput = () => {
          renderStoreDeepDive(currentBrand);
        };
      }
      applyStore();
    }

    function renderProfitability(currentBrand, selectedSeason = "ALL") {
      const normalize = (value) => String(value || "").trim();
      const seasonKey = normalize(selectedSeason) || "ALL";

      const scopedStyleRows = (dataset.profitability_style_season || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const scopedCategoryRows = (dataset.profitability_category_season || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const baseStyleRows = (dataset.profitability_style || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );
      const baseCategoryRows = (dataset.profitability_category || []).filter(
        (row) => !currentBrand || normalizeBrand(row.brand) === currentBrand,
      );

      const styleRows =
        seasonKey !== "ALL" && scopedStyleRows.length
          ? scopedStyleRows.filter((row) => normalize(row.year_season) === seasonKey)
          : baseStyleRows;
      const categoryRows =
        seasonKey !== "ALL" && scopedCategoryRows.length
          ? scopedCategoryRows.filter((row) => normalize(row.year_season) === seasonKey)
          : baseCategoryRows;

      const topStyles = [...styleRows]
        .sort((a, b) => (Number(b.gross_margin_amt_ty) || 0) - (Number(a.gross_margin_amt_ty) || 0))
        .slice(0, 12);
      upsertChart("chartProfitTopStyles", {
        type: "bar",
        data: {
          labels: topStyles.map((r) => r.style_name || r.style_no || "Unknown"),
          datasets: [{
            label: "마진액(백만원)",
            data: topStyles.map((r) => (Number(r.gross_margin_amt_ty) || 0) / 1000000),
            backgroundColor: "#0d6e6e",
            borderRadius: 6,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          indexAxis: "y",
          plugins: {
            legend: { display: false },
            tooltip: { callbacks: { label: (ctx) => `마진액: ${ctx.raw.toLocaleString("ko-KR", { minimumFractionDigits: 1, maximumFractionDigits: 1 })} 백만원` } },
          },
          scales: { x: { beginAtZero: true, ticks: { callback: (v) => `${v} 백만원` } } },
        },
      });

      const topCategories = [...categoryRows]
        .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
        .slice(0, 12);
      upsertChart("chartProfitCategoryRate", {
        type: "bar",
        data: {
          labels: topCategories.map((r) => {
            const brand = normalize(r.brand);
            const category = normalize(r.category);
            if (brand && category) return `${brand} | ${category}`;
            return brand || category || "Unknown";
          }),
          datasets: [
            {
              label: "마진율(%)",
              data: topCategories.map((r) => (Number(r.gross_margin_rate_ty) || 0) * 100),
              backgroundColor: "#2a9d8f",
              borderRadius: 6,
            },
            {
              label: "할인율(%)",
              data: topCategories.map((r) => (Number(r.discount_rate_ty) || 0) * 100),
              backgroundColor: "#e9c46a",
              borderRadius: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.raw.toFixed(1)}%` } },
          },
          scales: { y: { beginAtZero: true, ticks: { callback: (v) => `${v}%` } } },
        },
      });

      const scatterRows = [...styleRows]
        .filter((r) => Number(r.sales_period_ty) > 0)
        .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
        .slice(0, 60);
      upsertChart("chartProfitScatter", {
        type: "bubble",
        data: {
          datasets: [
            {
              label: "Styles",
              data: scatterRows.map((r) => ({
                x: (Number(r.discount_rate_ty) || 0) * 100,
                y: (Number(r.gross_margin_rate_ty) || 0) * 100,
                r: Math.max(4, Math.min(20, Math.sqrt((Number(r.sales_period_ty) || 0) / 1000000) * 2.2)),
                style: r.style_name || r.style_no || "Unknown",
                salesM: (Number(r.sales_period_ty) || 0) / 1000000,
              })),
              backgroundColor: "rgba(13, 110, 110, 0.45)",
              borderColor: "rgba(13, 110, 110, 0.9)",
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const p = ctx.raw;
                  return `${p.style}: 매출 ${p.salesM.toFixed(1)}백만원 / 할인율 ${p.x.toFixed(1)}% / 마진율 ${p.y.toFixed(1)}%`;
                },
              },
            },
          },
          scales: {
            x: { title: { display: true, text: "할인율 (%)" }, ticks: { callback: (v) => `${v}%` } },
            y: { title: { display: true, text: "마진율 (%)" }, ticks: { callback: (v) => `${v}%` } },
          },
        },
      });
    }

    function renderSeasonSection(currentBrand, onSeasonChanged) {
      const seasonSelect = document.getElementById("seasonFilter");
      const rows = (dataset.season || []).filter((row) => !currentBrand || normalizeBrand(row.brand) === currentBrand);
      if (!seasonSelect) return;

      const selected = seasonSelect.value || "ALL";
      seasonSelect.innerHTML = "";
      const allOpt = document.createElement("option");
      allOpt.value = "ALL";
      allOpt.textContent = "전체";
      seasonSelect.appendChild(allOpt);

      const seasons = Array.from(new Set(rows.map((r) => r.year_season).filter(Boolean))).sort();
      seasons.forEach((s) => {
        const opt = document.createElement("option");
        opt.value = s;
        opt.textContent = s;
        seasonSelect.appendChild(opt);
      });
      seasonSelect.value = seasons.includes(selected) ? selected : "ALL";

      const draw = () => {
        const value = seasonSelect.value;
        const filtered = value === "ALL" ? rows : rows.filter((r) => r.year_season === value);
        renderDataTable(
          "seasonTable",
          [
            { key: "year_season", label: "시즌" },
            { key: "season_half", label: "SS/FW" },
            { key: "season", label: "Season" },
            { key: "new_old", label: "신/이월" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "sales_period_ly", label: "전년", format: "million" },
            { key: "sales_cum_ty", label: "기간 누적 TY", format: "million" },
            { key: "sales_cum_ly", label: "전년 누적 LY", format: "million" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
            { key: "yoy_sales_pct", label: "YoY" },
            { key: "sales_period_prev_week_ty", label: "전주", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "discount_rate_ty", label: "할인율", format: "pct" },
          ],
          filtered,
          filtered.length,
          { sortable: true, searchable: true, paginated: true, pageSize: 12 },
        );

        if (typeof onSeasonChanged === "function") {
          onSeasonChanged(value || "ALL");
        }
      };

      seasonSelect.onchange = draw;
      draw();
      return seasonSelect.value || "ALL";
    }

    function renderCategoryDetailSection(currentBrand) {
      const categorySelect = document.getElementById("categoryFilter");
      const itemSelect = document.getElementById("itemFilter");
      const styleSearch = document.getElementById("categoryStyleSearch");
      const styleSelect = document.getElementById("categoryStyleFilter");
      const bestStoreHint = document.getElementById("categoryItemBestStoreHint");
      const bestStoreTable = document.getElementById("categoryItemBestStoreTable");
      const allRows = (dataset.category_detail || []).filter((row) => !currentBrand || normalizeBrand(row.brand) === currentBrand);
      const itemStoreRows = (dataset.item_store_period || []).filter((row) => !currentBrand || normalizeBrand(row.brand) === currentBrand);
      const styleStoreRows = (dataset.style_store_period || []).filter((row) => !currentBrand || normalizeBrand(row.brand) === currentBrand);
      if (!categorySelect || !itemSelect || !styleSearch || !styleSelect || !bestStoreHint || !bestStoreTable) return;

      const normalize = (value) => String(value || "").trim();
      const styleKeyOf = (row) => {
        const styleNo = normalize(row.style_no);
        const styleName = normalize(row.style_name);
        if (styleNo) return `NO:${styleNo}`;
        if (styleName) return `NM:${styleName}`;
        return "";
      };

      const styleLabelOf = (row) => {
        const styleNo = normalize(row.style_no);
        const styleName = normalize(row.style_name);
        if (styleNo && styleName) return `${styleNo} | ${styleName}`;
        return styleNo || styleName || "(스타일 미지정)";
      };

      const selectedCategory = categorySelect.value || "ALL";
      const selectedItem = itemSelect.value || "ALL";
      const selectedStyle = styleSelect.value || "ALL";

      const populateCategories = (preferredCategory) => {
        const currentCategory = preferredCategory || categorySelect.value || "ALL";
        categorySelect.innerHTML = "";
        const allC = document.createElement("option");
        allC.value = "ALL";
        allC.textContent = "전체";
        categorySelect.appendChild(allC);

        const categories = Array.from(new Set(allRows.map((r) => r.category).filter(Boolean))).sort();
        categories.forEach((c) => {
          const opt = document.createElement("option");
          opt.value = c;
          opt.textContent = c;
          categorySelect.appendChild(opt);
        });
        categorySelect.value = categories.includes(currentCategory) ? currentCategory : "ALL";
      };

      const populateItems = (preferredItem) => {
        const currentItem = preferredItem || itemSelect.value || "ALL";
        const cat = categorySelect.value;
        const subset = cat === "ALL" ? allRows : allRows.filter((r) => r.category === cat);
        itemSelect.innerHTML = "";
        const allI = document.createElement("option");
        allI.value = "ALL";
        allI.textContent = "전체";
        itemSelect.appendChild(allI);
        const items = Array.from(new Set(subset.map((r) => r.item).filter(Boolean))).sort();
        items.forEach((i) => {
          const opt = document.createElement("option");
          opt.value = i;
          opt.textContent = i;
          itemSelect.appendChild(opt);
        });
        itemSelect.value = items.includes(currentItem) ? currentItem : "ALL";
      };

      const populateStyles = (preferredStyle) => {
        const currentStyle = preferredStyle || styleSelect.value || "ALL";
        const cat = categorySelect.value;
        const item = itemSelect.value;
        const query = (styleSearch.value || "").trim().toLowerCase();

        styleSelect.innerHTML = "";
        const allS = document.createElement("option");
        allS.value = "ALL";
        allS.textContent = "전체";
        styleSelect.appendChild(allS);

        if (item === "ALL") {
          styleSearch.value = "";
          styleSearch.placeholder = "아이템 선택 후 스타일 검색";
          styleSearch.disabled = true;
          styleSelect.disabled = true;
          styleSelect.value = "ALL";
          return;
        }

        styleSearch.placeholder = "스타일코드/스타일명 검색";
        styleSearch.disabled = false;
        styleSelect.disabled = false;

        let subset = allRows.filter((r) => r.item === item);
        if (cat !== "ALL") {
          subset = subset.filter((r) => r.category === cat);
        }

        const styleMap = new Map();
        subset.forEach((row) => {
          const key = styleKeyOf(row);
          if (!key) return;
          if (!styleMap.has(key)) {
            styleMap.set(key, styleLabelOf(row));
          }
        });

        let styleOptions = Array.from(styleMap.entries()).map(([value, label]) => ({
          value,
          label,
        }));

        if (query) {
          styleOptions = styleOptions.filter((opt) =>
            opt.label.toLowerCase().includes(query),
          );
        }

        styleOptions.sort((a, b) => a.label.localeCompare(b.label));
        styleOptions.forEach((optRow) => {
          const opt = document.createElement("option");
          opt.value = optRow.value;
          opt.textContent = optRow.label;
          styleSelect.appendChild(opt);
        });

        if (!styleOptions.length) {
          allS.textContent = "검색 결과 없음";
        }

        styleSelect.value = styleOptions.some((opt) => opt.value === currentStyle)
          ? currentStyle
          : "ALL";
      };

      const draw = () => {
        const cat = categorySelect.value;
        const item = itemSelect.value;
        const style = styleSelect.value;
        let filtered = allRows;
        if (cat !== "ALL") filtered = filtered.filter((r) => r.category === cat);
        if (item !== "ALL") filtered = filtered.filter((r) => r.item === item);
        if (style !== "ALL") filtered = filtered.filter((r) => styleKeyOf(r) === style);

        renderDataTable(
          "categoryDetailTable",
          [
            { key: "category", label: "카테고리" },
            { key: "item", label: "아이템" },
            { key: "style_no", label: "스타일코드" },
            { key: "style_name", label: "스타일명" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "sales_period_ly", label: "전년", format: "million" },
            { key: "sales_cum_ty", label: "기간 누적 TY", format: "million" },
            { key: "sales_cum_ly", label: "전년 누적 LY", format: "million" },
            { key: "cumulative_sell_through_ty", label: "누적판매율", format: "pct_plain" },
            { key: "yoy_sales_pct", label: "YoY" },
            { key: "sales_period_prev_week_ty", label: "전주", format: "million" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "discount_rate_ty", label: "할인율", format: "pct" },
            { key: "discount_rate_wow_p", label: "할인율 WoW", format: "pct" },
            { key: "discount_rate_yoy_p", label: "할인율 YoY", format: "pct" },
          ],
          filtered,
          100,
          { sortable: true, searchable: true, paginated: true, pageSize: 12 },
        );

        if (item === "ALL") {
          bestStoreHint.textContent = "아이템을 선택하면 판매 베스트 매장이 표시됩니다.";
          bestStoreHint.style.display = "block";
          bestStoreTable.innerHTML = "";
          bestStoreTable.style.display = "none";
          return;
        }

        if (style !== "ALL") {
          const selectedStyleLabel = styleSelect.options[styleSelect.selectedIndex]?.textContent || style;
          const rankedStyleStores = [...styleStoreRows]
            .filter((row) => styleKeyOf(row) === style)
            .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
            .slice(0, 20)
            .map((row, idx) => ({ ...row, rank: idx + 1 }));

          bestStoreHint.textContent = `선택 스타일: ${selectedStyleLabel} / 상위 ${rankedStyleStores.length}개 매장`;
          bestStoreHint.style.display = "block";
          bestStoreTable.style.display = "block";
          renderDataTable(
            "categoryItemBestStoreTable",
            [
              { key: "rank", label: "순위" },
              { key: "store_name", label: "매장" },
              { key: "channel", label: "채널" },
              { key: "style_no", label: "스타일코드" },
              { key: "style_name", label: "스타일명" },
              { key: "sales_period_ty", label: "당주 매출", format: "million" },
              { key: "sales_period_ly", label: "전년", format: "million" },
              { key: "yoy_sales_pct", label: "YoY" },
              { key: "wow_sales_pct", label: "WoW" },
              { key: "qty_period_ty", label: "판매수량" },
            ],
            rankedStyleStores,
            20,
            { sortable: true, searchable: false, paginated: false, fit: true },
          );
          return;
        }

        let itemStoreFiltered = itemStoreRows.filter((row) => row.item === item);
        if (cat !== "ALL") {
          itemStoreFiltered = itemStoreFiltered.filter((row) => row.category === cat);
        }

        const rankedStores = [...itemStoreFiltered]
          .sort((a, b) => (Number(b.sales_period_ty) || 0) - (Number(a.sales_period_ty) || 0))
          .slice(0, 20)
          .map((row, idx) => ({ ...row, rank: idx + 1 }));

        bestStoreHint.textContent = `선택 아이템: ${item} / 상위 ${rankedStores.length}개 매장`;
        bestStoreHint.style.display = "block";
        bestStoreTable.style.display = "block";
        renderDataTable(
          "categoryItemBestStoreTable",
          [
            { key: "rank", label: "순위" },
            { key: "store_name", label: "매장" },
            { key: "channel", label: "채널" },
            { key: "category", label: "카테고리" },
            { key: "sales_period_ty", label: "당주 매출", format: "million" },
            { key: "sales_period_ly", label: "전년", format: "million" },
            { key: "yoy_sales_pct", label: "YoY" },
            { key: "wow_sales_pct", label: "WoW" },
            { key: "qty_period_ty", label: "판매수량" },
          ],
          rankedStores,
          20,
          { sortable: true, searchable: false, paginated: false, fit: true },
        );
      };

      populateCategories(selectedCategory);
      populateItems(selectedItem);
      populateStyles(selectedStyle);
      categorySelect.onchange = () => {
        populateItems();
        styleSearch.value = "";
        populateStyles();
        draw();
      };
      itemSelect.onchange = () => {
        styleSearch.value = "";
        populateStyles();
        draw();
      };
      styleSearch.oninput = () => {
        populateStyles(styleSelect.value || "ALL");
        draw();
      };
      styleSelect.onchange = draw;
      draw();
    }

    function updateDashboard() {
      const selected = brandFilter.value || "ALL";
      const scope = selected === "ALL" ? "전체" : selected;
      scopeTag.textContent = `현재 범위: ${scope}`;
      const currentBrand = selected === "ALL" ? "" : normalizeBrand(selected);

      const brandRows = filteredRows(dataset.brand);
      const channelRows = filteredRows(dataset.channel);
      const categoryRows = filteredRows(dataset.category);
      const itemRows = filteredRows(dataset.item);
      const itemScopeRows = filteredRows(dataset.item_scope || []);
      const styleRows = filteredRows(dataset.style || []);
      const styleScopeRows = filteredRows(dataset.style_scope || []);
      const storeRows = filteredRows(dataset.store);
      const segmentRows = filteredRows(dataset.segment_mix || []);

      recordsTag.textContent = `Brand ${brandRows.length} / Channel ${channelRows.length} / Category ${categoryRows.length} / Style ${styleRows.length} / Store ${storeRows.length}`;

      updateNumericPanels(brandRows, channelRows, categoryRows, itemRows, storeRows);
      renderChannelMixCharts(channelRows);
      renderDimensionMixChart(categoryRows, "category", "chartCategoryMixCompare", 12);
      renderSegmentMixAndItemYoY(segmentRows, itemRows, itemScopeRows);
      renderTopInsights(
        storeRows,
        styleRows,
        styleScopeRows,
        () => renderSegmentMixAndItemYoY(segmentRows, itemRows, itemScopeRows),
      );
      renderItemSalesShareSection(currentBrand);
      renderStyleNoDrilldown(currentBrand);
      renderStoreDeepDive(currentBrand);
      renderSeasonSection(currentBrand, (seasonValue) => {
        renderProfitability(currentBrand, seasonValue);
      });
      renderCategoryDetailSection(currentBrand);
    }

    setupInteractiveFilters();
    setupBrandFilter();
    brandFilter.addEventListener("change", updateDashboard);
    if (itemProductTypeFilter) {
      itemProductTypeFilter.addEventListener("change", updateDashboard);
    }
    if (excelUploadInput) {
      excelUploadInput.addEventListener("change", handleExcelUploadChange);
    }
    if (resetUploadBtn) {
      resetUploadBtn.addEventListener("click", resetToDefaultDataset);
    }
    updateDashboard();
  </script>
</body>
</html>
"""

    html = (
        template.replace("__GENERATED_AT__", generated_at)
        .replace("__CARDS_HTML__", cards_html)
        .replace("__DATASET_JSON__", dataset_json)
    )
    path.write_text(html, encoding="utf-8")


def generate_mvp_outputs(latest_root: Path, output_dir: Path, top_n: int = 20) -> dict:
    marts_dir = latest_root / "marts"
    facts_dir = latest_root / "facts"
    output_dir.mkdir(parents=True, exist_ok=True)

    period_brand = pd.read_csv(marts_dir / "compare_period_brand.csv")
    period_channel = pd.read_csv(marts_dir / "compare_period_channel.csv")
    period_category = pd.read_csv(marts_dir / "compare_period_category.csv")
    period_item = pd.read_csv(marts_dir / "compare_period_item.csv")
    period_store = pd.read_csv(marts_dir / "compare_period_store.csv")

    for frame in [
        period_brand,
        period_channel,
        period_category,
        period_item,
        period_store,
    ]:
        for column in DECISION_COLUMNS:
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")

    brand_decisions = enrich_decision_table(period_brand, ["brand"])
    channel_decisions = enrich_decision_table(
        period_channel, ["brand", "channel", "channel_type"]
    )
    category_decisions = enrich_decision_table(period_category, ["brand", "category"])
    item_decisions = enrich_decision_table(period_item, ["brand", "item", "category"])
    store_decisions = enrich_decision_table(
        period_store, ["brand", "store_code", "store_name", "channel"]
    )

    fact = pd.read_csv(facts_dir / "sales_fact_all.csv", low_memory=False)
    for col in [
        "sales_amt_net_v_excl",
        "sales_amt_tag_v_excl",
        "sales_qty",
        "cost_amt_v_incl",
    ]:
        if col in fact.columns:
            fact[col] = pd.to_numeric(fact[col], errors="coerce")

    style_period = build_style_period_compare(fact)
    style_scope_period = build_style_period_compare(
        fact,
        keys=["brand", "year", "season", "style_no", "style_name", "category"],
    )

    style_type_lookup = (
        fact[["brand", "style_no", "category", "item"]]
        .dropna(subset=["brand", "style_no", "category"])
        .copy()
    )
    style_type_lookup["product_type"] = style_type_lookup.apply(
        lambda row: classify_product_type(row.get("item"), row.get("category")),
        axis=1,
    )
    style_type_lookup = (
        style_type_lookup.groupby(["brand", "style_no", "category"], dropna=False)[
            "product_type"
        ]
        .agg(lambda values: "용품" if "용품" in set(values) else "의류")
        .reset_index()
    )

    style_period = style_period.merge(
        style_type_lookup,
        on=["brand", "style_no", "category"],
        how="left",
    )
    style_scope_period = style_scope_period.merge(
        style_type_lookup,
        on=["brand", "style_no", "category"],
        how="left",
    )
    style_period["product_type"] = style_period["product_type"].fillna(
        style_period["category"].apply(
            lambda category: classify_product_type("", category)
        )
    )
    style_scope_period["product_type"] = style_scope_period["product_type"].fillna(
        style_scope_period["category"].apply(
            lambda category: classify_product_type("", category)
        )
    )

    item_scope_period = build_item_scope_period(fact)
    season_period = build_season_period_compare(fact)
    category_detail_period = build_category_detail_period(fact)
    item_store_period = build_item_store_period(fact)
    style_channel_period, style_store_period = build_style_channel_store_period(fact)
    segment_mix, product_mix = build_category_mix_breakdowns(fact)
    store_deep_dive = pd.DataFrame(build_store_deep_dive_records(fact))

    store_style_period = (
        fact[fact["timeframe_code"] == "period_ty"]
        .groupby(
            ["brand", "store_name", "style_no", "style_name", "category"], dropna=False
        )[["sales_amt_net_v_excl", "sales_qty"]]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                "sales_amt_net_v_excl": "sales_period_ty",
                "sales_qty": "sales_qty_period_ty",
            }
        )
    )

    store_category_mix = (
        fact[fact["timeframe_code"].isin(["period_ty", "period_ly"])]
        .groupby(["brand", "store_name", "category", "timeframe_code"], dropna=False)[
            "sales_amt_net_v_excl"
        ]
        .sum(min_count=1)
        .reset_index()
        .pivot_table(
            index=["brand", "store_name", "category"],
            columns="timeframe_code",
            values="sales_amt_net_v_excl",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )
    if "period_ty" not in store_category_mix.columns:
        store_category_mix["period_ty"] = 0
    if "period_ly" not in store_category_mix.columns:
        store_category_mix["period_ly"] = 0
    store_category_mix = store_category_mix.rename(
        columns={"period_ty": "sales_period_ty", "period_ly": "sales_period_ly"}
    )

    profitability_style = style_period[
        [
            "brand",
            "style_no",
            "style_name",
            "category",
            "sales_period_ty",
            "gross_margin_amt_ty",
            "gross_margin_rate_ty",
            "discount_rate_ty",
        ]
    ].copy()
    profitability_style = profitability_style.sort_values(
        "gross_margin_amt_ty", ascending=False
    ).reset_index(drop=True)
    profitability_style["rank"] = profitability_style.index + 1

    profitability_style_season = (
        fact[fact["timeframe_code"] == "period_ty"]
        .groupby(
            ["brand", "year_season", "style_no", "style_name", "category"],
            dropna=False,
        )[["sales_amt_net_v_excl", "cost_amt_v_incl", "sales_amt_tag_v_excl"]]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"sales_amt_net_v_excl": "sales_period_ty"})
    )
    profitability_style_season["gross_margin_amt_ty"] = (
        profitability_style_season["sales_period_ty"]
        - profitability_style_season["cost_amt_v_incl"]
    )
    profitability_style_season["gross_margin_rate_ty"] = _safe_ratio(
        profitability_style_season["gross_margin_amt_ty"],
        profitability_style_season["sales_period_ty"],
    )
    profitability_style_season["discount_rate_ty"] = 1 - _safe_ratio(
        profitability_style_season["sales_period_ty"],
        profitability_style_season["sales_amt_tag_v_excl"],
    )

    profitability_category = (
        fact[fact["timeframe_code"] == "period_ty"]
        .groupby(["brand", "category"], dropna=False)[
            ["sales_amt_net_v_excl", "cost_amt_v_incl", "sales_amt_tag_v_excl"]
        ]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"sales_amt_net_v_excl": "sales_period_ty"})
    )
    profitability_category["gross_margin_amt_ty"] = (
        profitability_category["sales_period_ty"]
        - profitability_category["cost_amt_v_incl"]
    )
    profitability_category["gross_margin_rate_ty"] = _safe_ratio(
        profitability_category["gross_margin_amt_ty"],
        profitability_category["sales_period_ty"],
    )
    profitability_category["discount_rate_ty"] = 1 - _safe_ratio(
        profitability_category["sales_period_ty"],
        profitability_category["sales_amt_tag_v_excl"],
    )

    profitability_category_season = (
        fact[fact["timeframe_code"] == "period_ty"]
        .groupby(["brand", "year_season", "category"], dropna=False)[
            ["sales_amt_net_v_excl", "cost_amt_v_incl", "sales_amt_tag_v_excl"]
        ]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={"sales_amt_net_v_excl": "sales_period_ty"})
    )
    profitability_category_season["gross_margin_amt_ty"] = (
        profitability_category_season["sales_period_ty"]
        - profitability_category_season["cost_amt_v_incl"]
    )
    profitability_category_season["gross_margin_rate_ty"] = _safe_ratio(
        profitability_category_season["gross_margin_amt_ty"],
        profitability_category_season["sales_period_ty"],
    )
    profitability_category_season["discount_rate_ty"] = 1 - _safe_ratio(
        profitability_category_season["sales_period_ty"],
        profitability_category_season["sales_amt_tag_v_excl"],
    )

    sales_period_ty = float(brand_decisions["sales_period_ty"].sum())
    sales_period_ly = float(brand_decisions["sales_period_ly"].sum())
    sales_period_prev_week_ty = float(
        brand_decisions["sales_period_prev_week_ty"].sum()
    )

    yoy_sales_diff = sales_period_ty - sales_period_ly
    wow_sales_diff = sales_period_ty - sales_period_prev_week_ty

    kpi: dict[str, float | None] = {
        "sales_period_ty": sales_period_ty,
        "sales_period_ly": sales_period_ly,
        "sales_period_prev_week_ty": sales_period_prev_week_ty,
        "yoy_sales_diff": yoy_sales_diff,
        "wow_sales_diff": wow_sales_diff,
        "yoy_sales_pct": safe_pct(yoy_sales_diff, sales_period_ly),
        "wow_sales_pct": safe_pct(wow_sales_diff, sales_period_prev_week_ty),
    }

    brand_opp, brand_risk = find_opportunity_and_risk(brand_decisions, ["brand"])
    channel_opp, channel_risk = find_opportunity_and_risk(
        channel_decisions, ["brand", "channel"]
    )
    category_opp, category_risk = find_opportunity_and_risk(
        category_decisions, ["brand", "category"]
    )
    item_opp, item_risk = find_opportunity_and_risk(item_decisions, ["brand", "item"])
    store_opp, store_risk = find_opportunity_and_risk(
        store_decisions, ["brand", "store_name"]
    )

    roadmap = pd.DataFrame(
        [
            {
                "step": 1,
                "focus": "브랜드 포트폴리오",
                "objective": "브랜드별 예산/재고 우선순위 결정",
                "opportunity_target": brand_opp,
                "risk_target": brand_risk,
                "recommended_action": "상위 성장 브랜드 재고/노출 확대, 하락 브랜드 긴급 원인 점검",
            },
            {
                "step": 2,
                "focus": "채널 운영",
                "objective": "채널별 매출 효율 최적화",
                "opportunity_target": channel_opp,
                "risk_target": channel_risk,
                "recommended_action": "성장 채널 광고/물량 확대, 하락 채널 프로모션 조건 재설계",
            },
            {
                "step": 3,
                "focus": "카테고리 믹스",
                "objective": "카테고리별 매출 비중 재배치",
                "opportunity_target": category_opp,
                "risk_target": category_risk,
                "recommended_action": "고성장 카테고리 확장, 저성과 카테고리 가격/진열 정책 수정",
            },
            {
                "step": 4,
                "focus": "아이템 액션",
                "objective": "아이템 단위 리오더/클리어런스 판단",
                "opportunity_target": item_opp,
                "risk_target": item_risk,
                "recommended_action": "고성장 SKU 리오더 검토, 하락 SKU 판매정책 즉시 조정",
            },
            {
                "step": 5,
                "focus": "매장 실행",
                "objective": "매장별 실행안(인력, 재고, 행사) 확정",
                "opportunity_target": store_opp,
                "risk_target": store_risk,
                "recommended_action": "상승 매장 성공요인 확산, 하락 매장 매대/사이즈/프로모션 긴급 보정",
            },
        ]
    )

    generated_at = datetime.now().isoformat(timespec="seconds")

    kpi_df = pd.DataFrame([kpi])
    kpi_file = output_dir / "kpi_summary.csv"
    brand_file = output_dir / "brand_decisions.csv"
    channel_file = output_dir / "channel_decisions.csv"
    category_file = output_dir / "category_decisions.csv"
    item_file = output_dir / "item_decisions.csv"
    store_file = output_dir / "store_decisions.csv"
    roadmap_file = output_dir / "decision_roadmap.csv"

    kpi_df.to_csv(kpi_file, index=False, encoding="utf-8")
    brand_decisions.to_csv(brand_file, index=False, encoding="utf-8")
    channel_decisions.to_csv(channel_file, index=False, encoding="utf-8")
    category_decisions.to_csv(category_file, index=False, encoding="utf-8")
    item_decisions.head(top_n).to_csv(item_file, index=False, encoding="utf-8")
    store_decisions.head(top_n).to_csv(store_file, index=False, encoding="utf-8")
    roadmap.to_csv(roadmap_file, index=False, encoding="utf-8")

    excel_file = output_dir / "weekly_sales_mvp_dashboard.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        kpi_df.to_excel(writer, sheet_name="KPI", index=False)
        roadmap.to_excel(writer, sheet_name="Roadmap", index=False)
        brand_decisions.to_excel(writer, sheet_name="Brand", index=False)
        channel_decisions.head(top_n).to_excel(
            writer, sheet_name="Channel", index=False
        )
        category_decisions.head(top_n).to_excel(
            writer, sheet_name="Category", index=False
        )
        item_decisions.head(top_n).to_excel(writer, sheet_name="Item", index=False)
        store_decisions.head(top_n).to_excel(writer, sheet_name="Store", index=False)

    markdown_file = output_dir / "weekly_sales_mvp_brief.md"
    write_markdown_brief(
        path=markdown_file,
        generated_at=generated_at,
        kpi=kpi,
        roadmap=roadmap,
        brand_table=brand_decisions,
        channel_table=channel_decisions,
    )

    style_channel_payload = _cap_rows_per_brand(style_channel_period, limit=2500)
    style_store_payload = _cap_rows_per_brand(style_store_period, limit=3000)
    store_style_payload = _cap_rows_per_brand(store_style_period, limit=4000)

    extra_data = {
        "style": style_period.to_dict(orient="records"),
        "style_scope": style_scope_period.to_dict(orient="records"),
        "item_scope": item_scope_period.to_dict(orient="records"),
        "season": season_period.to_dict(orient="records"),
        "category_detail": category_detail_period.to_dict(orient="records"),
        "item_store_period": item_store_period.to_dict(orient="records"),
        "style_channel_period": style_channel_payload.to_dict(orient="records"),
        "style_store_period": style_store_payload.to_dict(orient="records"),
        "store_deep_dive": store_deep_dive.to_dict(orient="records"),
        "store_style": store_style_payload.to_dict(orient="records"),
        "store_category_mix": store_category_mix.to_dict(orient="records"),
        "segment_mix": segment_mix.to_dict(orient="records"),
        "product_mix": product_mix.to_dict(orient="records"),
        "profitability_style": profitability_style.to_dict(orient="records"),
        "profitability_category": profitability_category.to_dict(orient="records"),
        "profitability_style_season": profitability_style_season.to_dict(
            orient="records"
        ),
        "profitability_category_season": profitability_category_season.to_dict(
            orient="records"
        ),
    }

    html_file = output_dir / "weekly_sales_mvp_dashboard.html"
    write_html_dashboard(
        path=html_file,
        generated_at=generated_at,
        kpi=kpi,
        brand_table=brand_decisions,
        channel_table=channel_decisions,
        category_table=category_decisions,
        item_table=item_decisions,
        store_table=store_decisions,
        extra_data=extra_data,
    )

    manifest = {
        "generated_at": generated_at,
        "source_latest_root": str(latest_root),
        "top_n": top_n,
        "payload_limits": {
            "style_channel_period_per_brand": 2500,
            "style_store_period_per_brand": 3000,
            "store_style_per_brand": 4000,
        },
        "outputs": {
            "kpi_file": str(kpi_file),
            "brand_file": str(brand_file),
            "channel_file": str(channel_file),
            "category_file": str(category_file),
            "item_file": str(item_file),
            "store_file": str(store_file),
            "roadmap_file": str(roadmap_file),
            "excel_file": str(excel_file),
            "markdown_file": str(markdown_file),
            "html_file": str(html_file),
        },
        "kpi": kpi,
    }

    manifest_file = output_dir / "mvp_manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest["outputs"]["manifest_file"] = str(manifest_file)
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Generate weekly sales dashboard MVP files from processed marts."
    )
    parser.add_argument(
        "--latest-root",
        type=Path,
        default=root / "Data" / "processed" / "latest",
        help="Path to processed latest directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "Data" / "processed" / "latest" / "mvp",
        help="Directory for MVP outputs.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top rows to keep for item/store/channel/category outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress generated-file logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_mvp_outputs(
        latest_root=args.latest_root,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )

    if args.quiet:
        return

    print("Generated weekly sales MVP outputs:")
    for key, path in manifest["outputs"].items():
        print(f"- {key}: {path}")


if __name__ == "__main__":
    main()
