#!/usr/bin/env python3
"""Convert Weekly_Sales_Review workbook into dashboard-ready CSV datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from build_weekly_sales_mvp import generate_mvp_outputs


SHEET_TIMEFRAME_MAP = {
    "cum_TY": "cum_ty",
    "cum_LY": "cum_ly",
    "period_TY": "period_ty",
    "period-1w(7d)_TY": "period_prev_week_ty",
    "period_LY": "period_ly",
    "order": "order",
}

TIMEFRAME_GROUP_MAP = {
    "cum_ty": "cumulative",
    "cum_ly": "cumulative",
    "period_ty": "period",
    "period_prev_week_ty": "period",
    "period_ly": "period",
    "order": "order",
}

COLUMN_RENAME_MAP = {
    "Sales Month": "sales_month",
    "Brand": "brand",
    "Item": "item",
    "Uni/Wms/Kids": "segment",
    "Year": "year",
    "Season": "season",
    "SS/FW": "season_half",
    "Year+Season": "year_season",
    "New/Old": "new_old",
    "Channel": "channel",
    "Store": "store_code",
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
    "Category": "category",
    "Sales Amt_TAG(V-)": "sales_amt_tag_v_excl",
    "Sales Amt_Net(V-)": "sales_amt_net_v_excl",
    "basic_item_code": "style_no",
}

EXPECTED_COLUMNS = [
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
    "tag_price",
    "cost_per_pcs_v_incl",
    "sales_qty",
    "sales_amt_net",
    "sales_amt_tag",
    "cost_amt_v_incl",
    "channel_type",
    "category",
    "sales_amt_tag_v_excl",
    "sales_amt_net_v_excl",
    "order_qty",
    "order_amt",
]

STRING_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
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
]

DIMENSIONS = {
    "brand": ["brand"],
    "channel": ["brand", "channel", "channel_type"],
    "category": ["brand", "category"],
    "item": ["brand", "item", "category"],
    "store": ["brand", "store_code", "store_name", "channel"],
}

PERIOD_COMPARE_CODES = ["period_ty", "period_ly", "period_prev_week_ty"]
CUMULATIVE_COMPARE_CODES = ["cum_ty", "cum_ly"]


def normalize_header(header: object) -> str:
    text = str(header or "").strip()
    return re.sub(r"\s+", " ", text)


def slugify_sheet_name(sheet_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", sheet_name).strip("_").lower()


def parse_week_label(path: Path) -> str:
    match = re.search(r"[Ww](\d+)", path.stem)
    if not match:
        return ""
    return f"W{match.group(1)}"


def clean_numeric(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.replace(",", "", regex=False)
    text = text.replace({"": pd.NA, "-": pd.NA, "nan": pd.NA})
    return pd.to_numeric(text, errors="coerce")


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint(path: Path) -> dict:
    stat = path.stat()
    return {
        "file_name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": file_sha256(path),
    }


def clean_sheet(df: pd.DataFrame, sheet_name: str, week_label: str) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        normalized = normalize_header(col)
        if normalized in COLUMN_RENAME_MAP:
            rename_map[col] = COLUMN_RENAME_MAP[normalized]
        else:
            snake = re.sub(r"[^a-zA-Z0-9]+", "_", normalized).strip("_").lower()
            rename_map[col] = snake or "unknown"

    out = df.rename(columns=rename_map).copy()

    for column in EXPECTED_COLUMNS:
        if column not in out.columns:
            out[column] = pd.NA

    for column in STRING_COLUMNS:
        out[column] = out[column].astype("string").fillna("").str.strip()

    for column in NUMERIC_COLUMNS:
        out[column] = clean_numeric(out[column])

    timeframe_code = SHEET_TIMEFRAME_MAP.get(sheet_name, "unknown")
    out["source_sheet"] = sheet_name
    out["timeframe_code"] = timeframe_code
    out["timeframe_group"] = TIMEFRAME_GROUP_MAP.get(timeframe_code, "unknown")
    out["source_week"] = week_label

    ordered = EXPECTED_COLUMNS + [
        "source_sheet",
        "timeframe_code",
        "timeframe_group",
        "source_week",
    ]
    extra = [col for col in out.columns if col not in ordered]
    return out[ordered + extra]


def build_aggregate(fact: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metric_columns = [
        "sales_qty",
        "sales_amt_net",
        "sales_amt_tag",
        "sales_amt_net_v_excl",
        "sales_amt_tag_v_excl",
        "cost_amt_v_incl",
    ]

    grouped = (
        fact.groupby(["timeframe_code", "timeframe_group", *keys], dropna=False)[
            metric_columns
        ]
        .sum(min_count=1)
        .reset_index()
    )

    grouped["gross_margin_amt"] = (
        grouped["sales_amt_net_v_excl"] - grouped["cost_amt_v_incl"]
    )

    grouped["gross_margin_rate"] = grouped["gross_margin_amt"].div(
        grouped["sales_amt_net_v_excl"]
    )
    grouped["gross_margin_rate"] = grouped["gross_margin_rate"].where(
        grouped["sales_amt_net_v_excl"].ne(0), pd.NA
    )

    grouped["avg_selling_price"] = grouped["sales_amt_net_v_excl"].div(
        grouped["sales_qty"]
    )
    grouped["avg_selling_price"] = grouped["avg_selling_price"].where(
        grouped["sales_qty"].ne(0), pd.NA
    )

    grouped["discount_rate"] = 1 - grouped["sales_amt_net_v_excl"].div(
        grouped["sales_amt_tag_v_excl"]
    )
    grouped["discount_rate"] = grouped["discount_rate"].where(
        grouped["sales_amt_tag_v_excl"].ne(0), pd.NA
    )

    return grouped


def _build_metric_pivot(
    frame: pd.DataFrame,
    keys: list[str],
    timeframe_codes: list[str],
    metric_column: str,
    value_prefix: str,
) -> pd.DataFrame:
    pivot = frame.pivot_table(
        index=keys,
        columns="timeframe_code",
        values=metric_column,
        aggfunc="sum",
        fill_value=0,
    ).reset_index()

    for code in timeframe_codes:
        if code not in pivot.columns:
            pivot[code] = 0

    rename_map = {code: f"{value_prefix}_{code}" for code in timeframe_codes}
    out = pivot[keys + timeframe_codes].rename(columns=rename_map)
    return out


def _safe_pct(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    pct = numerator.div(denominator)
    return pct.where(denominator.ne(0), pd.NA)


def build_period_comparison(aggregate: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    scope = aggregate[aggregate["timeframe_code"].isin(PERIOD_COMPARE_CODES)].copy()

    sales = _build_metric_pivot(
        scope,
        keys,
        PERIOD_COMPARE_CODES,
        metric_column="sales_amt_net_v_excl",
        value_prefix="sales",
    )
    qty = _build_metric_pivot(
        scope,
        keys,
        PERIOD_COMPARE_CODES,
        metric_column="sales_qty",
        value_prefix="qty",
    )

    comparison = sales.merge(qty, on=keys, how="outer")

    comparison["yoy_sales_diff"] = (
        comparison["sales_period_ty"] - comparison["sales_period_ly"]
    )
    comparison["yoy_sales_pct"] = _safe_pct(
        comparison["yoy_sales_diff"], comparison["sales_period_ly"]
    )

    comparison["wow_sales_diff"] = (
        comparison["sales_period_ty"] - comparison["sales_period_prev_week_ty"]
    )
    comparison["wow_sales_pct"] = _safe_pct(
        comparison["wow_sales_diff"], comparison["sales_period_prev_week_ty"]
    )

    comparison["yoy_qty_diff"] = (
        comparison["qty_period_ty"] - comparison["qty_period_ly"]
    )
    comparison["yoy_qty_pct"] = _safe_pct(
        comparison["yoy_qty_diff"], comparison["qty_period_ly"]
    )

    comparison["wow_qty_diff"] = (
        comparison["qty_period_ty"] - comparison["qty_period_prev_week_ty"]
    )
    comparison["wow_qty_pct"] = _safe_pct(
        comparison["wow_qty_diff"], comparison["qty_period_prev_week_ty"]
    )

    comparison = comparison.sort_values("sales_period_ty", ascending=False).reset_index(
        drop=True
    )
    comparison["rank_period_ty"] = comparison.index + 1
    return comparison


def build_cumulative_comparison(
    aggregate: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    scope = aggregate[aggregate["timeframe_code"].isin(CUMULATIVE_COMPARE_CODES)].copy()

    sales = _build_metric_pivot(
        scope,
        keys,
        CUMULATIVE_COMPARE_CODES,
        metric_column="sales_amt_net_v_excl",
        value_prefix="sales",
    )
    qty = _build_metric_pivot(
        scope,
        keys,
        CUMULATIVE_COMPARE_CODES,
        metric_column="sales_qty",
        value_prefix="qty",
    )

    comparison = sales.merge(qty, on=keys, how="outer")
    comparison["yoy_sales_diff"] = (
        comparison["sales_cum_ty"] - comparison["sales_cum_ly"]
    )
    comparison["yoy_sales_pct"] = _safe_pct(
        comparison["yoy_sales_diff"], comparison["sales_cum_ly"]
    )

    comparison["yoy_qty_diff"] = comparison["qty_cum_ty"] - comparison["qty_cum_ly"]
    comparison["yoy_qty_pct"] = _safe_pct(
        comparison["yoy_qty_diff"], comparison["qty_cum_ly"]
    )

    comparison = comparison.sort_values("sales_cum_ty", ascending=False).reset_index(
        drop=True
    )
    comparison["rank_cum_ty"] = comparison.index + 1
    return comparison


def build_insights_snapshot(
    period_brand: pd.DataFrame, period_channel: pd.DataFrame, fact: pd.DataFrame
) -> dict:
    period_scope = fact[fact["timeframe_code"].isin(PERIOD_COMPARE_CODES)]
    total_period = (
        period_scope.groupby("timeframe_code", dropna=False)["sales_amt_net_v_excl"]
        .sum(min_count=1)
        .to_dict()
    )

    def top_rows(frame: pd.DataFrame, metric: str, limit: int = 10) -> list[dict]:
        if frame.empty or metric not in frame.columns:
            return []
        cols = [col for col in ["brand", "channel", metric] if col in frame.columns]
        rows = frame.nlargest(limit, metric)[cols]
        return rows.to_dict(orient="records")

    growth_scope = period_brand[period_brand["sales_period_ly"] > 0]
    return {
        "period_totals": {
            "sales_period_ty": total_period.get("period_ty", 0),
            "sales_period_ly": total_period.get("period_ly", 0),
            "sales_period_prev_week_ty": total_period.get("period_prev_week_ty", 0),
        },
        "top_brands_by_sales_ty": top_rows(period_brand, "sales_period_ty"),
        "top_brands_by_yoy_pct": top_rows(growth_scope, "yoy_sales_pct"),
        "top_channels_by_sales_ty": top_rows(period_channel, "sales_period_ty"),
    }


def build_latest_outputs(input_path: Path, processed_root: Path) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    latest_root = processed_root / "latest"
    if latest_root.exists():
        shutil.rmtree(latest_root)

    raw_dir = latest_root / "raw"
    facts_dir = latest_root / "facts"
    marts_dir = latest_root / "marts"
    insights_dir = latest_root / "insights"
    for path in [raw_dir, facts_dir, marts_dir, insights_dir]:
        path.mkdir(parents=True, exist_ok=True)

    week_label = parse_week_label(input_path)
    workbook = pd.read_excel(input_path, sheet_name=None)

    sheet_frames: dict[str, pd.DataFrame] = {}
    sheet_summaries: dict[str, dict] = {}

    for sheet_name, frame in workbook.items():
        cleaned = clean_sheet(frame, sheet_name, week_label)
        sheet_frames[sheet_name] = cleaned

        sheet_file = raw_dir / f"sheet_{slugify_sheet_name(sheet_name)}.csv"
        cleaned.to_csv(sheet_file, index=False, encoding="utf-8")
        sheet_summaries[sheet_name] = {
            "rows": int(len(cleaned)),
            "columns": list(cleaned.columns),
            "file": str(sheet_file),
        }

    fact = pd.concat(sheet_frames.values(), ignore_index=True)
    fact_file = facts_dir / "sales_fact_all.csv"
    fact.to_csv(fact_file, index=False, encoding="utf-8")

    aggregate_files = {}
    period_compare_files = {}
    cumulative_compare_files = {}
    aggregate_frames: dict[str, pd.DataFrame] = {}
    period_frames: dict[str, pd.DataFrame] = {}

    for name, keys in DIMENSIONS.items():
        aggregate = build_aggregate(fact, keys)
        aggregate_frames[name] = aggregate
        aggregate_file = marts_dir / f"agg_{name}.csv"
        aggregate.to_csv(aggregate_file, index=False, encoding="utf-8")
        aggregate_files[name] = str(aggregate_file)

        period_compare = build_period_comparison(aggregate, keys)
        period_frames[name] = period_compare
        period_file = marts_dir / f"compare_period_{name}.csv"
        period_compare.to_csv(period_file, index=False, encoding="utf-8")
        period_compare_files[name] = str(period_file)

        cumulative_compare = build_cumulative_comparison(aggregate, keys)
        cumulative_file = marts_dir / f"compare_cumulative_{name}.csv"
        cumulative_compare.to_csv(cumulative_file, index=False, encoding="utf-8")
        cumulative_compare_files[name] = str(cumulative_file)

    insights = build_insights_snapshot(
        period_brand=period_frames["brand"],
        period_channel=period_frames["channel"],
        fact=fact,
    )
    insights_file = insights_dir / "insights_snapshot.json"
    insights_file.write_text(
        json.dumps(insights, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    mvp_manifest = generate_mvp_outputs(
        latest_root=latest_root,
        output_dir=latest_root / "mvp",
        top_n=20,
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_file": str(input_path),
        "source_week": week_label,
        "source_fingerprint": source_fingerprint(input_path),
        "sheets": sheet_summaries,
        "fact_file": str(fact_file),
        "aggregate_files": aggregate_files,
        "period_comparison_files": period_compare_files,
        "cumulative_comparison_files": cumulative_compare_files,
        "insights_file": str(insights_file),
        "mvp_outputs": mvp_manifest["outputs"],
        "mvp_kpi": mvp_manifest["kpi"],
    }

    manifest_file = latest_root / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    readme = latest_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Weekly Sales Processed Data",
                "",
                f"- Source workbook: `{input_path}`",
                f"- Generated at: `{manifest['generated_at']}`",
                "- Key outputs:",
                "  - `raw/sheet_*.csv` (sheet-level normalized raw exports)",
                "  - `facts/sales_fact_all.csv` (combined fact table)",
                "  - `marts/agg_*.csv` (dimension aggregates)",
                "  - `marts/compare_period_*.csv` (기간 TY vs LY vs 전주)",
                "  - `marts/compare_cumulative_*.csv` (누적 TY vs LY)",
                "  - `insights/insights_snapshot.json` (quick insight seeds)",
                "  - `manifest.json` (schema + source fingerprint)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "latest_root": str(latest_root),
        "manifest_file": str(manifest_file),
        "readme_file": str(readme),
        "fact_file": str(fact_file),
        "aggregate_files": aggregate_files,
        "period_comparison_files": period_compare_files,
        "cumulative_comparison_files": cumulative_compare_files,
        "insights_file": str(insights_file),
        "mvp_outputs": mvp_manifest["outputs"],
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert Weekly_Sales_Review workbook to analysis-ready CSV datasets."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "Data" / "Weekly_Sales_Review_W6.xlsx",
        help="Path to source xlsx file.",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=root / "Data" / "processed",
        help="Root directory for processed outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress generated-file logs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated = build_latest_outputs(args.input, args.processed_root)

    if args.quiet:
        return

    print("Generated latest weekly-sales datasets:")
    print(f"- {generated['fact_file']}")
    print(f"- {generated['manifest_file']}")
    print(f"- {generated['insights_file']}")
    for name, path in generated["period_comparison_files"].items():
        print(f"- compare_period_{name}: {path}")


if __name__ == "__main__":
    main()
