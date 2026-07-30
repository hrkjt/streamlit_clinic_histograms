from __future__ import annotations

import json
import os
from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


st.set_page_config(
    page_title="Clinic Parameter Histograms",
    layout="wide",
)


ID_COLUMN = "ダミーID"
DATE_COLUMN = "診察日"
CLINIC_ORDER = ["日本橋", "関西", "表参道", "福岡"]
CLINIC_BY_PREFIX = {
    "T": "日本橋",
    "K": "関西",
    "H": "表参道",
    "F": "福岡",
}
PARAMETERS = [
    "月齢",
    "前後径",
    "左右径",
    "頭囲",
    "短頭率",
    "前頭部対称率",
    "CA",
    "後頭部対称率",
    "CVAI",
    "CI",
    "後頭部突出度",
    "二五平面短頭率",
]
OUTLIER_LIMITS = {
    "後頭部突出度": (0, 20),
    "二五平面短頭率": (70, 140),
}
TREATMENT_ORDER = ["治療あり", "治療なし"]
TREATMENT_COLORS = {
    "治療あり": "#2563eb",
    "治療なし": "#06b6d4",
}
CLINIC_COLORS = {
    "日本橋": "#1f77b4",
    "関西": "#2ca02c",
    "表参道": "#ff7f0e",
    "福岡": "#d62728",
    "不明": "#6b7280",
}


def get_api_url() -> str:
    try:
        api_url = st.secrets.get("API_URL", "")
    except Exception:
        api_url = ""
    api_url = api_url or os.environ.get("API_URL", "")
    if not api_url:
        st.error("Streamlit Secrets か環境変数に API_URL を設定してください。")
        st.stop()
    return api_url


@st.cache_data(ttl=60 * 30, show_spinner="APIからデータを取得しています")
def fetch_data(api_url: str) -> dict[str, Any]:
    response = requests.get(api_url, timeout=90)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("API response must be a JSON object.")
    return data


def drop_invalid_dummy_id(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COLUMN not in df.columns:
        return df.iloc[0:0].copy()
    valid = df[ID_COLUMN].notna() & (df[ID_COLUMN].astype(str).str.strip() != "")
    return df.loc[valid].copy()


def map_clinic(dummy_id: Any) -> str:
    prefix = str(dummy_id).strip()[:1]
    return CLINIC_BY_PREFIX.get(prefix, "不明")


def normalize_progress(data: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(data.get("経過", []))
    df = drop_invalid_dummy_id(df)
    if df.empty:
        return df

    df[ID_COLUMN] = df[ID_COLUMN].astype(str).str.strip()
    for column in PARAMETERS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["クリニック"] = df[ID_COLUMN].apply(map_clinic)
    if "治療ステータス" in df.columns:
        before = df[df["治療ステータス"].astype(str).str.contains("治療前", na=False)].copy()
        if not before.empty:
            df = before

    sort_cols = [ID_COLUMN]
    if "月齢" in df.columns:
        sort_cols.append("月齢")
    return df.sort_values(sort_cols).drop_duplicates(ID_COLUMN, keep="first")


def build_treatment_ids(data: dict[str, Any]) -> set[str]:
    treated_ids: set[str] = set()

    helmets = drop_invalid_dummy_id(pd.DataFrame(data.get("ヘルメット", [])))
    if not helmets.empty and "ヘルメット" in helmets.columns:
        helmet_name = helmets["ヘルメット"].astype(str).str.strip()
        treated = helmets[helmet_name.ne("") & helmet_name.ne("経過観察")]
        treated_ids.update(treated[ID_COLUMN].astype(str).str.strip().tolist())

    patients = drop_invalid_dummy_id(pd.DataFrame(data.get("患者数", [])))
    if not patients.empty and "発注有無" in patients.columns:
        ordered = patients[patients["発注有無"].astype(str).str.strip().eq("発注済")]
        treated_ids.update(ordered[ID_COLUMN].astype(str).str.strip().tolist())

    return treated_ids


def build_patient_dates(data: dict[str, Any]) -> pd.DataFrame:
    patients = drop_invalid_dummy_id(pd.DataFrame(data.get("患者数", [])))
    if patients.empty or DATE_COLUMN not in patients.columns:
        return pd.DataFrame(columns=[ID_COLUMN, DATE_COLUMN])

    patients = patients[[ID_COLUMN, DATE_COLUMN]].copy()
    patients[ID_COLUMN] = patients[ID_COLUMN].astype(str).str.strip()
    patients[DATE_COLUMN] = pd.to_datetime(
        patients[DATE_COLUMN].astype("string"),
        format="mixed",
        errors="coerce",
        cache=False,
    )
    return (
        patients.dropna(subset=[DATE_COLUMN])
        .sort_values(DATE_COLUMN)
        .drop_duplicates(ID_COLUMN, keep="first")
    )


def prepare_analysis_df(data: dict[str, Any]) -> pd.DataFrame:
    df = normalize_progress(data)
    if df.empty:
        return df

    treated_ids = build_treatment_ids(data)
    df["治療有無"] = np.where(df[ID_COLUMN].isin(treated_ids), "治療あり", "治療なし")
    patient_dates = build_patient_dates(data)
    if not patient_dates.empty:
        df = df.merge(patient_dates, on=ID_COLUMN, how="left")

    for parameter, (lower, upper) in OUTLIER_LIMITS.items():
        if parameter in df.columns:
            df = df[df[parameter].between(lower, upper) | df[parameter].isna()].copy()

    return df


def histogram_bins(series: pd.Series, bin_count: int) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.array([])
    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        pad = max(abs(lower) * 0.05, 1.0)
        lower -= pad
        upper += pad
    return np.linspace(lower, upper, bin_count + 1)


def kde_density(values: pd.Series, x_grid: np.ndarray) -> np.ndarray:
    clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if clean_values.size < 2:
        return np.array([])

    std = float(np.std(clean_values, ddof=1))
    if std == 0:
        return np.array([])

    bandwidth = 1.06 * std * clean_values.size ** (-1 / 5)
    if bandwidth <= 0:
        return np.array([])

    scaled = (x_grid[:, None] - clean_values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled**2).sum(axis=1) / (
        clean_values.size * bandwidth * np.sqrt(2 * np.pi)
    )
    return density


def kde_as_counts(values: pd.Series, x_grid: np.ndarray, bin_width: float) -> np.ndarray:
    density = kde_density(values, x_grid)
    sample_count = pd.to_numeric(values, errors="coerce").dropna().size
    if density.size == 0:
        return np.array([])
    return density * sample_count * bin_width


def make_histogram_figure(
    df: pd.DataFrame,
    parameter: str,
    clinics: list[str],
    bin_count: int,
    show_kde: bool = True,
) -> go.Figure:
    bins = histogram_bins(df[parameter], bin_count)
    fig = make_subplots(
        rows=len(clinics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=clinics,
    )
    if bins.size == 0:
        return fig

    bin_width = float(bins[1] - bins[0])
    centers = bins[:-1] + bin_width / 2
    kde_x = np.linspace(float(bins[0]), float(bins[-1]), 240)
    for row, clinic in enumerate(clinics, start=1):
        clinic_df = df[df["クリニック"] == clinic]
        counts_by_treatment: dict[str, np.ndarray] = {}
        for treatment in TREATMENT_ORDER:
            values = clinic_df[clinic_df["治療有無"] == treatment][parameter].dropna()
            counts, _ = np.histogram(values, bins=bins)
            counts_by_treatment[treatment] = counts
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    width=bin_width * 0.92,
                    name=treatment,
                    marker_color=TREATMENT_COLORS[treatment],
                    legendgroup=treatment,
                    showlegend=row == 1,
                    opacity=0.9,
                    customdata=np.stack([bins[:-1], bins[1:]], axis=-1),
                    hovertemplate=(
                        f"{clinic}<br>{treatment}<br>"
                        f"{parameter}: %{{customdata[0]:.2f}}-%{{customdata[1]:.2f}}<br>"
                        "人数: %{y}<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )

        treated_counts = counts_by_treatment.get("治療あり", np.zeros(len(centers), dtype=int))
        total_counts = sum(counts_by_treatment.values())
        rates = np.divide(
            treated_counts,
            total_counts,
            out=np.zeros_like(treated_counts, dtype=float),
            where=total_counts > 0,
        )
        labels = [f"{rate * 100:.0f}%" if total > 0 else "" for rate, total in zip(rates, total_counts)]
        fig.add_trace(
            go.Scatter(
                x=centers,
                y=total_counts,
                mode="text",
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color="#111827"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=1,
        )

        if show_kde:
            kde_y = kde_as_counts(clinic_df[parameter], kde_x, bin_width)
            if kde_y.size:
                fig.add_trace(
                    go.Scatter(
                        x=kde_x,
                        y=kde_y,
                        mode="lines",
                        name="KDE",
                        line=dict(color="#111827", width=2),
                        legendgroup="KDE",
                        showlegend=row == 1,
                        hovertemplate=f"{clinic}<br>KDE<br>{parameter}: %{{x:.2f}}<br>推定人数: %{{y:.1f}}<extra></extra>",
                    ),
                    row=row,
                    col=1,
                )

    fig.update_layout(
        barmode="stack",
        bargap=0.05,
        height=max(360, 260 * len(clinics)),
        margin=dict(l=40, r=24, t=64, b=48),
        title=dict(text=f"{parameter}の分布: クリニック別・治療有無スタック", x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=None, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(title_text="人数", showgrid=True, gridcolor="#e5e7eb", zeroline=False, rangemode="tozero")
    return fig


def make_kde_comparison_figure(
    df: pd.DataFrame,
    parameter: str,
    clinics: list[str],
    treatment_filter: str | None = None,
) -> go.Figure:
    if treatment_filter:
        df = df[df["治療有無"] == treatment_filter].copy()

    bins = histogram_bins(df[parameter], 40)
    fig = go.Figure()
    if bins.size == 0:
        return fig

    x_grid = np.linspace(float(bins[0]), float(bins[-1]), 320)
    for clinic in clinics:
        clinic_df = df[df["クリニック"] == clinic]
        density = kde_density(clinic_df[parameter], x_grid)
        if density.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=x_grid,
                y=density,
                mode="lines",
                name=f"{clinic} KDE",
                line=dict(color=CLINIC_COLORS.get(clinic, "#6b7280"), width=3),
                hovertemplate=(
                    f"{clinic}<br>{treatment_filter or '全体'} KDE<br>"
                    f"{parameter}: %{{x:.2f}}<br>密度: %{{y:.4f}}<extra></extra>"
                ),
            )
        )

    title_suffix = f"{treatment_filter}: " if treatment_filter else ""
    fig.update_layout(
        height=420,
        margin=dict(l=40, r=24, t=56, b=48),
        title=dict(text=f"{parameter}のKDE比較: {title_suffix}クリニック別", x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text=None, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(title_text="密度", showgrid=True, gridcolor="#e5e7eb", zeroline=False, rangemode="tozero")
    return fig


def make_summary(df: pd.DataFrame, parameter: str) -> pd.DataFrame:
    grouped = (
        df.dropna(subset=[parameter])
        .groupby(["クリニック", "治療有無"], observed=False)
        .agg(
            症例数=(ID_COLUMN, "nunique"),
            平均=(parameter, "mean"),
            中央値=(parameter, "median"),
            標準偏差=(parameter, "std"),
            最小=(parameter, "min"),
            最大=(parameter, "max"),
        )
        .reset_index()
    )
    for column in ["平均", "中央値", "標準偏差", "最小", "最大"]:
        grouped[column] = grouped[column].round(2)
    return grouped


def period_label(selected_period: Any) -> str:
    if isinstance(selected_period, (tuple, list)) and len(selected_period) == 2:
        start_date, end_date = selected_period
        return f"{start_date} - {end_date}"
    return "全期間"


def make_export_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    export_columns = [DATE_COLUMN, "クリニック", "治療有無"] + [
        column for column in PARAMETERS if column in df.columns
    ]
    export_df = df[export_columns].copy()
    if DATE_COLUMN in export_df.columns:
        export_df[DATE_COLUMN] = export_df[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    return json.loads(export_df.to_json(orient="records", force_ascii=False))


def make_export_html(
    fig: go.Figure,
    kde_comparison_fig: go.Figure | None,
    treated_kde_comparison_fig: go.Figure | None,
    summary: pd.DataFrame,
    export_records: list[dict[str, Any]],
    available_parameters: list[str],
    parameter: str,
    selected_clinics: list[str],
    selected_treatment: list[str],
    selected_period: Any,
    bin_count: int,
    show_kde: bool,
    patient_count: int,
    treated_count: int,
    untreated_count: int,
) -> str:
    chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
    kde_chart_html = (
        "<h2>院別KDE比較</h2>"
        + kde_comparison_fig.to_html(full_html=False, include_plotlyjs=False)
        if kde_comparison_fig is not None and len(kde_comparison_fig.data) > 0
        else ""
    )
    treated_kde_chart_html = (
        "<h2>院別KDE比較（治療あり）</h2>"
        + treated_kde_comparison_fig.to_html(full_html=False, include_plotlyjs=False)
        if treated_kde_comparison_fig is not None and len(treated_kde_comparison_fig.data) > 0
        else ""
    )
    summary_html = summary.to_html(index=False, border=0, classes="summary-table")
    clinics_text = ", ".join(selected_clinics) if selected_clinics else "なし"
    treatment_text = ", ".join(selected_treatment) if selected_treatment else "なし"
    records_json = json.dumps(export_records, ensure_ascii=False, allow_nan=False)
    parameters_json = json.dumps(available_parameters, ensure_ascii=False)
    clinics_json = json.dumps(selected_clinics, ensure_ascii=False)
    treatment_json = json.dumps(selected_treatment, ensure_ascii=False)
    default_period_json = json.dumps(
        list(selected_period) if isinstance(selected_period, (tuple, list)) and len(selected_period) == 2 else [],
        ensure_ascii=False,
        default=str,
    )

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(parameter)} clinic histogram report</title>
  <style>
    body {{
      margin: 0;
      color: #111827;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .app-shell {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 20px;
      border-right: 1px solid #e5e7eb;
      background: #ffffff;
    }}
    main {{
      min-width: 0;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 24px;
      letter-spacing: 0;
    }}
    h2 {{
      margin-top: 28px;
      font-size: 18px;
    }}
    label {{
      display: block;
      margin: 12px 0 6px;
      font-size: 13px;
      font-weight: 700;
    }}
    select,
    input[type="date"],
    input[type="number"] {{
      width: 100%;
      box-sizing: border-box;
      padding: 7px 8px;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      background: #ffffff;
    }}
    .check-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 7px 0;
      font-size: 14px;
      font-weight: 400;
    }}
    .check-row input {{
      width: auto;
    }}
    .chart-card {{
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
      padding: 14px;
      margin-bottom: 18px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin: 16px 0 24px;
    }}
    .meta div {{
      padding: 10px 12px;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      background: #f9fafb;
      font-size: 14px;
    }}
    .summary-table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 24px;
      font-size: 14px;
    }}
    .summary-table th,
    .summary-table td {{
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      text-align: left;
    }}
    .summary-table th {{
      background: #f3f4f6;
      font-weight: 700;
    }}
    @media (max-width: 820px) {{
      .app-shell {{
        grid-template-columns: 1fr;
      }}
      aside {{
        position: static;
        height: auto;
      }}
    }}
  </style>
</head>
<body>
<div class="app-shell">
  <aside>
    <h2>表示設定</h2>
    <label for="parameterSelect">パラメータ</label>
    <select id="parameterSelect"></select>
    <label for="startDate">対象期間</label>
    <input id="startDate" type="date">
    <label for="endDate">終了日</label>
    <input id="endDate" type="date">
    <label for="binCount">bin数</label>
    <input id="binCount" type="number" min="5" max="50" step="1" value="{bin_count}">
    <label>クリニック</label>
    <div id="clinicControls"></div>
    <label>治療有無</label>
    <div id="treatmentControls"></div>
    <label class="check-row"><input id="showKde" type="checkbox" {"checked" if show_kde else ""}>ヒストグラムにKDEを重ねる</label>
    <label class="check-row"><input id="showKdeComparison" type="checkbox" checked>院別KDE比較を表示</label>
    <label class="check-row"><input id="showTreatedKdeComparison" type="checkbox" checked>院別KDE比較（治療あり）を表示</label>
  </aside>
  <main>
    <h1>クリニック別 パラメータ分布</h1>
    <div id="metrics" class="meta">
      <div><strong>パラメータ</strong><br>{escape(parameter)}</div>
      <div><strong>対象期間</strong><br>{escape(period_label(selected_period))}</div>
      <div><strong>クリニック</strong><br>{escape(clinics_text)}</div>
      <div><strong>治療有無</strong><br>{escape(treatment_text)}</div>
      <div><strong>bin数</strong><br>{bin_count}</div>
      <div><strong>KDE</strong><br>{"表示" if show_kde else "非表示"}</div>
      <div><strong>対象患者</strong><br>{patient_count:,}</div>
      <div><strong>治療あり</strong><br>{treated_count:,}</div>
      <div><strong>治療なし</strong><br>{untreated_count:,}</div>
    </div>
    <div style="display:none">{chart_html}</div>
    <div class="chart-card"><div id="histChart"></div></div>
    <div id="kdeComparisonWrap" class="chart-card"><div id="kdeComparisonChart"></div></div>
    <div id="treatedKdeComparisonWrap" class="chart-card"><div id="treatedKdeComparisonChart"></div></div>
    <h2>集計表</h2>
    <div id="summaryTable">{summary_html}</div>
  </main>
</div>
<script>
const DATA = {records_json};
const PARAMETERS = {parameters_json};
const CLINICS = ["日本橋", "関西", "表参道", "福岡"].filter(c => DATA.some(r => r["クリニック"] === c));
const TREATMENTS = ["治療あり", "治療なし"];
const DEFAULT_PARAMETER = {json.dumps(parameter, ensure_ascii=False)};
const DEFAULT_CLINICS = {clinics_json};
const DEFAULT_TREATMENTS = {treatment_json};
const DEFAULT_PERIOD = {default_period_json};
const TREATMENT_COLORS_JS = {{"治療あり":"#2563eb","治療なし":"#06b6d4"}};
const CLINIC_COLORS_JS = {json.dumps(CLINIC_COLORS, ensure_ascii=False)};

function finiteValues(rows, parameter) {{
  return rows.map(r => Number(r[parameter])).filter(Number.isFinite);
}}
function linspace(start, end, count) {{
  if (count <= 1) return [start];
  const step = (end - start) / (count - 1);
  return Array.from({{length: count}}, (_, i) => start + i * step);
}}
function makeBins(values, binCount) {{
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {{
    const pad = Math.max(Math.abs(min) * 0.05, 1);
    min -= pad;
    max += pad;
  }}
  return linspace(min, max, binCount + 1);
}}
function histogram(values, bins) {{
  const counts = Array(bins.length - 1).fill(0);
  for (const value of values) {{
    for (let i = 0; i < bins.length - 1; i++) {{
      const isLast = i === bins.length - 2;
      if ((value >= bins[i] && value < bins[i + 1]) || (isLast && value === bins[i + 1])) {{
        counts[i] += 1;
        break;
      }}
    }}
  }}
  return counts;
}}
function mean(values) {{
  return values.reduce((a, b) => a + b, 0) / values.length;
}}
function std(values) {{
  if (values.length < 2) return 0;
  const m = mean(values);
  return Math.sqrt(values.reduce((s, v) => s + (v - m) ** 2, 0) / (values.length - 1));
}}
function kdeDensity(values, xs) {{
  if (values.length < 2) return [];
  const s = std(values);
  if (!Number.isFinite(s) || s === 0) return [];
  const bw = 1.06 * s * values.length ** (-1 / 5);
  if (bw <= 0) return [];
  const norm = values.length * bw * Math.sqrt(2 * Math.PI);
  return xs.map(x => values.reduce((sum, v) => sum + Math.exp(-0.5 * ((x - v) / bw) ** 2), 0) / norm);
}}
function selectedCheckboxes(name) {{
  return Array.from(document.querySelectorAll(`input[name="${{name}}"]:checked`)).map(el => el.value);
}}
function currentState() {{
  return {{
    parameter: document.getElementById("parameterSelect").value,
    startDate: document.getElementById("startDate").value,
    endDate: document.getElementById("endDate").value,
    binCount: Math.max(5, Math.min(50, Number(document.getElementById("binCount").value) || 20)),
    clinics: selectedCheckboxes("clinic"),
    treatments: selectedCheckboxes("treatment"),
    showKde: document.getElementById("showKde").checked,
    showKdeComparison: document.getElementById("showKdeComparison").checked,
    showTreatedKdeComparison: document.getElementById("showTreatedKdeComparison").checked,
  }};
}}
function filterRows(state) {{
  return DATA.filter(r =>
    state.clinics.includes(r["クリニック"]) &&
    state.treatments.includes(r["治療有無"]) &&
    r[state.parameter] !== null &&
    Number.isFinite(Number(r[state.parameter])) &&
    (!state.startDate || r["診察日"] >= state.startDate) &&
    (!state.endDate || r["診察日"] <= state.endDate)
  );
}}
function uniqueCount(rows) {{
  return rows.length;
}}
function renderMetrics(rows, state) {{
  const treated = rows.filter(r => r["治療有無"] === "治療あり").length;
  const untreated = rows.filter(r => r["治療有無"] === "治療なし").length;
  document.getElementById("metrics").innerHTML = `
    <div><strong>パラメータ</strong><br>${{state.parameter}}</div>
    <div><strong>対象期間</strong><br>${{state.startDate}} - ${{state.endDate}}</div>
    <div><strong>クリニック</strong><br>${{state.clinics.join(", ") || "なし"}}</div>
    <div><strong>治療有無</strong><br>${{state.treatments.join(", ") || "なし"}}</div>
    <div><strong>bin数</strong><br>${{state.binCount}}</div>
    <div><strong>KDE</strong><br>${{state.showKde ? "表示" : "非表示"}}</div>
    <div><strong>対象患者</strong><br>${{uniqueCount(rows).toLocaleString()}}</div>
    <div><strong>治療あり</strong><br>${{treated.toLocaleString()}}</div>
    <div><strong>治療なし</strong><br>${{untreated.toLocaleString()}}</div>`;
}}
function renderHistogram(rows, state) {{
  const values = finiteValues(rows, state.parameter);
  if (!values.length) {{
    Plotly.react("histChart", [], {{title: "対象データがありません"}});
    return;
  }}
  const bins = makeBins(values, state.binCount);
  const binWidth = bins[1] - bins[0];
  const centers = bins.slice(0, -1).map((b, i) => b + binWidth / 2);
  const traces = [];
  const annotations = [];
  const rowCount = state.clinics.length;
  state.clinics.forEach((clinic, idx) => {{
    const axisSuffix = idx === 0 ? "" : String(idx + 1);
    const clinicRows = rows.filter(r => r["クリニック"] === clinic);
    const countsByTreatment = {{}};
    TREATMENTS.forEach(treatment => {{
      const vals = finiteValues(clinicRows.filter(r => r["治療有無"] === treatment), state.parameter);
      const counts = histogram(vals, bins);
      countsByTreatment[treatment] = counts;
      traces.push({{
        type: "bar",
        x: centers,
        y: counts,
        width: binWidth * 0.92,
        name: treatment,
        marker: {{color: TREATMENT_COLORS_JS[treatment]}},
        legendgroup: treatment,
        showlegend: idx === 0,
        xaxis: "x" + axisSuffix,
        yaxis: "y" + axisSuffix,
        hovertemplate: `${{clinic}}<br>${{treatment}}<br>${{state.parameter}}: %{{x:.2f}}<br>人数: %{{y}}<extra></extra>`
      }});
    }});
    const total = countsByTreatment["治療あり"].map((_, i) => countsByTreatment["治療あり"][i] + countsByTreatment["治療なし"][i]);
    const labels = total.map((n, i) => n > 0 ? `${{Math.round(countsByTreatment["治療あり"][i] / n * 100)}}%` : "");
    traces.push({{
      type: "scatter",
      mode: "text",
      x: centers,
      y: total,
      text: labels,
      textposition: "top center",
      textfont: {{size: 12, color: "#111827"}},
      hoverinfo: "skip",
      showlegend: false,
      xaxis: "x" + axisSuffix,
      yaxis: "y" + axisSuffix
    }});
    if (state.showKde) {{
      const kdeX = linspace(bins[0], bins[bins.length - 1], 240);
      const kdeValues = finiteValues(clinicRows, state.parameter);
      const kdeY = kdeDensity(kdeValues, kdeX).map(d => d * kdeValues.length * binWidth);
      if (kdeY.length) {{
        traces.push({{
          type: "scatter",
          mode: "lines",
          x: kdeX,
          y: kdeY,
          name: "KDE",
          line: {{color: "#111827", width: 2}},
          legendgroup: "KDE",
          showlegend: idx === 0,
          xaxis: "x" + axisSuffix,
          yaxis: "y" + axisSuffix,
          hovertemplate: `${{clinic}}<br>KDE<br>${{state.parameter}}: %{{x:.2f}}<br>推定人数: %{{y:.1f}}<extra></extra>`
        }});
      }}
    }}
    annotations.push({{
      text: clinic,
      x: 0.5,
      xref: "paper",
      y: 1 - idx / rowCount,
      yref: "paper",
      yanchor: "bottom",
      showarrow: false,
      font: {{size: 16}}
    }});
  }});
  const layout = {{
    title: `${{state.parameter}}の分布: クリニック別・治療有無スタック`,
    grid: {{rows: rowCount, columns: 1, pattern: "independent"}},
    barmode: "stack",
    bargap: 0.05,
    height: Math.max(360, 260 * rowCount),
    margin: {{l: 48, r: 24, t: 64, b: 48}},
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    annotations
  }};
  for (let i = 1; i <= rowCount; i++) {{
    const suffix = i === 1 ? "" : String(i);
    layout["xaxis" + suffix] = {{showgrid: true, gridcolor: "#e5e7eb", zeroline: false, title: ""}};
    layout["yaxis" + suffix] = {{showgrid: true, gridcolor: "#e5e7eb", zeroline: false, title: "人数", rangemode: "tozero"}};
  }}
  Plotly.react("histChart", traces, layout, {{responsive: true}});
}}
function renderKdeChart(divId, rows, state, treatmentFilter, titlePrefix) {{
  const sourceRows = treatmentFilter ? rows.filter(r => r["治療有無"] === treatmentFilter) : rows;
  const values = finiteValues(sourceRows, state.parameter);
  if (!values.length) {{
    Plotly.react(divId, [], {{title: "対象データがありません"}});
    return;
  }}
  const bins = makeBins(values, 40);
  const xs = linspace(bins[0], bins[bins.length - 1], 320);
  const traces = [];
  state.clinics.forEach(clinic => {{
    const clinicRows = sourceRows.filter(r => r["クリニック"] === clinic);
    const vals = finiteValues(clinicRows, state.parameter);
    const ys = kdeDensity(vals, xs);
    if (ys.length) {{
      traces.push({{
        type: "scatter",
        mode: "lines",
        x: xs,
        y: ys,
        name: `${{clinic}} KDE`,
        line: {{color: CLINIC_COLORS_JS[clinic] || "#6b7280", width: 3}},
        hovertemplate: `${{clinic}}<br>${{treatmentFilter || "全体"}} KDE<br>${{state.parameter}}: %{{x:.2f}}<br>密度: %{{y:.4f}}<extra></extra>`
      }});
    }}
  }});
  Plotly.react(divId, traces, {{
    title: `${{state.parameter}}のKDE比較: ${{titlePrefix}}クリニック別`,
    height: 420,
    margin: {{l: 48, r: 24, t: 56, b: 48}},
    plot_bgcolor: "white",
    paper_bgcolor: "white",
    xaxis: {{title: "", showgrid: true, gridcolor: "#e5e7eb", zeroline: false}},
    yaxis: {{title: "密度", showgrid: true, gridcolor: "#e5e7eb", zeroline: false, rangemode: "tozero"}},
    legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}}
  }}, {{responsive: true}});
}}
function renderSummary(rows, state) {{
  const groups = [];
  state.clinics.forEach(clinic => TREATMENTS.forEach(treatment => {{
    if (!state.treatments.includes(treatment)) return;
    const vals = finiteValues(rows.filter(r => r["クリニック"] === clinic && r["治療有無"] === treatment), state.parameter);
    if (!vals.length) return;
    vals.sort((a, b) => a - b);
    const m = mean(vals);
    const median = vals.length % 2 ? vals[(vals.length - 1) / 2] : (vals[vals.length / 2 - 1] + vals[vals.length / 2]) / 2;
    groups.push([clinic, treatment, vals.length, m, median, std(vals), vals[0], vals[vals.length - 1]]);
  }}));
  const rowsHtml = groups.map(g => `<tr>${{g.map((v, i) => `<td>${{i >= 3 ? Number(v).toFixed(2) : v}}</td>`).join("")}}</tr>`).join("");
  document.getElementById("summaryTable").innerHTML = `<table class="summary-table"><thead><tr><th>クリニック</th><th>治療有無</th><th>症例数</th><th>平均</th><th>中央値</th><th>標準偏差</th><th>最小</th><th>最大</th></tr></thead><tbody>${{rowsHtml}}</tbody></table>`;
}}
function render() {{
  const state = currentState();
  const rows = filterRows(state);
  renderMetrics(rows, state);
  renderHistogram(rows, state);
  document.getElementById("kdeComparisonWrap").style.display = state.showKdeComparison ? "block" : "none";
  document.getElementById("treatedKdeComparisonWrap").style.display = state.showTreatedKdeComparison ? "block" : "none";
  if (state.showKdeComparison) renderKdeChart("kdeComparisonChart", rows, state, null, "");
  if (state.showTreatedKdeComparison) renderKdeChart("treatedKdeComparisonChart", rows, state, "治療あり", "治療あり: ");
  renderSummary(rows, state);
}}
function initControls() {{
  const parameterSelect = document.getElementById("parameterSelect");
  PARAMETERS.forEach(p => {{
    const opt = document.createElement("option");
    opt.value = p;
    opt.textContent = p;
    if (p === DEFAULT_PARAMETER) opt.selected = true;
    parameterSelect.appendChild(opt);
  }});
  const dates = DATA.map(r => r["診察日"]).filter(Boolean).sort();
  const minDate = dates[0];
  const maxDate = dates[dates.length - 1];
  document.getElementById("startDate").value = DEFAULT_PERIOD[0] || minDate;
  document.getElementById("startDate").min = minDate;
  document.getElementById("startDate").max = maxDate;
  document.getElementById("endDate").value = DEFAULT_PERIOD[1] || maxDate;
  document.getElementById("endDate").min = minDate;
  document.getElementById("endDate").max = maxDate;
  document.getElementById("clinicControls").innerHTML = CLINICS.map(c => `<label class="check-row"><input name="clinic" type="checkbox" value="${{c}}" ${{DEFAULT_CLINICS.includes(c) ? "checked" : ""}}>${{c}}</label>`).join("");
  document.getElementById("treatmentControls").innerHTML = TREATMENTS.map(t => `<label class="check-row"><input name="treatment" type="checkbox" value="${{t}}" ${{DEFAULT_TREATMENTS.includes(t) ? "checked" : ""}}>${{t}}</label>`).join("");
  document.querySelectorAll("select,input").forEach(el => el.addEventListener("change", render));
  render();
}}
initControls();
</script>
</body>
</html>
"""


def main() -> None:
    st.title("クリニック別 パラメータ分布")

    try:
        data = fetch_data(get_api_url())
        df = prepare_analysis_df(data)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    available_parameters = [column for column in PARAMETERS if column in df.columns and df[column].notna().any()]
    available_clinics = [clinic for clinic in CLINIC_ORDER if clinic in set(df["クリニック"])]
    unknown_count = int((df["クリニック"] == "不明").sum()) if "クリニック" in df.columns else 0

    if df.empty or not available_parameters or not available_clinics:
        st.warning("表示できるデータがありません。APIレスポンスと列名を確認してください。")
        st.stop()

    with st.sidebar:
        st.header("表示設定")
        selected_parameter = st.selectbox("パラメータ", available_parameters, index=available_parameters.index("CVAI") if "CVAI" in available_parameters else 0)
        if DATE_COLUMN in df.columns and df[DATE_COLUMN].notna().any():
            date_values = df[DATE_COLUMN].dropna()
            selected_period = st.date_input(
                "対象期間",
                value=(date_values.min().date(), date_values.max().date()),
                min_value=date_values.min().date(),
                max_value=date_values.max().date(),
            )
        else:
            selected_period = None
        selected_clinics = st.multiselect("クリニック", available_clinics, default=available_clinics)
        selected_treatment = st.multiselect("治療有無", TREATMENT_ORDER, default=TREATMENT_ORDER)
        bin_count = st.slider("bin数", min_value=5, max_value=50, value=20, step=1)
        show_kde = st.checkbox("ヒストグラムにKDEを重ねる", value=True)
        show_kde_comparison = st.checkbox("院別KDE比較を表示", value=True)
        show_treated_kde_comparison = st.checkbox("院別KDE比較（治療あり）を表示", value=True)
        show_table = st.checkbox("集計表を表示", value=True)

    filtered = df[
        df["クリニック"].isin(selected_clinics)
        & df["治療有無"].isin(selected_treatment)
        & df[selected_parameter].notna()
    ].copy()
    if selected_period and len(selected_period) == 2 and DATE_COLUMN in filtered.columns:
        start_date, end_date = selected_period
        filtered = filtered[
            filtered[DATE_COLUMN].dt.date.between(start_date, end_date)
        ].copy()

    patient_count = int(filtered[ID_COLUMN].nunique())
    treated_count = int(filtered[filtered["治療有無"] == "治療あり"][ID_COLUMN].nunique())
    untreated_count = int(filtered[filtered["治療有無"] == "治療なし"][ID_COLUMN].nunique())

    c1, c2, c3 = st.columns(3)
    c1.metric("対象患者", f"{patient_count:,}")
    c2.metric("治療あり", f"{treated_count:,}")
    c3.metric("治療なし", f"{untreated_count:,}")

    if unknown_count:
        st.caption(f"クリニック不明IDは {unknown_count:,} 件あり、比較対象から除外しています。")

    if filtered.empty:
        st.warning("現在のフィルタでは対象データがありません。")
        st.stop()

    fig = make_histogram_figure(filtered, selected_parameter, selected_clinics, bin_count, show_kde)
    kde_comparison_fig = (
        make_kde_comparison_figure(filtered, selected_parameter, selected_clinics)
        if show_kde_comparison
        else None
    )
    treated_kde_comparison_fig = (
        make_kde_comparison_figure(filtered, selected_parameter, selected_clinics, "治療あり")
        if show_treated_kde_comparison
        else None
    )
    summary = make_summary(filtered, selected_parameter)

    st.plotly_chart(fig, width="stretch")
    if kde_comparison_fig is not None and len(kde_comparison_fig.data) > 0:
        st.plotly_chart(kde_comparison_fig, width="stretch")
    if treated_kde_comparison_fig is not None and len(treated_kde_comparison_fig.data) > 0:
        st.plotly_chart(treated_kde_comparison_fig, width="stretch")

    export_html = make_export_html(
        fig=fig,
        kde_comparison_fig=kde_comparison_fig,
        treated_kde_comparison_fig=treated_kde_comparison_fig,
        summary=summary,
        export_records=make_export_records(df),
        available_parameters=available_parameters,
        parameter=selected_parameter,
        selected_clinics=selected_clinics,
        selected_treatment=selected_treatment,
        selected_period=selected_period,
        bin_count=bin_count,
        show_kde=show_kde,
        patient_count=patient_count,
        treated_count=treated_count,
        untreated_count=untreated_count,
    )
    st.download_button(
        "現在の表示をHTMLでダウンロード",
        data=export_html.encode("utf-8"),
        file_name=f"clinic_histograms_{selected_parameter}.html",
        mime="text/html",
    )

    if show_table:
        st.subheader("集計表")
        st.dataframe(summary, width="stretch", hide_index=True)

    with st.expander("データ確認"):
        st.write("APIキー:", list(data.keys()))
        st.write("使用列:", [ID_COLUMN, DATE_COLUMN, "クリニック", "治療有無", selected_parameter])
        st.dataframe(
            filtered[[column for column in [ID_COLUMN, DATE_COLUMN, "クリニック", "治療有無", selected_parameter] if column in filtered.columns]].head(100),
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
