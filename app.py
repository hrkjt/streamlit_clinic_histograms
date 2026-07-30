from __future__ import annotations

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


def make_histogram_figure(df: pd.DataFrame, parameter: str, clinics: list[str], bin_count: int) -> go.Figure:
    bins = histogram_bins(df[parameter], bin_count)
    fig = make_subplots(
        rows=len(clinics),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=clinics,
    )
    if bins.size == 0:
        return fig

    bin_width = float(bins[1] - bins[0])
    centers = bins[:-1] + bin_width / 2
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
    fig.update_xaxes(title_text=parameter, showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(title_text="人数", showgrid=True, gridcolor="#e5e7eb", zeroline=False, rangemode="tozero")
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


def make_export_html(
    fig: go.Figure,
    summary: pd.DataFrame,
    parameter: str,
    selected_clinics: list[str],
    selected_treatment: list[str],
    selected_period: Any,
    bin_count: int,
    patient_count: int,
    treated_count: int,
    untreated_count: int,
) -> str:
    chart_html = fig.to_html(full_html=False, include_plotlyjs=True)
    summary_html = summary.to_html(index=False, border=0, classes="summary-table")
    clinics_text = ", ".join(selected_clinics) if selected_clinics else "なし"
    treatment_text = ", ".join(selected_treatment) if selected_treatment else "なし"

    return f"""<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(parameter)} clinic histogram report</title>
  <style>
    body {{
      margin: 0;
      padding: 32px;
      color: #111827;
      background: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      background: #ffffff;
      padding: 28px;
      border: 1px solid #e5e7eb;
      border-radius: 8px;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 24px;
      letter-spacing: 0;
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
  </style>
</head>
<body>
  <main>
    <h1>クリニック別 パラメータ分布</h1>
    <div class="meta">
      <div><strong>パラメータ</strong><br>{escape(parameter)}</div>
      <div><strong>対象期間</strong><br>{escape(period_label(selected_period))}</div>
      <div><strong>クリニック</strong><br>{escape(clinics_text)}</div>
      <div><strong>治療有無</strong><br>{escape(treatment_text)}</div>
      <div><strong>bin数</strong><br>{bin_count}</div>
      <div><strong>対象患者</strong><br>{patient_count:,}</div>
      <div><strong>治療あり</strong><br>{treated_count:,}</div>
      <div><strong>治療なし</strong><br>{untreated_count:,}</div>
    </div>
    {chart_html}
    <h2>集計表</h2>
    {summary_html}
  </main>
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

    fig = make_histogram_figure(filtered, selected_parameter, selected_clinics, bin_count)
    summary = make_summary(filtered, selected_parameter)

    st.plotly_chart(fig, width="stretch")

    export_html = make_export_html(
        fig=fig,
        summary=summary,
        parameter=selected_parameter,
        selected_clinics=selected_clinics,
        selected_treatment=selected_treatment,
        selected_period=selected_period,
        bin_count=bin_count,
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
