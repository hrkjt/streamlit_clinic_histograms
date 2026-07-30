from __future__ import annotations

import os
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


def prepare_analysis_df(data: dict[str, Any]) -> pd.DataFrame:
    df = normalize_progress(data)
    if df.empty:
        return df

    treated_ids = build_treatment_ids(data)
    df["治療有無"] = np.where(df[ID_COLUMN].isin(treated_ids), "治療あり", "治療なし")

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

    xbins = dict(start=float(bins[0]), end=float(bins[-1]), size=float(bins[1] - bins[0]))
    for row, clinic in enumerate(clinics, start=1):
        clinic_df = df[df["クリニック"] == clinic]
        for treatment in TREATMENT_ORDER:
            values = clinic_df[clinic_df["治療有無"] == treatment][parameter].dropna()
            fig.add_trace(
                go.Histogram(
                    x=values,
                    xbins=xbins,
                    name=treatment,
                    marker_color=TREATMENT_COLORS[treatment],
                    legendgroup=treatment,
                    showlegend=row == 1,
                    opacity=0.9,
                    hovertemplate=f"{clinic}<br>{treatment}<br>{parameter}: %{{x}}<br>人数: %{{y}}<extra></extra>",
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
    fig.update_yaxes(title_text="人数", showgrid=True, gridcolor="#e5e7eb", zeroline=False)
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
        selected_clinics = st.multiselect("クリニック", available_clinics, default=available_clinics)
        selected_treatment = st.multiselect("治療有無", TREATMENT_ORDER, default=TREATMENT_ORDER)
        bin_count = st.slider("bin数", min_value=5, max_value=50, value=20, step=1)
        show_table = st.checkbox("集計表を表示", value=True)

    filtered = df[
        df["クリニック"].isin(selected_clinics)
        & df["治療有無"].isin(selected_treatment)
        & df[selected_parameter].notna()
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("対象患者", f"{filtered[ID_COLUMN].nunique():,}")
    c2.metric("治療あり", f"{filtered[filtered['治療有無'] == '治療あり'][ID_COLUMN].nunique():,}")
    c3.metric("治療なし", f"{filtered[filtered['治療有無'] == '治療なし'][ID_COLUMN].nunique():,}")

    if unknown_count:
        st.caption(f"クリニック不明IDは {unknown_count:,} 件あり、比較対象から除外しています。")

    if filtered.empty:
        st.warning("現在のフィルタでは対象データがありません。")
        st.stop()

    st.plotly_chart(
        make_histogram_figure(filtered, selected_parameter, selected_clinics, bin_count),
        use_container_width=True,
    )

    if show_table:
        st.subheader("集計表")
        st.dataframe(make_summary(filtered, selected_parameter), use_container_width=True, hide_index=True)

    with st.expander("データ確認"):
        st.write("APIキー:", list(data.keys()))
        st.write("使用列:", [ID_COLUMN, "クリニック", "治療有無", selected_parameter])
        st.dataframe(
            filtered[[ID_COLUMN, "クリニック", "治療有無", selected_parameter]].head(100),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
