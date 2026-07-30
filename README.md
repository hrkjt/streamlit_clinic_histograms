# streamlit_clinic_histograms

位置的頭蓋変形データを API から取得し、各パラメータのヒストグラムをクリニック別に比較する Streamlit アプリです。

## 機能

- 治療前データを患者ごとに 1 行へ集約
- `ダミーID` の prefix からクリニックを分類
- ヘルメット情報または `発注有無 == "発注済"` から治療有無を判定
- 各パラメータのヒストグラムを治療有無でスタック
- 同じ bin 幅でクリニック別に縦並び比較
- クリニック・治療有無・bin 数を UI から変更

## クリニック分類

| prefix | クリニック |
|---|---|
| `T` | 日本橋 |
| `K` | 関西 |
| `H` | 表参道 |
| `F` | 福岡 |

## Secrets

`.streamlit/secrets.toml` に以下を設定してください。

```toml
API_URL = "https://script.google.com/macros/s/AKfycby3oyGaFq8X_JkxOFUB_QrXccegZs4kNpIZvSSt6Dtx9poU8pEf_rQEvLFQzK-OlmX0/exec"
```

Streamlit Cloud ではアプリの Secrets 設定に同じ値を登録します。

## ローカル実行

```bash
pip install -r requirements.txt
streamlit run app.py
```

Apple Silicon環境でNumPyのarchitecture mismatchが出る場合:

```bash
arch -arm64 python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
arch -arm64 python -m streamlit run app.py
```

## 参考

- <https://github.com/hrkjt/streamlit_PHDcharts>
