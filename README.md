# Advanced Packaging Warpage Simulator

Advanced packaging warpage simulator (multi-layer plate theory + Streamlit UI). Includes a **2D Optimization & Process Window** tab: grid search (custom Min/Max, 20×20), ±1 mm safe zone contour, and optimal ★ marker.

**Author · Shih-Ho Lin**

📄 完整功能、參數、公式與物理說明請見 **[使用說明.md](./使用說明.md)**。

## Setup (CMD)

```bat
cd fowlp_simulator
python -m venv venv
call venv\Scripts\activate
pip install streamlit numpy pandas plotly scipy
```

Or double-click `setup.bat` once.

## Run on localhost

```bat
call venv\Scripts\activate
streamlit run app.py
```

Browser: http://localhost:8501

Or double-click `run.bat` after setup.
