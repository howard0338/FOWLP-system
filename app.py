"""
FOWLP Warpage Simulator — Input (E, CTE, t) → Calculation → Output (σ, z_NA).

Run: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from constants import INPUT_LAYER_SPECS, PARAM_RANGES, WARPAGE_CRITICAL_MM, WAFER_RADIUS_MM, ProcessStep
from layer_inputs import (
    MaterialInputConfig,
    default_material_inputs,
    input_current_table_rows,
    input_reference_table_rows,
)
from warpage_engine import (
    plot_process_timeline,
    plot_stress_map,
    plot_wafer_surface,
    simulate_process_sequence,
)


st.set_page_config(
    page_title="FOWLP Warpage Simulator | 3DIC",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_critical_border(critical: bool) -> None:
    color = "#e53935" if critical else "#1e88e5"
    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            border: 4px solid {color};
            border-radius: 8px;
        }}
        .critical-banner {{
            background: #ffebee;
            color: #b71c1c;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            margin-bottom: 8px;
        }}
        .io-caption {{
            color: #546e7a;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _num(label: str, key: str, lo: float, hi: float, default: float, step: float, help_text: str) -> float:
    return st.sidebar.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=default,
        step=step,
        help=help_text,
        key=key,
    )


def render_author_credit(*, sidebar: bool = False) -> None:
    target = st.sidebar if sidebar else st
    target.markdown(
        '<p style="margin:0 0 0.25rem 0;color:#546e7a;font-size:0.9rem;">'
        "Author · <strong>Shih-Ho Lin</strong></p>",
        unsafe_allow_html=True,
    )


def render_material_inputs() -> MaterialInputConfig:
    """Sidebar: intrinsic material properties (Input layer)."""
    render_author_credit(sidebar=True)
    st.sidebar.header("① Material Input")
    st.sidebar.caption("FOWLP 多層板 **材料原生性質** (E, CTE, t)，含矽 Interposer — 非計算結果。")

    cfg = default_material_inputs()

    lo, hi, default = PARAM_RANGES["stress_free_T"]
    cfg.T_ref = _num(
        "Stress-free T (°C)",
        "stress_free_T",
        lo,
        hi,
        default,
        5.0,
        "模封固化溫度 · 建議 150–200°C",
    )

    cfg.wafer_radius_mm = st.sidebar.slider(
        "Wafer radius (mm)",
        50.0,
        150.0,
        float(WAFER_RADIUS_MM),
        5.0,
    )

    for spec in INPUT_LAYER_SPECS:
        st.sidebar.markdown(f"**{spec.display_name}**")
        t_lo, t_hi = spec.thickness_um_range
        cfg.thickness_um[spec.material_key] = _num(
            f"  t (µm)",
            f"t_{spec.material_key}",
            t_lo,
            t_hi,
            spec.thickness_um_default,
            5.0 if t_hi > 50 else 1.0,
            spec.note,
        )
        e_lo, e_hi = spec.E_gpa_range
        cfg.E_gpa[spec.material_key] = _num(
            f"  E (GPa)",
            f"E_{spec.material_key}",
            e_lo,
            e_hi,
            spec.E_gpa_default,
            0.5,
            spec.note,
        )

    st.sidebar.markdown("**CTE 分段 (EMC / RDL)**")
    lo, hi, default = PARAM_RANGES["emc_alpha1"]
    cfg.emc_alpha1 = _num("EMC CTE₁ (< Tg)", "emc_a1", lo, hi, default, 0.5, "Tg 前")
    lo, hi, default = PARAM_RANGES["emc_alpha2"]
    cfg.emc_alpha2 = _num("EMC CTE₂ (> Tg)", "emc_a2", lo, hi, default, 1.0, "Tg 後")
    lo, hi, default = PARAM_RANGES["rdl_alpha1"]
    cfg.rdl_alpha1 = _num("RDL CTE₁", "rdl_a1", lo, hi, default, 1.0, "PI 低溫段")
    lo, hi, default = PARAM_RANGES["rdl_alpha2"]
    cfg.rdl_alpha2 = _num("RDL CTE₂", "rdl_a2", lo, hi, default, 1.0, "PI 高溫段")
    lo, hi, default = PARAM_RANGES["rdl_Tg"]
    cfg.rdl_Tg = _num("RDL Tg (°C)", "rdl_Tg", lo, hi, default, 5.0, "玻璃轉化溫度")

    return cfg


def render_calculation_output(result, snap) -> None:
    """Computed stack state — Output layer (σ, z_NA from plate theory)."""
    st.markdown(
        '<p class="io-caption">② <strong>Calculation Output</strong> — '
        "由輸入層 E / CTE / t 經多層板公式求得（含溫度依賴 E<sub>eff</sub>、α<sub>eff</sub>）。</p>",
        unsafe_allow_html=True,
    )
    st.metric(
        "Neutral axis z_NA (from bottom)",
        f"{result.neutral_axis_m * 1e6:.2f} µm",
        help="整疊層中性軸位置，非材料輸入參數",
    )
    st.dataframe(
        [
            {
                "Layer (層別)": ls.material.name,
                "t (µm)": ls.thickness_m * 1e6,
                "E_eff (GPa)": round(ls.E_eff_pa / 1e9, 3),
                "α_eff (ppm/K)": round(ls.alpha_eff_ppm, 2),
                "σ_residual (MPa)": round(ls.sigma_pa / 1e6, 3),
                "z from NA (µm)": round(ls.z_from_na_m * 1e6, 2),
            }
            for ls in result.layers
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_input_reference_section(cfg: MaterialInputConfig) -> None:
    """Static defaults + current user input (not solver output)."""
    render_author_credit()
    st.markdown(
        '<p class="io-caption">① <strong>Material Input</strong> — '
        "材料原生性質；與下方 Output 表格欄位不同，兩者不衝突。</p>",
        unsafe_allow_html=True,
    )
    tab_ref, tab_cur = st.tabs(["Engineering defaults (建議值)", "Your current input (目前輸入)"])
    with tab_ref:
        st.dataframe(input_reference_table_rows(), use_container_width=True, hide_index=True)
    with tab_cur:
        st.dataframe(input_current_table_rows(cfg), use_container_width=True, hide_index=True)


def _pick_step_index(selection, n_steps: int) -> int | None:
    if selection is None:
        return None
    points = getattr(getattr(selection, "selection", None), "points", None)
    if not points:
        return None
    pt = points[0]
    return int(pt.get("point_index", pt.get("point_number", 0))) % n_steps


def main() -> None:
    render_author_credit()
    st.title("FOWLP Warpage Simulator")
    st.caption(
        "Center-referenced w(r)=κr²/2 · FOWLP stack · 點選製程曲線查看翹曲"
    )

    st.info(
        "**FOWLP 層疊**：基板 → **矽 Interposer（100–200 µm，可選高剛性層）** → EMC / Die / RDL。"
        "計算中 Interposer 以較高彎矩權重模擬**強約束**，EMC 軟層彎矩權重降低。"
        "側邊欄可調 **材料輸入** → Stoney / 多層板 **計算** → 下方 **z_NA、σ** 輸出。"
    )

    mat_in = render_material_inputs()
    solver_kw = mat_in.to_solver_kwargs()

    snapshots = simulate_process_sequence(
        wafer_radius_mm=mat_in.wafer_radius_mm,
        T_ref_initial=mat_in.T_ref,
        **solver_kw,
    )
    n_steps = len(snapshots)
    labels = [s.label for s in snapshots]

    if "selected_step_idx" not in st.session_state:
        st.session_state.selected_step_idx = 0

    st.subheader("Process warpage timeline")
    fig_curve = plot_process_timeline(snapshots, st.session_state.selected_step_idx)
    selection = st.plotly_chart(
        fig_curve,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="process_timeline_chart",
        config={"scrollZoom": True},
    )
    clicked = _pick_step_index(selection, n_steps)
    if clicked is not None:
        st.session_state.selected_step_idx = clicked

    picked = st.selectbox(
        "製程步驟",
        options=list(range(n_steps)),
        format_func=lambda i: labels[i],
        index=st.session_state.selected_step_idx,
        key="step_selectbox",
    )
    st.session_state.selected_step_idx = picked

    snap = snapshots[picked]
    result = snap.result
    w_char = snap.warpage_edge_mm
    critical = abs(w_char) > WARPAGE_CRITICAL_MM
    inject_critical_border(critical)

    if critical:
        st.markdown(
            '<div class="critical-banner">⚠ Critical Fail: |warpage| > 1.5 mm</div>',
            unsafe_allow_html=True,
        )
    if snap.step == ProcessStep.DEBONDING:
        st.info("**Debonding**：載板約束釋放 · 翹曲跳升（signed）")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Warpage @ edge", f"{w_char:.3f} mm")
    c2.metric("Warpage @ center", f"{snap.warpage_center_mm:.3f} mm", help="Center-referenced: always 0")
    c3.metric("Shape", snap.shape_label)
    c4.metric("κ (1/m)", f"{result.kappa_1_per_m:.3e}")

    st.subheader(f"③ Warpage map — {snap.label}")
    step_title = f"{snap.label} @ {snap.temperature_c:.0f}°C"
    kappa = result.kappa_1_per_m
    radius_mm = mat_in.wafer_radius_mm
    chart_key = f"{picked}_{kappa:.6e}_{w_char:.4f}"

    col_w, col_s = st.columns([1.2, 1.0])
    with col_w:
        st.plotly_chart(
            plot_wafer_surface(kappa, radius_mm, title=f"3D warpage — {step_title}"),
            use_container_width=True,
            key=f"surf_{chart_key}",
            config={"scrollZoom": True},
        )
    with col_s:
        st.caption(
            "應力圖為**簡化估算**：各層殘留熱應力 σᵢ 疊加 + 曲率 κ 引起之彎曲應力（非完整 FEM）。"
        )
        st.plotly_chart(
            plot_stress_map(result, radius_mm, title=f"Stress σ — {step_title}"),
            use_container_width=True,
            key=f"stress_{chart_key}",
            config={"scrollZoom": True},
        )

    st.divider()
    render_input_reference_section(mat_in)

    st.divider()
    with st.expander("Calculation output — layer stack (計算輸出)", expanded=True):
        render_calculation_output(result, snap)


if __name__ == "__main__":
    main()
