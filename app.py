"""

Advanced Packaging Warpage Simulator — Input (E, CTE, t) → Calculation → Output (σ, z_NA).



Run: streamlit run app.py

"""



from __future__ import annotations



from pathlib import Path

from typing import Any



import streamlit as st



from constants import WARPAGE_CRITICAL_MM, WARPAGE_CRITICAL_UM, WARPAGE_UNIT, mm_to_um

from design_window import (

    cfg_to_hashable,

    material_config_from_hashable,

    render_design_window_page,

    run_design_window_sweep,

)

from layer_inputs import (
    MaterialInputConfig,
    input_current_table_rows,
    input_reference_table_rows,
)
from sidebar_inputs import render_material_inputs

from stack_visual import render_stack_schematic

from warpage_engine import (

    plot_process_timeline,

    plot_stress_map,

    plot_wafer_surface,

    simulate_process_sequence,

)





st.set_page_config(

    page_title="Advanced Packaging Warpage Simulator | 3DIC",

    page_icon="🔬",

    layout="wide",

    initial_sidebar_state="expanded",

)





@st.cache_data(show_spinner="計算 Design Window…")

def cached_design_window_sweep(
    cfg_hash: tuple[Any, ...],
    param1_id: str,
    param2_id: str,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n1: int,
    n2: int,
    metric: str,
    step_name: str,
) -> dict[str, Any]:
    cfg = material_config_from_hashable(cfg_hash)
    return run_design_window_sweep(
        cfg,
        param1_id,
        param2_id,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        n1=n1,
        n2=n2,
        metric=metric,  # type: ignore[arg-type]
        eval_label=step_name,
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

        div[data-testid="column"]:has([data-testid="stPlotlyChart"]) {{

            min-width: 280px;

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





def render_simulator_tab(mat_in: MaterialInputConfig) -> None:
    from packaging_arch import (
        architecture_process_description,
        timeline_labels_for_architecture,
        timeline_layer_active,
        timeline_steps_for_architecture,
    )

    solver_kw = mat_in.to_solver_kwargs()

    snapshots = simulate_process_sequence(
        wafer_radius_mm=mat_in.wafer_radius_mm,
        T_ref_initial=mat_in.T_ref,
        **solver_kw,
    )
    n_steps = len(snapshots)
    labels = timeline_labels_for_architecture(mat_in.packaging_architecture)

    arch_key = mat_in.packaging_architecture.replace(" ", "_")
    if st.session_state.get("timeline_arch_key") != arch_key:
        st.session_state.timeline_arch_key = arch_key
        st.session_state.selected_step_idx = 0

    if "selected_step_idx" not in st.session_state:
        st.session_state.selected_step_idx = 0
    if st.session_state.selected_step_idx >= n_steps:
        st.session_state.selected_step_idx = 0

    render_author_credit()

    st.title("Advanced Packaging Warpage Simulator")
    st.caption(
        "Center-referenced w(r)=κr²/2 · 2.5D stack · 點選製程曲線或下拉選單，右側堆疊同步更新"
    )
    st.info(architecture_process_description(mat_in.packaging_architecture))
    st.caption(
        "側邊欄層級由下而上排列。製程曲線依時間軸**逐步動態 Active**；"
        "製程時間軸：模封前 **ΔT=0**；模封後 **ΔT = Process T − Stress-free T**（動態熱歷史）。"
    )

    with st.expander("📖 使用說明 — 功能、參數、公式與物理意義", expanded=False):
        doc_path = Path(__file__).resolve().parent / "使用說明.md"
        if doc_path.is_file():
            st.markdown(doc_path.read_text(encoding="utf-8"))
        else:
            st.caption("請見專案根目錄 `使用說明.md`。")

    col_timeline, col_stack = st.columns([1.45, 1.0], gap="large")

    with col_timeline:
        st.subheader("Process warpage timeline")
        fig_curve = plot_process_timeline(
            snapshots,
            st.session_state.selected_step_idx,
            architecture_label=mat_in.packaging_architecture,
        )
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
        st.session_state.selected_step_idx = int(picked)

    idx = st.session_state.selected_step_idx
    snap = snapshots[idx]
    tsteps = timeline_steps_for_architecture(mat_in.packaging_architecture)
    step_active = timeline_layer_active(tsteps[idx], mat_in.packaging_architecture)

    with col_stack:
        st.caption(f"**檢視製程站點：** {snap.label}")
        render_stack_schematic(
            mat_in,
            snap.step,
            snap.label,
            layer_active=step_active,
            z_na_um=snap.result.neutral_axis_m * 1e6,
            temperature_c=snap.temperature_c,
        )

    st.divider()

    result = snap.result

    w_char = snap.warpage_edge_mm

    critical = abs(w_char) > WARPAGE_CRITICAL_MM

    inject_critical_border(critical)



    if critical:

        st.markdown(

            f'<div class="critical-banner">⚠ Critical Fail: |warpage| > {WARPAGE_CRITICAL_UM:.0f} {WARPAGE_UNIT}</div>',

            unsafe_allow_html=True,

        )

    if snap.label == "Carrier Release":
        st.info("**Carrier Release**：載板約束釋放 · 翹曲跳升（signed）")
    elif snap.label == "Substrate Integration":
        st.info("**Substrate Integration**：結合基板 · 整疊層參與計算")



    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Warpage @ edge", f"{mm_to_um(w_char):.1f} {WARPAGE_UNIT}")

    c2.metric(
        "Warpage @ center",
        f"{mm_to_um(snap.warpage_center_mm):.1f} {WARPAGE_UNIT}",
        help="Center-referenced: always 0",
    )

    c3.metric("Shape", snap.shape_label)

    c4.metric("κ (1/m)", f"{result.kappa_1_per_m:.3e}")



    st.subheader(f"③ Warpage map — {snap.label}")

    step_title = f"{snap.label} @ {snap.temperature_c:.0f}°C"

    kappa = result.kappa_1_per_m

    radius_mm = mat_in.wafer_radius_mm

    chart_key = f"{picked}_{kappa:.6e}_{w_char:.4f}"



    col_w, col_s = st.columns([1.2, 1.0], gap="medium")

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





def main() -> None:

    mat_in = render_material_inputs()



    tab_sim, tab_dw = st.tabs(
        ["Warpage Simulator", "2D Optimization & Process Window"]
    )



    with tab_sim:

        render_simulator_tab(mat_in)



    with tab_dw:

        render_design_window_page(mat_in, sweep_runner=cached_design_window_sweep)





if __name__ == "__main__":

    main()


