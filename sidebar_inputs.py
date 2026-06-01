"""Dynamic sidebar material inputs by packaging architecture."""

from __future__ import annotations

import streamlit as st

from constants import INPUT_LAYER_SPECS, PARAM_RANGES, WAFER_RADIUS_MM
from layer_inputs import MaterialInputConfig, default_material_inputs
from packaging_arch import (
    ARCHITECTURE_LAYERS,
    PackagingArchitecture,
    architecture_process_description,
    default_layer_active,
    parse_architecture,
)


def _num(
    label: str,
    key: str,
    lo: float,
    hi: float,
    default: float,
    step: float,
    help_text: str,
) -> float:
    return st.sidebar.number_input(
        label,
        min_value=lo,
        max_value=hi,
        value=default,
        step=step,
        help=help_text,
        key=key,
    )


def _spec_for_key(material_key: str):
    return next(s for s in INPUT_LAYER_SPECS if s.material_key == material_key)


def _render_author_credit() -> None:
    st.sidebar.markdown(
        '<p style="margin:0 0 0.25rem 0;color:#546e7a;font-size:0.9rem;">'
        "Author · <strong>Shih-Ho Lin</strong></p>",
        unsafe_allow_html=True,
    )


def render_material_inputs() -> MaterialInputConfig:
    """Sidebar: architecture + layer stack (bottom→top); all layers Active."""
    _render_author_credit()
    st.sidebar.header("① Material Input")

    arch_label = st.sidebar.radio(
        "Packaging Architecture (封裝架構)",
        [a.value for a in PackagingArchitecture],
        index=0,
        key="packaging_architecture",
    )
    arch = parse_architecture(arch_label)

    st.sidebar.caption(architecture_process_description(arch))
    st.sidebar.divider()

    cfg = default_material_inputs(arch)
    cfg.packaging_architecture = arch.value
    cfg.process_station = ""
    cfg.layer_active = default_layer_active(arch)

    lo, hi, default = PARAM_RANGES["stress_free_T"]
    cfg.T_ref = _num(
        "Stress-free T (°C)",
        f"stress_free_T_{arch.name}",
        lo,
        hi,
        default,
        5.0,
        "模封固化溫度 · 建議 150–200°C",
    )
    lo, hi, default = PARAM_RANGES["measurement_T"]
    cfg.measurement_T_c = _num(
        "Measurement T (°C) (量測溫度)",
        f"measurement_T_{arch.name}",
        lo,
        hi,
        default,
        5.0,
        "最終室溫量測參考（製程曲線用各站 Process T 算 ΔT）",
    )

    st.sidebar.write("")
    cfg.wafer_radius_mm = st.sidebar.slider(
        "Wafer radius (mm)",
        50.0,
        150.0,
        float(WAFER_RADIUS_MM),
        5.0,
        key=f"wafer_r_{arch.name}",
    )

    st.sidebar.divider()
    st.sidebar.markdown("**層疊順序（由下而上 ↑）· 全部 Active**")

    show_emc_cte = False
    show_rdl_cte = False

    for row in ARCHITECTURE_LAYERS[arch]:
        spec = _spec_for_key(row.material_key)
        st.sidebar.markdown(f"**{row.display_name}**")
        cfg.layer_active[row.material_key] = True

        if row.show_split_cte and "EMC" in row.material_key:
            show_emc_cte = True
        if row.show_split_cte and "RDL" in row.material_key:
            show_rdl_cte = True

        t_lo, t_hi = spec.thickness_um_range
        cfg.thickness_um[row.material_key] = _num(
            "  t (µm)",
            f"t_{arch.name}_{row.material_key}",
            t_lo,
            t_hi,
            spec.thickness_um_default,
            5.0 if t_hi > 50 else 1.0,
            spec.note,
        )
        e_lo, e_hi = spec.E_gpa_range
        e_step = 0.001 if spec.E_gpa_default < 0.1 else 0.5
        cfg.E_gpa[row.material_key] = _num(
            "  E (GPa)",
            f"E_{arch.name}_{row.material_key}",
            e_lo,
            e_hi,
            spec.E_gpa_default,
            e_step,
            spec.note,
        )
        if row.material_key == "Solder Bumps":
            st.sidebar.caption(
                "  *Discrete structure: minimal flexural rigidity",
            )
        if row.material_key == "Die/Chip":
            lo, hi, default = PARAM_RANGES["die_area_fraction"]
            cfg.die_area_fraction = _num(
                "  Area Fraction (Die 面積佔比)",
                f"die_frac_{arch.name}",
                lo,
                hi,
                default,
                0.05,
                "模封層內 Die 面積佔比 · 與 EMC 同層複合 (Rule of Mixtures)",
            )
        if not spec.has_split_cte:
            st.sidebar.caption(f"  CTE (ppm/K): {spec.cte_display} (fixed Si/glass)")
        st.sidebar.write("")

    if show_emc_cte or show_rdl_cte:
        st.sidebar.divider()
        st.sidebar.markdown("**CTE 分段 (EMC / RDL)**")

    if show_emc_cte:
        lo, hi, default = PARAM_RANGES["emc_alpha1"]
        cfg.emc_alpha1 = _num("EMC CTE₁ (< Tg)", f"emc_a1_{arch.name}", lo, hi, default, 0.5, "Tg 前")
        lo, hi, default = PARAM_RANGES["emc_alpha2"]
        cfg.emc_alpha2 = _num("EMC CTE₂ (> Tg)", f"emc_a2_{arch.name}", lo, hi, default, 1.0, "Tg 後")

    if show_rdl_cte:
        lo, hi, default = PARAM_RANGES["rdl_alpha1"]
        cfg.rdl_alpha1 = _num("RDL CTE₁", f"rdl_a1_{arch.name}", lo, hi, default, 1.0, "PI 低溫段")
        lo, hi, default = PARAM_RANGES["rdl_alpha2"]
        cfg.rdl_alpha2 = _num("RDL CTE₂", f"rdl_a2_{arch.name}", lo, hi, default, 1.0, "PI 高溫段")
        lo, hi, default = PARAM_RANGES["rdl_Tg"]
        cfg.rdl_Tg = _num("RDL Tg (°C)", f"rdl_tg_{arch.name}", lo, hi, default, 5.0, "玻璃轉化溫度")

    return cfg
