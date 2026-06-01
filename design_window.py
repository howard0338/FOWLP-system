"""2D parameter optimization & process-window grid search + contour visualization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from constants import INPUT_LAYER_SPECS, UM_PER_MM, WARPAGE_UNIT, mm_to_um
from packaging_arch import final_timeline_label, timeline_labels_for_architecture
from layer_inputs import MaterialInputConfig
from packaging_arch import DW_DEFAULT_PARAMS, PackagingArchitecture, parse_architecture, visible_material_keys
from warpage_engine import simulate_process_sequence

MetricKind = Literal["abs_debond", "abs_max_process", "signed_debond"]

# Safe process window (signed warpage at evaluation step)
PROCESS_WINDOW_W_MIN_MM = -1.0
PROCESS_WINDOW_W_MAX_MM = 1.0
DEFAULT_GRID_POINTS = 20

# 2D Optimization axis options only — omit fixed Si die props & standard solder alloy
_OPT_SKIP: dict[str, frozenset[str]] = {
    "Die/Chip": frozenset({"E_gpa"}),
    "Solder Bumps": frozenset({"thickness", "E_gpa"}),
}


@dataclass(frozen=True)
class OptParamSpec:
    """Single tunable scalar for 2D design-window sweeps."""

    param_id: str
    label: str
    kind: str
    lo: float
    hi: float
    default: float
    material_key: str | None = None
    step: float = 1.0


def build_opt_param_catalog(cfg: MaterialInputConfig | None = None) -> list[OptParamSpec]:
    """Parameters available for X/Y axis selectboxes in 2D Optimization tab."""
    arch = parse_architecture(cfg.packaging_architecture if cfg else PackagingArchitecture.FOWLP_INFO.value)
    visible = set(visible_material_keys(arch))

    specs: list[OptParamSpec] = []
    for layer in INPUT_LAYER_SPECS:
        if layer.material_key not in visible:
            continue
        skip = _OPT_SKIP.get(layer.material_key, frozenset())
        t_lo, t_hi = layer.thickness_um_range
        step = 5.0 if t_hi > 50 else 1.0
        if "thickness" not in skip:
            specs.append(
                OptParamSpec(
                    param_id=f"t:{layer.material_key}",
                    label=f"{layer.display_name} — 厚度 t (µm)",
                    kind="thickness",
                    lo=t_lo,
                    hi=t_hi,
                    default=layer.thickness_um_default,
                    material_key=layer.material_key,
                    step=step,
                )
            )
        if "E_gpa" not in skip:
            e_lo, e_hi = layer.E_gpa_range
            specs.append(
                OptParamSpec(
                    param_id=f"E:{layer.material_key}",
                    label=f"{layer.display_name} — 模數 E (GPa)",
                    kind="E_gpa",
                    lo=e_lo,
                    hi=e_hi,
                    default=layer.E_gpa_default,
                    material_key=layer.material_key,
                    step=0.5,
                )
            )

    from constants import PARAM_RANGES

    show_emc = "EMC (Molding)" in visible
    show_rdl = "RDL (Polyimide)" in visible

    scalar_defs: list[tuple[str, str, str, tuple[float, float, float], float]] = [
        ("T_ref", "Stress-free T (°C)", "T_ref", PARAM_RANGES["stress_free_T"], 5.0),
    ]
    if show_emc:
        scalar_defs.extend(
            [
                ("emc_alpha1", "EMC CTE₁ (ppm/K)", "emc_alpha1", PARAM_RANGES["emc_alpha1"], 0.5),
                ("emc_alpha2", "EMC CTE₂ (ppm/K)", "emc_alpha2", PARAM_RANGES["emc_alpha2"], 1.0),
            ]
        )
    if show_rdl:
        scalar_defs.extend(
            [
                ("rdl_alpha1", "RDL CTE₁ (ppm/K)", "rdl_alpha1", PARAM_RANGES["rdl_alpha1"], 1.0),
                ("rdl_alpha2", "RDL CTE₂ (ppm/K)", "rdl_alpha2", PARAM_RANGES["rdl_alpha2"], 1.0),
                ("rdl_Tg", "RDL Tg (°C)", "rdl_Tg", PARAM_RANGES["rdl_Tg"], 5.0),
            ]
        )
    for pid, label, kind, (lo, hi, default), step in scalar_defs:
        if any(s.param_id == pid for s in specs):
            continue
        specs.append(
            OptParamSpec(
                param_id=pid,
                label=label,
                kind=kind,
                lo=lo,
                hi=hi,
                default=default,
                step=step,
            )
        )

    # Deduplicate by param_id (RDL thickness may appear twice)
    seen: set[str] = set()
    unique: list[OptParamSpec] = []
    for s in specs:
        if s.param_id in seen:
            continue
        seen.add(s.param_id)
        unique.append(s)
    return unique


@dataclass(frozen=True)
class PhysicalScanBounds:
    """Semiconductor process physical limits for 2D sweep axis widgets."""

    min_limit: float
    max_limit: float
    default_lo: float
    default_hi: float


def physical_bounds_from_label(label: str, spec: OptParamSpec | None = None) -> PhysicalScanBounds:
    """
    Map selectbox display label (keyword rules) → (hard Min, hard Max, default sweep lo, default sweep hi).
    Falls back to catalog OptParamSpec.lo/hi when no rule matches.
    """
    text = label

    if "厚度" in text:
        if "EMC" in text:
            return PhysicalScanBounds(50.0, 800.0, 100.0, 400.0)
        if "RDL" in text:
            return PhysicalScanBounds(2.0, 50.0, 5.0, 20.0)
        if "Die" in text:
            return PhysicalScanBounds(20.0, 800.0, 50.0, 200.0)
        if "Carrier" in text:
            return PhysicalScanBounds(300.0, 1500.0, 500.0, 1000.0)

    if "模數" in text:
        if "EMC" in text:
            return PhysicalScanBounds(3.0, 40.0, 10.0, 30.0)
        if "RDL" in text:
            return PhysicalScanBounds(1.0, 15.0, 2.0, 10.0)
        if "Die" in text:
            return PhysicalScanBounds(100.0, 200.0, 120.0, 150.0)
        if "Carrier" in text:
            return PhysicalScanBounds(50.0, 90.0, 60.0, 80.0)
        if "Solder" in text:
            return PhysicalScanBounds(0.01, 50.0, 0.01, 10.0)

    if spec is not None:
        kind = spec.kind
        if kind == "emc_alpha1":
            return PhysicalScanBounds(3.0, 30.0, 5.0, 20.0)
        if kind == "emc_alpha2":
            return PhysicalScanBounds(20.0, 120.0, 30.0, 80.0)
        if kind == "rdl_alpha1":
            return PhysicalScanBounds(20.0, 80.0, 30.0, 60.0)
        if kind == "rdl_alpha2":
            return PhysicalScanBounds(30.0, 150.0, 50.0, 100.0)
        if kind == "T_ref":
            return PhysicalScanBounds(80.0, 250.0, 100.0, 200.0)
        if kind == "rdl_Tg":
            return PhysicalScanBounds(200.0, 400.0, 200.0, 300.0)

    if "EMC" in text and ("CTE₁" in text or "CTE1" in text):
        return PhysicalScanBounds(3.0, 30.0, 5.0, 20.0)
    if "EMC" in text and ("CTE₂" in text or "CTE2" in text):
        return PhysicalScanBounds(20.0, 120.0, 30.0, 80.0)
    if "RDL" in text and ("CTE₁" in text or "CTE1" in text):
        return PhysicalScanBounds(20.0, 80.0, 30.0, 60.0)
    if "RDL" in text and ("CTE₂" in text or "CTE2" in text):
        return PhysicalScanBounds(30.0, 150.0, 50.0, 100.0)

    if "Stress-free" in text:
        return PhysicalScanBounds(80.0, 250.0, 100.0, 200.0)
    if "Tg" in text:
        return PhysicalScanBounds(200.0, 400.0, 200.0, 300.0)

    if spec is not None:
        lo, hi = float(spec.lo), float(spec.hi)
        if lo >= hi:
            return PhysicalScanBounds(lo, hi, lo, hi)
        span = 0.35 * (hi - lo)
        mid = 0.5 * (lo + hi)
        return PhysicalScanBounds(lo, hi, max(lo, mid - span), min(hi, mid + span))

    return PhysicalScanBounds(0.0, 1.0, 0.0, 1.0)


def default_scan_bounds(spec: OptParamSpec) -> tuple[float, float]:
    """Suggested default sweep interval (lo, hi) for help text."""
    b = physical_bounds_from_label(spec.label, spec)
    return b.default_lo, b.default_hi


def _clamp_axis_scan_values(st: Any, axis: str, bounds: PhysicalScanBounds) -> None:
    min_key = f"pw_{axis}_min"
    max_key = f"pw_{axis}_max"
    lo = float(st.session_state.get(min_key, bounds.default_lo))
    hi = float(st.session_state.get(max_key, bounds.default_hi))
    lo = max(bounds.min_limit, min(lo, bounds.max_limit))
    hi = max(bounds.min_limit, min(hi, bounds.max_limit))
    if lo >= hi:
        lo, hi = bounds.default_lo, bounds.default_hi
    st.session_state[min_key] = lo
    st.session_state[max_key] = hi


def _apply_axis_physical_bounds(
    st: Any,
    axis: str,
    param_id: str,
    spec: OptParamSpec,
    *,
    reset_defaults: bool,
) -> None:
    bounds = physical_bounds_from_label(spec.label, spec)
    track_key = f"pw_{axis}_bound_param"
    if reset_defaults or st.session_state.get(track_key) != param_id:
        st.session_state[track_key] = param_id
        st.session_state[f"pw_{axis}_min"] = bounds.default_lo
        st.session_state[f"pw_{axis}_max"] = bounds.default_hi
    _clamp_axis_scan_values(st, axis, bounds)


def _make_pw_param_change_callback(axis: str, id_to_spec: dict[str, OptParamSpec]):
    def _on_change() -> None:
        import streamlit as st

        key = "pw_param_x" if axis == "x" else "pw_param_y"
        param_id = st.session_state[key]
        spec = id_to_spec[param_id]
        _apply_axis_physical_bounds(st, axis, param_id, spec, reset_defaults=True)

    return _on_change


def _sync_axis_scan_bounds(st: Any, axis: str, param_id: str, spec: OptParamSpec) -> None:
    """Sync Min/Max when axis parameter changes (rerun without callback)."""
    _apply_axis_physical_bounds(
        st, axis, param_id, spec, reset_defaults=st.session_state.get(f"pw_{axis}_bound_param") != param_id
    )


def _ensure_axis_scan_bounds(st: Any, axis: str, spec: OptParamSpec, param_id: str) -> None:
    min_key = f"pw_{axis}_min"
    if min_key not in st.session_state:
        _apply_axis_physical_bounds(st, axis, param_id, spec, reset_defaults=True)


def default_dw_param_pair(cfg: MaterialInputConfig) -> tuple[str, str]:
    arch = parse_architecture(cfg.packaging_architecture)
    catalog_ids = {s.param_id for s in build_opt_param_catalog(cfg)}
    p1, p2 = DW_DEFAULT_PARAMS[arch]
    if p1 not in catalog_ids:
        p1 = next(iter(catalog_ids), p1)
    if p2 not in catalog_ids or p2 == p1:
        p2 = next((i for i in catalog_ids if i != p1), p1)
    return p1, p2


def param_spec_by_id(param_id: str, cfg: MaterialInputConfig | None = None) -> OptParamSpec:
    for spec in build_opt_param_catalog(cfg):
        if spec.param_id == param_id:
            return spec
    raise KeyError(param_id)


def apply_param_value(cfg: MaterialInputConfig, spec: OptParamSpec, value: float) -> MaterialInputConfig:
    out = copy.deepcopy(cfg)
    if spec.kind == "thickness" and spec.material_key:
        out.thickness_um[spec.material_key] = float(value)
    elif spec.kind == "E_gpa" and spec.material_key:
        out.E_gpa[spec.material_key] = float(value)
    elif spec.kind == "T_ref":
        out.T_ref = float(value)
    elif spec.kind == "emc_alpha1":
        out.emc_alpha1 = float(value)
    elif spec.kind == "emc_alpha2":
        out.emc_alpha2 = float(value)
    elif spec.kind == "rdl_alpha1":
        out.rdl_alpha1 = float(value)
    elif spec.kind == "rdl_alpha2":
        out.rdl_alpha2 = float(value)
    elif spec.kind == "rdl_Tg":
        out.rdl_Tg = float(value)
    return out


def material_config_from_hashable(data: tuple[Any, ...]) -> MaterialInputConfig:
    (
        arch,
        active_items,
        t_items,
        e_items,
        emc_a1,
        emc_a2,
        rdl_a1,
        rdl_a2,
        rdl_tg,
        t_ref,
        meas_t,
        die_frac,
        r_mm,
    ) = data
    return MaterialInputConfig(
        packaging_architecture=arch,
        layer_active=dict(active_items),
        thickness_um=dict(t_items),
        E_gpa=dict(e_items),
        emc_alpha1=emc_a1,
        emc_alpha2=emc_a2,
        rdl_alpha1=rdl_a1,
        rdl_alpha2=rdl_a2,
        rdl_Tg=rdl_tg,
        T_ref=t_ref,
        measurement_T_c=meas_t,
        die_area_fraction=die_frac,
        wafer_radius_mm=r_mm,
    )


def cfg_to_hashable(cfg: MaterialInputConfig) -> tuple[Any, ...]:
    active_items = tuple(sorted(cfg.layer_active.items()))
    t_items = tuple(sorted(cfg.thickness_um.items()))
    e_items = tuple(sorted(cfg.E_gpa.items()))
    return (
        cfg.packaging_architecture,
        active_items,
        t_items,
        e_items,
        cfg.emc_alpha1,
        cfg.emc_alpha2,
        cfg.rdl_alpha1,
        cfg.rdl_alpha2,
        cfg.rdl_Tg,
        cfg.T_ref,
        cfg.measurement_T_c,
        cfg.die_area_fraction,
        cfg.wafer_radius_mm,
    )


def evaluate_warpage_at_step(
    cfg: MaterialInputConfig,
    eval_label: str,
    *,
    metric: MetricKind = "signed_debond",
) -> float:
    """Return warpage at an architecture timeline label (signed edge warpage by default)."""
    snaps = simulate_process_sequence(
        wafer_radius_mm=cfg.wafer_radius_mm,
        T_ref_initial=cfg.T_ref,
        **cfg.to_solver_kwargs(),
    )
    if metric == "abs_max_process":
        return float(max(abs(s.warpage_edge_mm) for s in snaps))
    snap = next((s for s in snaps if s.label == eval_label), snaps[-1])
    if metric == "abs_debond":
        return float(abs(snap.warpage_edge_mm))
    return float(snap.warpage_edge_mm)


def run_design_window_sweep(
    base_cfg: MaterialInputConfig,
    param1_id: str,
    param2_id: str,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n1: int = DEFAULT_GRID_POINTS,
    n2: int = DEFAULT_GRID_POINTS,
    metric: MetricKind = "signed_debond",
    eval_label: str | None = None,
    w_safe_min: float = PROCESS_WINDOW_W_MIN_MM,
    w_safe_max: float = PROCESS_WINDOW_W_MAX_MM,
) -> dict[str, Any]:
    """
    Grid search: linspace + meshgrid over (param X, param Y) → warpage Z.

    Optimal = grid point with minimum |warpage| at eval_step.
    """
    s1 = param_spec_by_id(param1_id, base_cfg)
    s2 = param_spec_by_id(param2_id, base_cfg)
    n1 = max(int(n1), 3)
    n2 = max(int(n2), 3)
    x_lo, x_hi = (min(x_min, x_max), max(x_min, x_max))
    y_lo, y_hi = (min(y_min, y_max), max(y_min, y_max))

    x_vals = np.linspace(x_lo, x_hi, n1)
    y_vals = np.linspace(y_lo, y_hi, n2)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)

    label = eval_label or final_timeline_label(base_cfg.packaging_architecture)
    z_signed = np.zeros_like(X_grid, dtype=float)

    for j in range(n2):
        for i in range(n1):
            v1 = float(X_grid[j, i])
            v2 = float(Y_grid[j, i])
            cfg = apply_param_value(base_cfg, s1, v1)
            cfg = apply_param_value(cfg, s2, v2)
            z_signed[j, i] = evaluate_warpage_at_step(cfg, label, metric=metric)

    z_abs = np.abs(z_signed)
    safe_mask = (z_signed >= w_safe_min) & (z_signed <= w_safe_max)

    flat_idx = int(np.argmin(z_abs))
    j_opt, i_opt = np.unravel_index(flat_idx, z_abs.shape)

    return {
        "x_vals": x_vals,
        "y_vals": y_vals,
        "X_grid": X_grid,
        "Y_grid": Y_grid,
        "z_signed": z_signed,
        "z_abs": z_abs,
        "safe_mask": safe_mask,
        "param1": s1,
        "param2": s2,
        "opt_x": float(x_vals[i_opt]),
        "opt_y": float(y_vals[j_opt]),
        "opt_w_signed": float(z_signed[j_opt, i_opt]),
        "opt_w_abs": float(z_abs[j_opt, i_opt]),
        "safe_fraction": float(np.mean(safe_mask)),
        "metric": metric,
        "eval_label": label,
        "w_safe_min": w_safe_min,
        "w_safe_max": w_safe_max,
        "n1": n1,
        "n2": n2,
    }


def _signed_warpage_contour_limits(z_um: np.ndarray) -> tuple[float, float]:
    """Symmetric z limits about 0 so colormap midpoint = flat (w ≈ 0), in µm."""
    max_abs = float(np.max(np.abs(z_um))) if z_um.size else 0.0
    max_abs = max(max_abs, 50.0)
    return -max_abs, max_abs


def plot_design_window_contour(
    sweep: dict[str, Any],
    *,
    show_safe_region: bool = True,
) -> "go.Figure":
    import plotly.graph_objects as go

    s1: OptParamSpec = sweep["param1"]
    s2: OptParamSpec = sweep["param2"]
    z_signed = sweep["z_signed"]
    z_um = z_signed * UM_PER_MM
    z_abs_um = sweep["z_abs"] * UM_PER_MM
    zmin_c, zmax_c = _signed_warpage_contour_limits(z_um)
    safe_mask = sweep["safe_mask"]
    w_lo = sweep["w_safe_min"]
    w_hi = sweep["w_safe_max"]
    w_lo_um = mm_to_um(w_lo)
    w_hi_um = mm_to_um(w_hi)

    hover = np.empty(z_signed.shape, dtype=object)
    for j in range(z_signed.shape[0]):
        for i in range(z_signed.shape[1]):
            status = "SAFE" if safe_mask[j, i] else "OUT"
            hover[j, i] = (
                f"{s1.label}={sweep['x_vals'][i]:.3g}<br>"
                f"{s2.label}={sweep['y_vals'][j]:.3g}<br>"
                f"Warpage w={z_um[j,i]:.1f} {WARPAGE_UNIT}<br>"
                f"|w|={z_abs_um[j,i]:.1f} {WARPAGE_UNIT}<br>{status}"
            )

    fig = go.Figure()

    if show_safe_region:
        fig.add_trace(
            go.Heatmap(
                x=sweep["x_vals"],
                y=sweep["y_vals"],
                z=np.where(safe_mask, 1.0, 0.0),
                zmin=0,
                zmax=1,
                colorscale=[[0, "rgba(255,255,255,0)"], [1, "rgba(46,125,50,0.35)"]],
                showscale=False,
                hoverinfo="skip",
                name=f"Safe window [{w_lo_um:+.0f}, {w_hi_um:+.0f}] {WARPAGE_UNIT}",
            )
        )

    fig.add_trace(
        go.Contour(
            x=sweep["x_vals"],
            y=sweep["y_vals"],
            z=z_um,
            colorscale="RdBu",
            zmid=0,
            zmin=zmin_c,
            zmax=zmax_c,
            colorbar=dict(title=f"Warpage w ({WARPAGE_UNIT})"),
            contours=dict(coloring="heatmap", showlabels=False),
            hovertext=hover,
            hovertemplate="%{hovertext}<extra></extra>",
            name="Warpage (signed)",
        )
    )

    for level_um, color, dash in (
        (w_hi_um, "#2E7D32", "solid"),
        (w_lo_um, "#2E7D32", "solid"),
    ):
        fig.add_trace(
            go.Contour(
                x=sweep["x_vals"],
                y=sweep["y_vals"],
                z=z_um,
                contours=dict(
                    coloring="none",
                    showlabels=True,
                    labelfont=dict(size=10, color=color),
                    type="constraint",
                    operation="=",
                    value=level_um,
                ),
                line=dict(color=color, width=2, dash=dash),
                showscale=False,
                name=f"w = {level_um:+.0f} {WARPAGE_UNIT}",
                hoverinfo="skip",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[sweep["opt_x"]],
            y=[sweep["opt_y"]],
            mode="markers+text",
            marker=dict(
                size=16,
                color="#D32F2F",
                symbol="star",
                line=dict(width=2, color="#FFEBEE"),
            ),
            text=["Min |w|"],
            textposition="top center",
            textfont=dict(color="#B71C1C", size=11),
            name="Optimal (Min Warpage)",
            hovertemplate=(
                f"<b>最佳解</b><br>{s1.label}=%{{x:.4g}}<br>{s2.label}=%{{y:.4g}}"
                f"<br>w={mm_to_um(sweep['opt_w_signed']):.1f} {WARPAGE_UNIT}"
                f"<br>|w|={mm_to_um(sweep['opt_w_abs']):.1f} {WARPAGE_UNIT}<extra></extra>"
            ),
        )
    )

    step_label = sweep.get("eval_label", "Final")
    title_html = (
        "2D Parameter Optimization &amp; Process Window"
        f"<br><sup>Eval: {step_label} · Green = {w_lo_um:+.0f} to {w_hi_um:+.0f} {WARPAGE_UNIT} · ★ Min |warpage|</sup>"
    )
    fig.update_layout(
        title=dict(text=title_html, x=0.02, xanchor="left"),
        xaxis_title=s1.label,
        yaxis_title=s2.label,
        height=580,
        margin=dict(l=60, r=40, t=95, b=60),
    )
    return fig


def render_design_window_page(
    base_cfg: MaterialInputConfig,
    *,
    sweep_runner: Any | None = None,
) -> None:
    """Streamlit UI: 2D parameter optimization & process window analysis."""
    import streamlit as st

    run_sweep = sweep_runner or _run_cached_sweep

    st.subheader("雙參數最佳化與製程視窗分析")
    st.caption(
        "2D Parameter Optimization & Process Window · "
        f"架構：**{base_cfg.packaging_architecture}** · "
        f"Grid Search → 最終步驟翹曲 · 安全區間 {mm_to_um(PROCESS_WINDOW_W_MIN_MM):+.0f} ~ "
        f"{mm_to_um(PROCESS_WINDOW_W_MAX_MM):+.0f} {WARPAGE_UNIT}"
    )

    catalog = build_opt_param_catalog(base_cfg)
    if len(catalog) < 2:
        st.warning("可掃描參數不足，請切換封裝架構或確認側邊欄層級設定。")
        return

    id_to_label = {s.param_id: s.label for s in catalog}
    id_to_spec = {s.param_id: s for s in catalog}
    all_ids = [s.param_id for s in catalog]
    def_p1, def_p2 = default_dw_param_pair(base_cfg)

    if st.session_state.get("pw_catalog_arch") != base_cfg.packaging_architecture:
        st.session_state["pw_catalog_arch"] = base_cfg.packaging_architecture
        for k in (
            "pw_x_bound_param",
            "pw_y_bound_param",
            "pw_x_min",
            "pw_x_max",
            "pw_y_min",
            "pw_y_max",
        ):
            st.session_state.pop(k, None)

    st.markdown("##### ① 選擇掃描參數 (X / Y 軸)")
    c1, c2 = st.columns(2)
    with c1:
        param1_id = st.selectbox(
            "X 軸參數",
            all_ids,
            index=all_ids.index(def_p1) if def_p1 in all_ids else 0,
            format_func=lambda k: id_to_label[k],
            key="pw_param_x",
            on_change=_make_pw_param_change_callback("x", id_to_spec),
        )
    with c2:
        p2_options = [k for k in all_ids if k != param1_id]
        param2_id = st.selectbox(
            "Y 軸參數",
            p2_options,
            index=p2_options.index(def_p2) if def_p2 in p2_options else 0,
            format_func=lambda k: id_to_label[k],
            key="pw_param_y",
            on_change=_make_pw_param_change_callback("y", id_to_spec),
        )

    s1 = id_to_spec[param1_id]
    s2 = id_to_spec[param2_id]

    _ensure_axis_scan_bounds(st, "x", s1, param1_id)
    _ensure_axis_scan_bounds(st, "y", s2, param2_id)
    _sync_axis_scan_bounds(st, "x", param1_id, s1)
    _sync_axis_scan_bounds(st, "y", param2_id, s2)

    bx = physical_bounds_from_label(s1.label, s1)
    by = physical_bounds_from_label(s2.label, s2)
    _clamp_axis_scan_values(st, "x", bx)
    _clamp_axis_scan_values(st, "y", by)

    st.divider()
    st.markdown("##### ② 掃描範圍與網格密度")
    rx1, rx2, rx3 = st.columns(3)
    with rx1:
        x_min = st.number_input(
            "X Min",
            min_value=float(bx.min_limit),
            max_value=float(bx.max_limit),
            step=float(s1.step),
            key="pw_x_min",
            help=(
                f"{s1.label} · 物理允許 [{bx.min_limit:g}, {bx.max_limit:g}] · "
                f"建議掃描 {bx.default_lo:g} ~ {bx.default_hi:g}"
            ),
        )
    with rx2:
        x_max = st.number_input(
            "X Max",
            min_value=float(bx.min_limit),
            max_value=float(bx.max_limit),
            step=float(s1.step),
            key="pw_x_max",
            help=f"{s1.label} · 物理允許 [{bx.min_limit:g}, {bx.max_limit:g}]",
        )
    with rx3:
        n1 = st.number_input("X 切割點數", min_value=3, max_value=40, value=DEFAULT_GRID_POINTS, key="pw_n1")

    st.write("")
    ry1, ry2, ry3 = st.columns(3)
    with ry1:
        y_min = st.number_input(
            "Y Min",
            min_value=float(by.min_limit),
            max_value=float(by.max_limit),
            step=float(s2.step),
            key="pw_y_min",
            help=(
                f"{s2.label} · 物理允許 [{by.min_limit:g}, {by.max_limit:g}] · "
                f"建議掃描 {by.default_lo:g} ~ {by.default_hi:g}"
            ),
        )
    with ry2:
        y_max = st.number_input(
            "Y Max",
            min_value=float(by.min_limit),
            max_value=float(by.max_limit),
            step=float(s2.step),
            key="pw_y_max",
            help=f"{s2.label} · 物理允許 [{by.min_limit:g}, {by.max_limit:g}]",
        )
    with ry3:
        n2 = st.number_input("Y 切割點數", min_value=3, max_value=40, value=DEFAULT_GRID_POINTS, key="pw_n2")

    total = int(n1) * int(n2)
    if x_min >= x_max or y_min >= y_max:
        st.error("請確認 Min < Max（X 與 Y 軸皆需成立）。")
        return
    if total > 900:
        st.warning(f"網格 {n1}×{n2} = {total} 點，計算時間可能較長。")

    st.divider()
    st.markdown("##### ③ 評估設定")
    ec1, ec2, ec3 = st.columns([1.2, 1.0, 1.0])
    with ec1:
        timeline_labels = timeline_labels_for_architecture(base_cfg.packaging_architecture)
        default_lbl = final_timeline_label(base_cfg.packaging_architecture)
        step_name = st.selectbox(
            "評估製程步驟（翹曲取該步）",
            timeline_labels,
            index=timeline_labels.index(default_lbl) if default_lbl in timeline_labels else -1,
            key=f"pw_step_{base_cfg.packaging_architecture}",
            help="依目前封裝架構的製程時間軸",
        )
    with ec2:
        metric = st.selectbox(
            "翹曲指標",
            options=["signed_debond", "abs_debond", "abs_max_process"],
            format_func=lambda m: {
                "signed_debond": f"Signed warpage（製程視窗 ±{mm_to_um(PROCESS_WINDOW_W_MAX_MM):.0f} {WARPAGE_UNIT}）",
                "abs_debond": "|warpage| 最小化",
                "abs_max_process": "全製程 max |warpage|",
            }[m],
            key="pw_metric",
        )
    with ec3:
        st.metric("網格點數", f"{total}")

    st.divider()
    run = st.button("執行 Grid Search / 繪製製程視窗", type="primary", key="pw_run_btn")

    if not run and "pw_last_sweep" not in st.session_state:
        st.info("設定 X/Y 參數與範圍後，按 **執行 Grid Search** 產生 2D 等高線製程視窗圖。")
        return

    if run:
        cfg_hash = cfg_to_hashable(base_cfg)
        with st.spinner(f"Grid Search {n1}×{n2} = {total} 組合…"):
            sweep = run_sweep(
                cfg_hash,
                param1_id,
                param2_id,
                float(x_min),
                float(x_max),
                float(y_min),
                float(y_max),
                int(n1),
                int(n2),
                metric,
                step_name,
            )
        st.session_state.pw_last_sweep = sweep
        st.session_state.dw_last_sweep = sweep

    sweep = st.session_state.get("pw_last_sweep") or st.session_state.get("dw_last_sweep")
    if not sweep:
        return

    st.markdown("##### ④ 最佳解 (Optimal Solution)")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Min |warpage|", f"{mm_to_um(sweep['opt_w_abs']):.1f} {WARPAGE_UNIT}")
    m2.metric("w @ optimum (signed)", f"{mm_to_um(sweep['opt_w_signed']):+.1f} {WARPAGE_UNIT}")
    m3.metric(s1.label.split("—")[0].strip(), f"{sweep['opt_x']:.4g}")
    m4.metric(s2.label.split("—")[0].strip(), f"{sweep['opt_y']:.4g}")
    m5.metric("安全區占比", f"{100 * sweep['safe_fraction']:.1f} %")

    st.plotly_chart(
        plot_design_window_contour(sweep),
        use_container_width=True,
        config={"scrollZoom": True},
        key="process_window_contour",
    )


def _run_cached_sweep(
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


# Backward-compatible alias
render_process_window_page = render_design_window_page
