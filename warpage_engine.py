"""2.5D advanced packaging warpage engine: multi-layer plate theory with thermo-elastic stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from constants import (
    DEBOND_RELEASE_RATIO,
    E_MIN_SOLDER_GPA,
    E_REF_PA_FLOOR,
    EMC_E_MIN_FRACTION,
    EMC_MOMENT_FACTOR_WITH_INTERPOSER,
    EMC_T_SOFTEN_END_C,
    EMC_T_SOFTEN_START_C,
    INTERPOSER_MOMENT_GAIN,
    SOLDER_MOMENT_FACTOR,
    LayerStack,
    LayerType,
    MaterialProps,
    PROCESS_TEMPERATURES,
    PROCESS_T_REF_C,
    RESIDUAL_MOMENT_GAIN,
    STEP_BUILDUP_GAIN,
    STRESS_FREE_RELAXATION,
    WARPAGE_CALIBRATION,
    ProcessStep,
    UM_PER_MM,
    WAFER_RADIUS_MM,
    WARPAGE_CRITICAL_MM,
    WARPAGE_CRITICAL_UM,
    WARPAGE_UNIT,
    build_stack_for_step,
)
from composite_layer import (
    MOLDING_COMPOSITE_KEY,
    composite_alpha_ppm,
    composite_e_gpa,
)
from packaging_arch import (
    FOWLP_COMPLIANT_LAYER_TYPES,
    INTERPOSER_NA_STIFFNESS_WEIGHT,
    PackagingArchitecture,
    parse_architecture,
    CURING_STATION_LABELS,
    process_temperature_for_step,
    thermal_delta_for_timeline_step,
    timeline_layer_active,
    timeline_step_by_label,
    timeline_steps_for_architecture,
)


@dataclass
class LayerState:
    index: int
    material: MaterialProps
    thickness_m: float
    z_centroid_m: float
    z_from_na_m: float
    sigma_pa: float
    alpha_eff_ppm: float
    E_eff_pa: float


@dataclass
class WarpageResult:
    kappa_1_per_m: float
    warpage_center_mm: float
    warpage_edge_mm: float
    neutral_axis_m: float
    total_thickness_m: float
    layers: list[LayerState]
    carrier_constrained: bool
    stress_map_mpa: np.ndarray
    r_grid_mm: np.ndarray
    theta_grid: np.ndarray


@dataclass
class ProcessStepSnapshot:
    """One packaging process step after sequential (residual-stress) analysis."""

    step: ProcessStep
    label: str
    temperature_c: float
    T_stress_free_c: float
    delta_T_c: float
    result: WarpageResult
    warpage_edge_mm: float  # signed @ edge, center-referenced: w(R) = κR²/2
    warpage_center_mm: float  # always 0 (center reference)
    shape_label: str  # Bow / Crown / Flat


# --- SI unit helpers (inputs → SI, outputs → display) ---
def um_to_m(t_um: float) -> float:
    return float(t_um) * 1e-6


def gpa_to_pa(E_gpa: float) -> float:
    return float(E_gpa) * 1e9


def ppm_to_per_k(alpha_ppm: float) -> float:
    return float(alpha_ppm) * 1e-6


def mm_to_m(r_mm: float) -> float:
    return float(r_mm) * 1e-3


def m_to_mm(w_m: float) -> float:
    return float(w_m) * 1e3


def warpage_at_radius_m(kappa_1_per_m: float, r_mm: float) -> float:
    """Center-referenced displacement w [m]: w = κ r² / 2, r converted from mm to m."""
    r_m = mm_to_m(r_mm)
    return 0.5 * float(kappa_1_per_m) * r_m**2


def warpage_at_radius_mm(kappa_1_per_m: float, r_mm: float) -> float:
    """Display helper: w [mm] from κ [1/m] and r [mm]."""
    return m_to_mm(warpage_at_radius_m(kappa_1_per_m, r_mm))


def kappa_from_edge_warpage_m(w_edge_m: float, r_mm: float) -> float:
    """Recover κ [1/m] from edge displacement w(R) [m] (center-referenced)."""
    r_m = mm_to_m(r_mm)
    return 2.0 * float(w_edge_m) / max(r_m**2, 1e-18)


def kappa_from_edge_warpage_mm(w_edge_mm: float, r_mm: float) -> float:
    """Recover κ [1/m] from edge displacement w(R) [mm]."""
    return kappa_from_edge_warpage_m(mm_to_m(w_edge_mm), r_mm)


def displacement_mm(kappa_1_per_m: float, r_mm: float) -> float:
    """Alias: signed warpage w(r) in mm."""
    return warpage_at_radius_mm(kappa_1_per_m, r_mm)


def warpage_shape_label(kappa: float) -> str:
    """Bow (κ>0, edge up) / Crown (κ<0, center up relative to edge)."""
    if abs(kappa) < 1e-14:
        return "Flat"
    return "Bow" if kappa > 0 else "Crown"


def _diverging_z_limits(z_mm: np.ndarray) -> tuple[float, float]:
    finite = z_mm[np.isfinite(z_mm)]
    zmax = float(np.max(np.abs(finite))) if finite.size else 1e-6
    zmax = max(zmax, 1e-9)
    return -zmax, zmax


def effective_alpha(
    material: MaterialProps,
    T_c: float,
    alpha_override: float | None = None,
    alpha2_override: float | None = None,
    Tg_override: float | None = None,
) -> float:
    """Piecewise CTE: alpha1 below Tg, alpha2 above Tg (RDL / user-defined)."""
    if alpha_override is not None:
        a1 = alpha_override
    else:
        a1 = material.alpha_ppm

    if material.alpha2_ppm is None and alpha2_override is None:
        return a1

    a2 = alpha2_override if alpha2_override is not None else material.alpha2_ppm
    Tg = Tg_override if Tg_override is not None else material.Tg_c
    if Tg is None:
        return a1
    return a1 if T_c < Tg else a2


def effective_emc_modulus(E_gpa: float, T_c: float) -> float:
    """EMC modulus decay above ~150°C (rubbery / gel-like behavior)."""
    if T_c <= EMC_T_SOFTEN_START_C:
        return E_gpa
    if T_c >= EMC_T_SOFTEN_END_C:
        return E_gpa * EMC_E_MIN_FRACTION
    t = (T_c - EMC_T_SOFTEN_START_C) / (EMC_T_SOFTEN_END_C - EMC_T_SOFTEN_START_C)
    # Smooth logistic-style decay
    decay = EMC_E_MIN_FRACTION + (1.0 - EMC_E_MIN_FRACTION) * (1.0 - t) ** 2
    return E_gpa * max(decay, EMC_E_MIN_FRACTION)


def effective_young_modulus(
    material: MaterialProps,
    T_c: float,
    E_override_gpa: float | None = None,
) -> float:
    E = E_override_gpa if E_override_gpa is not None else material.E_gpa
    if material.layer_type.value == "emc":
        return effective_emc_modulus(E, T_c)
    if material.layer_type == LayerType.SOLDER:
        return max(float(E), E_MIN_SOLDER_GPA)
    return E


def find_neutral_axis(
    layers: list[tuple[MaterialProps, float, float]],
    *,
    stiffness_weight: float = 1.0,
) -> float:
    """
    Neutral axis from centroid of axial stiffness (EA).
    layers: (material, thickness_m, E_eff_Pa)
    Returns z_na from bottom surface (m).
    """
    z = 0.0
    sum_ea = 0.0
    sum_eaz = 0.0
    for mat, h_m, E_pa in layers:
        z_mid = z + h_m / 2.0
        ea = E_pa * h_m * stiffness_weight
        sum_ea += ea
        sum_eaz += ea * z_mid
        z += h_m
    if sum_ea < 1e-30:
        return 0.0
    return sum_eaz / sum_ea


def flexural_rigidity_pa_m3(E_pa: float, h_m: float, nu: float) -> float:
    """D = E h³ / (12 (1 − ν²))  [N·m] (per unit width in multi-layer effective sense)."""
    return E_pa * h_m**3 / (12.0 * max(1.0 - nu**2, 1e-9))


def find_neutral_axis_for_architecture(
    layers: list[tuple[MaterialProps, float, float, str]],
    packaging_architecture: str,
) -> float:
    """
    Architecture-specific z_NA:
    - FOWLP: only active compliant/metal layers (EMC, RDL, Carrier).
    - CoWoS-S: full active stack with high stiffness weight on Si interposer.
    """
    arch = parse_architecture(packaging_architecture)
    if arch == PackagingArchitecture.FOWLP_INFO:
        compliant = [
            (mat, h_m, E_pa)
            for mat, h_m, E_pa, _key in layers
            if mat.layer_type.value in FOWLP_COMPLIANT_LAYER_TYPES
        ]
        if compliant:
            return find_neutral_axis(compliant)
        return find_neutral_axis([(m, h_m, E_pa) for m, h_m, E_pa, _ in layers])

    z = 0.0
    sum_ea = 0.0
    sum_eaz = 0.0
    for mat, h_m, E_pa, _key in layers:
        z_mid = z + h_m / 2.0
        weight = 0.0 if mat.layer_type == LayerType.SOLDER else (
            INTERPOSER_NA_STIFFNESS_WEIGHT
            if mat.layer_type == LayerType.INTERPOSER
            else 1.0
        )
        ea = E_pa * h_m * weight
        sum_ea += ea
        sum_eaz += ea * z_mid
        z += h_m
    if sum_ea < 1e-30:
        return 0.0
    return sum_eaz / sum_ea


def reference_alpha_sub_ppm(
    props: list[tuple[MaterialProps, float, float, float, float, str]],
    packaging_architecture: str,
    *,
    default_ppm: float = 2.6,
) -> float:
    """CTE reference layer for Δα · ΔT (architecture-dependent)."""
    arch = parse_architecture(packaging_architecture)
    if arch == PackagingArchitecture.FOWLP_INFO:
        for mat, _h, _e, alpha, _z, _key in props:
            if mat.layer_type == LayerType.CARRIER:
                return alpha * 1e6
        for mat, _h, _e, alpha, _z, _key in props:
            if mat.layer_type == LayerType.EMC:
                return alpha * 1e6
        return 5.0
    for mat, _h, _e, alpha, _z, _key in props:
        if mat.layer_type == LayerType.INTERPOSER:
            return alpha * 1e6
    for mat, _h, _e, alpha, _z, _key in props:
        if mat.layer_type == LayerType.SILICON:
            return alpha * 1e6
    return default_ppm


def layer_thermal_stress_pa(
    E_pa: float,
    nu: float,
    alpha_layer: float,
    alpha_sub: float,
    delta_T: float,
) -> float:
    """sigma_i = E/(1-nu) * (alpha_i - alpha_sub) * dT"""
    return (E_pa / (1.0 - nu)) * (alpha_layer - alpha_sub) * delta_T


def calculate_warpage(
    layer_stacks: list[LayerStack],
    T_c: float,
    T_ref_c: float = 25.0,
    *,
    measurement_T_c: float | None = None,
    evaluation_T_c: float | None = None,  # legacy alias
    stress_free_T_c: float | None = None,
    delta_t_mode: str = "process",
    thermal_delta_T_c: float | None = None,
    die_area_fraction: float = 0.30,
    wafer_radius_mm: float = WAFER_RADIUS_MM,
    carrier_constrained: bool = False,
    alpha_sub_ppm: float = 2.6,
    emc_alpha1: float | None = None,
    emc_alpha2: float | None = None,
    rdl_alpha1: float | None = None,
    rdl_alpha2: float | None = None,
    rdl_Tg: float | None = None,
    emc_E_gpa: float | None = None,
    layer_E_gpa: dict[str, float] | None = None,
    moment_multiplier: float = 1.0,
    packaging_architecture: str = PackagingArchitecture.FOWLP_INFO.value,
) -> WarpageResult:
    """
    Multi-layer plate warpage (strict SI internally):

    - t: µm → m, E: GPa → Pa, α: ppm/°C → 1/K, r: mm → m
    - M [N/m] = Σ σᵢ hᵢ zᵢ (zᵢ from z_NA); D [N·m] = E_ref h³ / (12(1−ν²))
    - κ [1/m] = M / D; w [m] = κ r² / 2; UI displays w in µm
    """
    active = [ls for ls in layer_stacks if ls.include]
    if not active:
        return WarpageResult(
            kappa_1_per_m=0.0,
            warpage_center_mm=0.0,
            warpage_edge_mm=0.0,
            neutral_axis_m=0.0,
            total_thickness_m=0.0,
            layers=[],
            carrier_constrained=carrier_constrained,
            stress_map_mpa=np.zeros((64, 64)),
            r_grid_mm=np.zeros((64, 64)),
            theta_grid=np.zeros((64, 64)),
        )

    T_sf_c = stress_free_T_c if stress_free_T_c is not None else T_ref_c
    T_meas = measurement_T_c if measurement_T_c is not None else evaluation_T_c
    if thermal_delta_T_c is not None:
        T_props_c = T_c
        delta_T = thermal_delta_T_c
    elif delta_t_mode == "measurement" and T_meas is not None:
        T_props_c = T_meas
        delta_T = T_meas - T_sf_c
    else:
        T_props_c = T_c
        delta_T = T_c - T_sf_c

    # Build effective properties per layer
    props: list[tuple[MaterialProps, float, float, float, float, str]] = []
    z_bottom = 0.0
    for ls in active:
        mat = ls.material
        h_m = um_to_m(ls.thickness_um)
        if ls.material_key == MOLDING_COMPOSITE_KEY:
            E_gpa = composite_e_gpa(
                T_props_c,
                die_area_fraction,
                layer_E_gpa=layer_E_gpa,
                emc_E_gpa=emc_E_gpa,
                effective_young_modulus=effective_young_modulus,
            )
            a_ppm = composite_alpha_ppm(
                T_props_c,
                die_area_fraction,
                emc_alpha1=emc_alpha1,
                emc_alpha2=emc_alpha2,
                effective_alpha=effective_alpha,
            )
        else:
            E_override = None
            if layer_E_gpa and ls.material_key in layer_E_gpa:
                E_override = layer_E_gpa[ls.material_key]
            elif mat.layer_type.value == "emc" and emc_E_gpa is not None:
                E_override = emc_E_gpa
            E_gpa = effective_young_modulus(mat, T_props_c, E_override_gpa=E_override)

            if mat.layer_type.value == "emc":
                a_ppm = effective_alpha(mat, T_props_c, emc_alpha1, emc_alpha2, None)
            elif mat.layer_type.value == "rdl":
                a_ppm = effective_alpha(mat, T_props_c, rdl_alpha1, rdl_alpha2, rdl_Tg)
            else:
                a_ppm = effective_alpha(mat, T_props_c)

        E_pa = gpa_to_pa(E_gpa)

        alpha_per_k = ppm_to_per_k(a_ppm)
        props.append((mat, h_m, E_pa, alpha_per_k, z_bottom + h_m / 2.0, ls.material_key))
        z_bottom += h_m

    total_h_m = z_bottom
    alpha_sub_ppm_eff = reference_alpha_sub_ppm(props, packaging_architecture, default_ppm=alpha_sub_ppm)
    alpha_sub = ppm_to_per_k(alpha_sub_ppm_eff)
    na_layers = [(p[0], p[1], p[2], p[5]) for p in props]
    z_na_m = find_neutral_axis_for_architecture(na_layers, packaging_architecture)

    arch = parse_architecture(packaging_architecture)
    if arch == PackagingArchitecture.FOWLP_INFO:
        ref_candidates = [
            p for p in props if p[0].layer_type in (LayerType.EMC, LayerType.CARRIER)
        ]
        sub = ref_candidates[0] if ref_candidates else props[0]
    else:
        sub = next((p for p in props if p[0].layer_type == LayerType.INTERPOSER), props[0])
    if sub[0].layer_type == LayerType.SOLDER:
        sub = next((p for p in props if p[0].layer_type != LayerType.SOLDER), sub)
    E_ref = max(sub[2], E_REF_PA_FLOOR)
    nu_ref = sub[0].nu

    has_interposer = any(p[0].layer_type == LayerType.INTERPOSER for p in props)
    moment_sum = 0.0
    layer_states: list[LayerState] = []

    for i, (mat, h_m, E_pa, alpha_per_k, z_c, _mkey) in enumerate(props):
        z_i = z_c - z_na_m
        sigma_pa = layer_thermal_stress_pa(E_pa, mat.nu, alpha_per_k, alpha_sub, delta_T)

        # Carrier constraint: glass carries share of moment until debond
        if carrier_constrained and mat.layer_type.value == "carrier":
            sigma_pa *= 0.35  # partial load transfer to carrier
        elif not carrier_constrained and mat.layer_type.value == "carrier":
            continue

        # Bending moment per unit width [N] (σ in Pa, h and z in m)
        moment_term = sigma_pa * h_m * z_i
        if mat.layer_type == LayerType.INTERPOSER:
            moment_term *= INTERPOSER_MOMENT_GAIN
        elif mat.layer_type == LayerType.EMC and has_interposer:
            moment_term *= EMC_MOMENT_FACTOR_WITH_INTERPOSER
        elif mat.layer_type == LayerType.SOLDER:
            moment_term *= SOLDER_MOMENT_FACTOR
        moment_sum += moment_term

        a_ppm_out = alpha_per_k * 1e6
        layer_states.append(
            LayerState(
                index=i,
                material=mat,
                thickness_m=h_m,
                z_centroid_m=z_c,
                z_from_na_m=z_i,
                sigma_pa=sigma_pa,
                alpha_eff_ppm=a_ppm_out,
                E_eff_pa=E_pa,
            )
        )

    moment_sum *= moment_multiplier

    h_m_stack = max(total_h_m, 1e-9)
    D_eff = flexural_rigidity_pa_m3(E_ref, h_m_stack, nu_ref)
    kappa_1_per_m = moment_sum / max(D_eff, 1e-30)

    w_edge_mm = warpage_at_radius_mm(kappa_1_per_m, wafer_radius_mm)
    w_center_mm = 0.0

    stress_map, r_grid, theta_grid = _build_stress_map(
        kappa_1_per_m, layer_states, wafer_radius_mm
    )

    return WarpageResult(
        kappa_1_per_m=kappa_1_per_m,
        warpage_center_mm=w_center_mm,
        warpage_edge_mm=w_edge_mm,
        neutral_axis_m=z_na_m,
        total_thickness_m=total_h_m,
        layers=layer_states,
        carrier_constrained=carrier_constrained,
        stress_map_mpa=stress_map,
        r_grid_mm=r_grid,
        theta_grid=theta_grid,
    )


def _build_stress_map(
    kappa: float,
    layers: list[LayerState],
    wafer_radius_mm: float,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate in-plane stress on top surface from bending + layer residual."""
    r = np.linspace(0, wafer_radius_mm, n)
    theta = np.linspace(0, 2 * np.pi, n)
    R, Theta = np.meshgrid(r, theta)

    base_stress = sum(ls.sigma_pa for ls in layers) / max(len(layers), 1) / 1e6
    # Radial variation from plate bending: sigma ~ E * z * kappa * r
    z_top = max((ls.z_from_na_m for ls in layers), default=0.0)
    E_avg = np.mean([ls.E_eff_pa for ls in layers]) if layers else 1e9
    # σ_bend ≈ E·z·κ [Pa] → MPa
    bend_factor = E_avg * z_top * kappa / 1e6

    stress = base_stress + bend_factor * (R / max(wafer_radius_mm, 1e-6)) ** 2
    # Azimuthal asymmetry from die shadow (simplified)
    stress += 0.15 * base_stress * np.cos(4 * Theta) * (R / wafer_radius_mm)

    return stress, R, Theta


def _scale_result(res: WarpageResult, scale: float) -> WarpageResult:
    """Scale κ and warpage amplitudes (package-level calibration)."""
    return WarpageResult(
        kappa_1_per_m=res.kappa_1_per_m * scale,
        warpage_center_mm=res.warpage_center_mm * scale,
        warpage_edge_mm=res.warpage_edge_mm * scale,
        neutral_axis_m=res.neutral_axis_m,
        total_thickness_m=res.total_thickness_m,
        layers=res.layers,
        carrier_constrained=res.carrier_constrained,
        stress_map_mpa=res.stress_map_mpa * scale,
        r_grid_mm=res.r_grid_mm,
        theta_grid=res.theta_grid,
    )


def _result_from_characteristic(
    template: WarpageResult,
    w_edge_mm: float,
    wafer_radius_mm: float,
) -> WarpageResult:
    """Rebuild κ [1/m] from signed edge warpage w [mm] (center-referenced)."""
    kappa = kappa_from_edge_warpage_mm(w_edge_mm, wafer_radius_mm)
    stress_map, r_grid, theta_grid = _build_stress_map(
        kappa, template.layers, wafer_radius_mm
    )
    return WarpageResult(
        kappa_1_per_m=kappa,
        warpage_center_mm=0.0,
        warpage_edge_mm=w_edge_mm,
        neutral_axis_m=template.neutral_axis_m,
        total_thickness_m=template.total_thickness_m,
        layers=template.layers,
        carrier_constrained=template.carrier_constrained,
        stress_map_mpa=stress_map,
        r_grid_mm=r_grid,
        theta_grid=theta_grid,
    )


def characteristic_warpage_mm(kappa_1_per_m: float, radius_mm: float) -> float:
    """Process-curve scalar: signed warpage at wafer edge (center-referenced)."""
    return displacement_mm(kappa_1_per_m, radius_mm)


def simulate_process_sequence(
    wafer_radius_mm: float = WAFER_RADIUS_MM,
    T_profile: dict[ProcessStep, float] | None = None,
    T_ref_initial: float = PROCESS_T_REF_C,
    **kwargs,
) -> list[ProcessStepSnapshot]:
    """
    Paper-style sequential 2.5D packaging warpage:
    - Evolving stress-free temperature (residual stress after cool/hold).
    - Moment memory across steps (not reset when T_step == T_ref).
    - Debonding: carrier constraint release → warpage jump.
    - Dynamic thermal history: ΔT = 0 until mold cure; then ΔT = Process T − stress-free T.
    """
    T_sf = T_ref_initial
    moment_memory = 0.0
    last_constrained_w: float = 0.0
    is_cured = False
    snapshots: list[ProcessStepSnapshot] = []

    substrate_thickness_um = kwargs.pop("substrate_thickness_um", None)
    layer_thickness_um = kwargs.pop("layer_thickness_um", None)
    layer_E_gpa = kwargs.pop("layer_E_gpa", None)
    kwargs.pop("layer_active", None)  # timeline uses per-step presets only
    kwargs.pop("measurement_T_c", None)
    kwargs.pop("evaluation_T_c", None)
    die_area_fraction = kwargs.pop("die_area_fraction", 0.30)
    stress_free_T_c = kwargs.pop("stress_free_T_c", T_ref_initial)
    packaging_architecture = kwargs.pop(
        "packaging_architecture",
        PackagingArchitecture.FOWLP_INFO.value,
    )
    kwargs.pop("warpage_reference", None)  # legacy; center-referenced only
    timeline = timeline_steps_for_architecture(packaging_architecture)

    for step_idx, tstep in enumerate(timeline):
        layer_active = timeline_layer_active(tstep, packaging_architecture)
        layers, constrained = build_stack_for_step(
            tstep.internal_step,
            packaging_architecture=packaging_architecture,
            substrate_thickness_um=substrate_thickness_um,
            layer_thickness_um=layer_thickness_um,
            layer_active=layer_active,
        )
        if tstep.carrier_constrained is not None:
            constrained = tstep.carrier_constrained

        T_proc = process_temperature_for_step(tstep, stress_free_T_c, T_profile)
        if tstep.label in CURING_STATION_LABELS:
            is_cured = True
        delta_T_step = thermal_delta_for_timeline_step(is_cured, T_proc, stress_free_T_c)

        moment_mult = (1.0 + RESIDUAL_MOMENT_GAIN * moment_memory) * (
            1.0 + STEP_BUILDUP_GAIN * step_idx
        )

        raw = calculate_warpage(
            layers,
            T_c=T_proc,
            T_ref_c=T_sf,
            stress_free_T_c=stress_free_T_c,
            delta_t_mode="process",
            thermal_delta_T_c=delta_T_step,
            die_area_fraction=die_area_fraction,
            wafer_radius_mm=wafer_radius_mm,
            carrier_constrained=constrained,
            moment_multiplier=moment_mult,
            layer_E_gpa=layer_E_gpa,
            packaging_architecture=packaging_architecture,
            **kwargs,
        )
        res = raw
        if WARPAGE_CALIBRATION != 1.0:
            res = _scale_result(raw, WARPAGE_CALIBRATION)

        w_char = res.warpage_edge_mm

        if tstep.release_carrier:
            mold_label = "EMC Molding" if "CoWoS" in packaging_architecture else "Molding"
            mold_tstep = timeline_step_by_label(packaging_architecture, mold_label)
            if mold_tstep is not None:
                la_mold = timeline_layer_active(mold_tstep, packaging_architecture)
                layers_cool, _ = build_stack_for_step(
                    mold_tstep.internal_step,
                    packaging_architecture=packaging_architecture,
                    substrate_thickness_um=substrate_thickness_um,
                    layer_thickness_um=layer_thickness_um,
                    layer_active=la_mold,
                )
                cool = calculate_warpage(
                    layers_cool,
                    T_c=T_proc,
                    T_ref_c=T_sf,
                    stress_free_T_c=stress_free_T_c,
                    delta_t_mode="process",
                    thermal_delta_T_c=delta_T_step,
                    die_area_fraction=die_area_fraction,
                    wafer_radius_mm=wafer_radius_mm,
                    carrier_constrained=False,
                    moment_multiplier=moment_mult,
                    layer_E_gpa=layer_E_gpa,
                    packaging_architecture=packaging_architecture,
                    **kwargs,
                )
                if abs(cool.warpage_edge_mm) >= abs(w_char):
                    res = cool
                    w_char = cool.warpage_edge_mm

        if tstep.release_carrier or tstep.internal_step == ProcessStep.DEBONDING:
            sign = 1.0 if last_constrained_w >= 0 else -1.0
            if last_constrained_w == 0.0:
                sign = 1.0 if w_char >= 0 else -1.0
            w_mag = max(abs(w_char), abs(last_constrained_w) * DEBOND_RELEASE_RATIO)
            w_target_mm = sign * w_mag
            res = _result_from_characteristic(res, w_target_mm, wafer_radius_mm)
            w_char = res.warpage_edge_mm

        w_edge = res.warpage_edge_mm
        w_center = 0.0

        if constrained and abs(w_char) > 1e-9:
            last_constrained_w = w_char

        moment_memory = 0.88 * moment_memory + abs(delta_T_step) / 120.0

        snapshots.append(
            ProcessStepSnapshot(
                step=tstep.internal_step,
                label=tstep.label,
                temperature_c=T_proc,
                T_stress_free_c=stress_free_T_c,
                delta_T_c=delta_T_step,
                result=res,
                warpage_edge_mm=w_edge,
                warpage_center_mm=w_center,
                shape_label=warpage_shape_label(res.kappa_1_per_m),
            )
        )

        T_sf = T_sf + STRESS_FREE_RELAXATION * (T_proc - T_sf)

    return snapshots


def simulate_process_curve(
    T_profile: dict[ProcessStep, float] | None = None,
    **kwargs,
) -> tuple[list[str], list[float], list[float]]:
    """Legacy API: labels, signed characteristic warpages (mm), temperatures."""
    snaps = simulate_process_sequence(T_profile=T_profile, **kwargs)
    warpages = [s.warpage_edge_mm for s in snaps]
    return ([s.label for s in snaps], warpages, [s.temperature_c for s in snaps])


def warpage_profile_radial(
    result: WarpageResult,
    n_points: int = 100,
    radius_mm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Signed radial profile w(r) = κ r² / 2 (center-referenced)."""
    r_max = radius_mm if radius_mm is not None else WAFER_RADIUS_MM
    r_mm = np.linspace(0, r_max, n_points)
    w_mm = np.array([displacement_mm(result.kappa_1_per_m, r) for r in r_mm])
    return r_mm, w_mm


def warpage_display_unit(peak_mm: float) -> tuple[float, str]:
    """Scale internal mm warpage to micrometers for all charts."""
    _ = peak_mm
    return UM_PER_MM, WARPAGE_UNIT


def padded_axis_range(
    values: np.ndarray,
    *,
    floor: float = 0.0,
    pad_frac: float = 0.12,
    min_span: float | None = None,
) -> list[float]:
    """Data-driven axis limits (no fixed paper-scale bounds)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        span = min_span or 1e-6
        return [floor, floor + span]

    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    if v_min == v_max:
        span = max(abs(v_max) * pad_frac, min_span or 1e-9, 1e-12)
        v_min -= span
        v_max += span
    else:
        span = (v_max - v_min) * pad_frac
        v_min -= span
        v_max += span

    if min_span is not None and (v_max - v_min) < min_span:
        mid = 0.5 * (v_max + v_min)
        half = min_span / 2.0
        v_min, v_max = mid - half, mid + half

    v_min = min(v_min, floor)
    v_max = max(v_max, floor + (min_span or 0.0))
    return [v_min, v_max]


def wafer_warpage_grid_mm(
    kappa_1_per_m: float,
    radius_mm: float,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Signed warpage on polar grid (mm), center-referenced."""
    r = np.linspace(0, radius_mm, n)
    theta = np.linspace(0, 2 * np.pi, n)
    R_mm, Theta = np.meshgrid(r, theta)
    Z_mm = np.vectorize(lambda rr: displacement_mm(kappa_1_per_m, rr))(R_mm)
    X_mm = R_mm * np.cos(Theta)
    Y_mm = R_mm * np.sin(Theta)
    return X_mm, Y_mm, Z_mm, R_mm


def plot_wafer_surface(
    kappa_1_per_m: float,
    radius_mm: float,
    n: int = 80,
    title: str = "Wafer Warpage Distribution (3D)",
) -> "go.Figure":
    """3D signed warpage (center-ref): red +Bow, blue −Crown."""
    import plotly.graph_objects as go

    X, Y, Z_mm, _ = wafer_warpage_grid_mm(kappa_1_per_m, radius_mm, n)
    zmin, zmax = _diverging_z_limits(Z_mm)
    scale, unit = warpage_display_unit(max(abs(zmin), abs(zmax)))
    Z = Z_mm * scale
    zmin, zmax = zmin * scale, zmax * scale

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="RdBu_r",
                cmin=zmin,
                cmax=zmax,
                colorbar=dict(
                    title=f"w ({unit})<br><span style='font-size:10px'>+Bow / −Crown</span>",
                ),
            )
        ]
    )
    z_span = zmax - zmin
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title=f"Warpage w ({unit})",
            zaxis=dict(autorange=False, range=[zmin, zmax]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=min(0.8, max(0.15, z_span / max(radius_mm, 1.0)))),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
        ),
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        uirevision=title,
    )
    return fig


def wafer_warpage_disk_mm(
    kappa_1_per_m: float,
    radius_mm: float,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed warpage on Cartesian disk (mm), center-referenced."""
    lin = np.linspace(-radius_mm, radius_mm, n)
    X, Y = np.meshgrid(lin, lin)
    R_mm = np.hypot(X, Y)
    Z = np.vectorize(lambda rr: displacement_mm(kappa_1_per_m, rr))(R_mm)
    Z[R_mm > radius_mm] = np.nan
    return X, Y, Z


def plot_wafer_warpage_contour(
    kappa_1_per_m: float,
    radius_mm: float,
    n: int = 80,
    title: str = "Wafer Warpage — Top View",
) -> "go.Figure":
    """Top-view diverging map (center-ref): red Bow (+), blue Crown (−)."""
    import plotly.graph_objects as go

    X, Y, Z_mm = wafer_warpage_disk_mm(kappa_1_per_m, radius_mm, n)
    zmin, zmax = _diverging_z_limits(Z_mm)
    scale, unit = warpage_display_unit(max(abs(zmin), abs(zmax)))
    Z = Z_mm * scale
    zmin, zmax = zmin * scale, zmax * scale
    lin = np.linspace(-radius_mm, radius_mm, n)
    fig = go.Figure(
        data=go.Contour(
            x=lin,
            y=lin,
            z=Z,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            colorbar=dict(title=f"w ({unit})"),
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            connectgaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", title="x (mm)"),
        yaxis=dict(title="y (mm)"),
        height=480,
        uirevision=title,
    )
    return fig


def plot_radial_warpage(
    r_mm: np.ndarray,
    w_mm: np.ndarray,
    *,
    kappa_1_per_m: float,
    title: str = "Radial profile w(r)",
    critical_mm: float = WARPAGE_CRITICAL_MM,
) -> "go.Figure":
    """Radial warpage line chart with data-scaled Y axis (critical line only if in view)."""
    import plotly.graph_objects as go

    peak_mm = float(np.max(w_mm)) if len(w_mm) else 0.0
    scale, unit = warpage_display_unit(peak_mm)
    w_plot = w_mm * scale
    crit_plot = critical_mm * scale
    crit_um = critical_mm * UM_PER_MM
    peak_um = peak_mm * UM_PER_MM

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r_mm,
            y=w_plot,
            mode="lines",
            name=f"w(r) [{unit}]",
            line=dict(color="#1565c0", width=3),
        )
    )

    # Do not pin Y axis to critical fail line when warpage is orders of magnitude smaller
    y_range = padded_axis_range(w_plot, floor=0.0, min_span=max(peak_mm * scale * 0.1, 1e-6))
    y_range[0] = 0.0
    if crit_plot <= y_range[1] * 1.25:
        fig.add_hline(
            y=crit_plot,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Critical {crit_um:.0f} {unit}",
        )
        y_range[1] = max(y_range[1], crit_plot * 1.05)
    else:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(color="red", size=11),
            text=f"Fail limit {crit_um:.0f} {unit} (off scale; peak ≈ {peak_um:.1f} {unit})",
        )

    fig.update_layout(
        title=f"{title} — κ = {kappa_1_per_m:.3e} 1/m",
        xaxis_title="Radius r (mm)",
        yaxis_title=f"Warpage w ({unit})",
        yaxis=dict(range=y_range, autorange=False),
        height=420,
        uirevision=title,
    )
    return fig


def plot_process_timeline(
    snapshots: list[ProcessStepSnapshot],
    selected_idx: int,
    *,
    architecture_label: str = "",
) -> "go.Figure":
    """Signed process warpage curve; click a point to inspect."""
    import plotly.graph_objects as go

    x = list(range(len(snapshots)))
    y = [float(s.warpage_edge_mm) * UM_PER_MM for s in snapshots]
    labels = [str(s.label) for s in snapshots]
    hover_text = [
        (
            f"{s.label} ({s.shape_label})<br>"
            f"Process T={float(s.temperature_c):.0f}°C<br>"
            f"T_sf={float(s.T_stress_free_c):.0f}°C<br>"
            f"ΔT={float(s.delta_T_c):.0f}°C<br>"
            f"w={float(s.warpage_edge_mm) * UM_PER_MM:.1f} {WARPAGE_UNIT}"
        )
        for s in snapshots
    ]
    marker_colors = [
        "#c62828" if i == selected_idx else ("#ef5350" if v > 0 else "#1e88e5" if v < 0 else "#9e9e9e")
        for i, v in enumerate(y)
    ]
    sizes = [18 if i == selected_idx else 11 for i in range(len(snapshots))]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Warpage",
            line=dict(color="#546e7a", width=2),
            marker=dict(size=sizes, color=marker_colors, line=dict(width=1, color="#37474f")),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_color="#424242", opacity=0.6)

    y_arr = np.asarray(y, dtype=float)
    ymax_data = float(np.max(np.abs(y_arr))) if y_arr.size else 0.0
    ymax = max(ymax_data * 1.25, WARPAGE_CRITICAL_UM * 0.01, 1e-6)
    y_range = [-ymax, ymax]

    if WARPAGE_CRITICAL_UM <= ymax * 2.5:
        fig.add_hline(
            y=WARPAGE_CRITICAL_UM,
            line_dash="dash",
            line_color="#e53935",
            line_width=2,
            annotation_text=f"Fail +{WARPAGE_CRITICAL_UM:.0f} {WARPAGE_UNIT}",
            annotation_position="right",
        )
        fig.add_hline(
            y=-WARPAGE_CRITICAL_UM,
            line_dash="dash",
            line_color="#e53935",
            line_width=2,
            annotation_text=f"Fail −{WARPAGE_CRITICAL_UM:.0f} {WARPAGE_UNIT}",
            annotation_position="right",
        )
        y_range[1] = max(y_range[1], WARPAGE_CRITICAL_UM * 1.05)
        y_range[0] = -y_range[1]
    else:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.99,
            y=0.98,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(color="#e53935", size=10),
            text=(
                f"Spec ±{WARPAGE_CRITICAL_UM:.0f} {WARPAGE_UNIT} "
                f"(off scale; peak |w| ≈ {ymax_data:.1f} {WARPAGE_UNIT})"
            ),
        )
    title = "Process warpage timeline (center-referenced, signed)"
    if architecture_label:
        title = f"{title}<br><sup>{architecture_label}</sup>"
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        xaxis=dict(tickmode="array", tickvals=x, ticktext=labels),
        yaxis_title=f"Warpage @ edge ({WARPAGE_UNIT})",
        yaxis=dict(range=y_range, autorange=False, zeroline=True),
        height=380,
        margin=dict(b=110),
        clickmode="event+select",
    )
    return fig


def stress_disk_map_mpa(
    result: WarpageResult,
    radius_mm: float,
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stress on wafer top surface mapped to Cartesian disk (MPa)."""
    lin = np.linspace(-radius_mm, radius_mm, n)
    X, Y = np.meshgrid(lin, lin)
    R = np.hypot(X, Y)
    Theta = np.arctan2(Y, X)
    mask = R <= radius_mm

    layers = result.layers
    kappa = result.kappa_1_per_m
    base = sum(ls.sigma_pa for ls in layers) / max(len(layers), 1) / 1e6
    z_top = max((ls.z_from_na_m for ls in layers), default=0.0)
    E_avg = np.mean([ls.E_eff_pa for ls in layers]) if layers else 1e9
    bend = E_avg * z_top * kappa / 1e6

    stress = base + bend * (R / max(radius_mm, 1e-6)) ** 2
    stress += 0.15 * base * np.cos(4 * Theta) * (R / radius_mm)
    stress = np.where(mask, stress, np.nan)
    return X, Y, stress


def plot_stress_map(
    result: WarpageResult,
    radius_mm: float,
    n: int = 80,
    title: str = "Top-surface stress map",
) -> "go.Figure":
    """
    In-plane stress on wafer top (MPa): layer residual σ plus bending from κ.
    Approximate — useful for comparing process steps, not a full FEM field.
    """
    import plotly.graph_objects as go

    X, Y, stress = stress_disk_map_mpa(result, radius_mm, n)
    zmin, zmax = _diverging_z_limits(stress)
    lin = np.linspace(-radius_mm, radius_mm, n)
    fig = go.Figure(
        data=go.Contour(
            x=lin,
            y=lin,
            z=stress,
            colorscale="RdYlBu_r",
            zmin=zmin,
            zmax=zmax,
            zmid=0.0,
            colorbar=dict(title="σ (MPa)"),
            contours=dict(coloring="heatmap", showlabels=True),
            connectgaps=False,
        )
    )
    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", title="x (mm)"),
        yaxis=dict(title="y (mm)"),
        height=520,
        margin=dict(t=50),
    )
    return fig
