"""User material input layer — intrinsic E, CTE, thickness (multi-layer plate inputs)."""

from __future__ import annotations

from dataclasses import dataclass, field

from composite_layer import DEFAULT_DIE_AREA_FRACTION
from constants import (
    DEFAULT_MEASUREMENT_T_C,
    DEFAULT_STRESS_FREE_T_C,
    INPUT_LAYER_SPECS,
    LayerInputSpec,
)
from packaging_arch import (
    PackagingArchitecture,
    default_layer_active,
    layer_display_name,
    parse_architecture,
    visible_material_keys,
)


@dataclass
class MaterialInputConfig:
    """Intrinsic properties fed into the plate-theory solver (Input layer)."""

    thickness_um: dict[str, float] = field(default_factory=dict)
    E_gpa: dict[str, float] = field(default_factory=dict)
    emc_alpha1: float = 12.0
    emc_alpha2: float = 40.0
    rdl_alpha1: float = 25.0
    rdl_alpha2: float = 45.0
    rdl_Tg: float = 300.0
    T_ref: float = DEFAULT_STRESS_FREE_T_C
    measurement_T_c: float = DEFAULT_MEASUREMENT_T_C
    die_area_fraction: float = DEFAULT_DIE_AREA_FRACTION
    wafer_radius_mm: float = 150.0
    packaging_architecture: str = PackagingArchitecture.FOWLP_INFO.value
    process_station: str = ""
    layer_active: dict[str, bool] = field(default_factory=dict)

    def to_solver_kwargs(self) -> dict:
        return {
            "substrate_thickness_um": self.thickness_um.get("Silicon Substrate"),
            "layer_thickness_um": self.thickness_um,
            "layer_E_gpa": self.E_gpa,
            "layer_active": dict(self.layer_active),
            "packaging_architecture": self.packaging_architecture,
            "emc_alpha1": self.emc_alpha1,
            "emc_alpha2": self.emc_alpha2,
            "rdl_alpha1": self.rdl_alpha1,
            "rdl_alpha2": self.rdl_alpha2,
            "rdl_Tg": self.rdl_Tg,
            "emc_E_gpa": self.E_gpa.get("EMC (Molding)"),
            "measurement_T_c": self.measurement_T_c,
            "evaluation_T_c": self.measurement_T_c,  # legacy
            "stress_free_T_c": self.T_ref,
            "die_area_fraction": self.die_area_fraction,
        }


def default_material_inputs(
    arch: PackagingArchitecture | str | None = None,
) -> MaterialInputConfig:
    parsed = parse_architecture(arch) if arch is not None else PackagingArchitecture.FOWLP_INFO
    t = {spec.material_key: spec.thickness_um_default for spec in INPUT_LAYER_SPECS}
    e = {spec.material_key: spec.E_gpa_default for spec in INPUT_LAYER_SPECS}
    emc = next(s for s in INPUT_LAYER_SPECS if s.material_key == "EMC (Molding)")
    rdl = next(s for s in INPUT_LAYER_SPECS if s.material_key == "RDL (Polyimide)")
    return MaterialInputConfig(
        thickness_um=t,
        E_gpa=e,
        emc_alpha1=emc.alpha1_ppm_default,
        emc_alpha2=emc.alpha2_ppm_default or 40.0,
        rdl_alpha1=rdl.alpha1_ppm_default,
        rdl_alpha2=rdl.alpha2_ppm_default or 45.0,
        rdl_Tg=rdl.Tg_c_default or 300.0,
        packaging_architecture=parsed.value,
        process_station="",
        layer_active=default_layer_active(parsed),
    )


def input_reference_table_rows() -> list[dict[str, str | float]]:
    """Static engineering reference (defaults) — not solver output."""
    rows: list[dict[str, str | float]] = []
    for spec in INPUT_LAYER_SPECS:
        rows.append(
            {
                "Layer (層別)": spec.display_name,
                "Thickness (t, μm)": spec.thickness_um_default,
                "Modulus (E, GPa)": spec.E_gpa_default,
                "CTE (ppm/K)": spec.cte_display,
                "Note (備註)": spec.note,
            }
        )
    return rows


def input_current_table_rows(cfg: MaterialInputConfig) -> list[dict[str, str | float]]:
    """What the user has set in the sidebar (current Input state)."""
    arch = parse_architecture(cfg.packaging_architecture)
    rows: list[dict[str, str | float]] = []
    for key in visible_material_keys(arch):
        spec = next(s for s in INPUT_LAYER_SPECS if s.material_key == key)
        active = cfg.layer_active.get(key, False)
        if spec.has_split_cte:
            if "EMC" in key:
                cte = f"{cfg.emc_alpha1:.0f} / {cfg.emc_alpha2:.0f}"
            else:
                cte = f"{cfg.rdl_alpha1:.0f} / {cfg.rdl_alpha2:.0f}"
        else:
            cte = spec.cte_display
        rows.append(
            {
                "Layer (層別)": layer_display_name(arch, key),
                "Active": "Yes" if active else "No",
                "Thickness (t, μm)": cfg.thickness_um.get(key, spec.thickness_um_default),
                "Modulus (E, GPa)": cfg.E_gpa.get(key, spec.E_gpa_default),
                "CTE (ppm/K)": cte,
                "Note (備註)": spec.note,
            }
        )
    return rows
