"""User material input layer — intrinsic E, CTE, thickness (multi-layer plate inputs)."""

from __future__ import annotations

from dataclasses import dataclass, field

from constants import DEFAULT_STRESS_FREE_T_C, INPUT_LAYER_SPECS, LayerInputSpec


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
    wafer_radius_mm: float = 150.0

    def to_solver_kwargs(self) -> dict:
        return {
            "substrate_thickness_um": self.thickness_um.get("Silicon Substrate"),
            "layer_thickness_um": self.thickness_um,
            "layer_E_gpa": self.E_gpa,
            "emc_alpha1": self.emc_alpha1,
            "emc_alpha2": self.emc_alpha2,
            "rdl_alpha1": self.rdl_alpha1,
            "rdl_alpha2": self.rdl_alpha2,
            "rdl_Tg": self.rdl_Tg,
            "emc_E_gpa": self.E_gpa.get("EMC (Molding)"),
        }


def default_material_inputs() -> MaterialInputConfig:
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
    rows: list[dict[str, str | float]] = []
    for spec in INPUT_LAYER_SPECS:
        key = spec.material_key
        if spec.has_split_cte:
            cte = f"{cfg.emc_alpha1:.0f} / {cfg.emc_alpha2:.0f}" if "EMC" in key else f"{cfg.rdl_alpha1:.0f} / {cfg.rdl_alpha2:.0f}"
        else:
            cte = spec.cte_display
        rows.append(
            {
                "Layer (層別)": spec.display_name,
                "Thickness (t, μm)": cfg.thickness_um.get(key, spec.thickness_um_default),
                "Modulus (E, GPa)": cfg.E_gpa.get(key, spec.E_gpa_default),
                "CTE (ppm/K)": cte,
                "Note (備註)": spec.note,
            }
        )
    return rows
