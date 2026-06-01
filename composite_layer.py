"""Coplanar EMC + Die molding composite for 1D plate warpage (same Z, EMC thickness)."""

from __future__ import annotations

from constants import MATERIALS, LayerStack, MaterialProps

MOLDING_COMPOSITE_KEY = "Molding (EMC+Die)"
EMC_KEY = "EMC (Molding)"
DIE_KEY = "Die/Chip"
DEFAULT_DIE_AREA_FRACTION = 0.30
COPLANAR_NOTE = "*Coplanar layer (同層複合結構)"


def arch_has_molding_pair(arch) -> bool:
    from packaging_arch import visible_material_keys

    keys = set(visible_material_keys(arch))
    return EMC_KEY in keys and DIE_KEY in keys


def rule_of_mixtures(value_die: float, value_emc: float, die_area_fraction: float) -> float:
    """Area-weighted Voigt mixture: f·die + (1−f)·emc."""
    f = max(0.0, min(1.0, die_area_fraction))
    return f * value_die + (1.0 - f) * value_emc


def emc_die_both_active(layer_active: dict[str, bool] | None) -> bool:
    active = layer_active or {}
    return bool(active.get(EMC_KEY, False) and active.get(DIE_KEY, False))


def collapse_emc_die_coplanar(stacks: list[LayerStack]) -> list[LayerStack]:
    """
    When EMC and Die are both included, replace them with one plate layer:
    thickness = EMC thickness, bottom Z aligned at the lower of the two indices.
    """
    emc_idx = die_idx = None
    for i, ls in enumerate(stacks):
        if not ls.include:
            continue
        if ls.material_key == EMC_KEY:
            emc_idx = i
        elif ls.material_key == DIE_KEY:
            die_idx = i

    if emc_idx is None or die_idx is None:
        return stacks

    emc = stacks[emc_idx]
    insert_at = min(emc_idx, die_idx)
    composite = LayerStack(MOLDING_COMPOSITE_KEY, emc.thickness_um, include=True)

    out: list[LayerStack] = []
    composite_placed = False
    for i, ls in enumerate(stacks):
        if ls.material_key in (EMC_KEY, DIE_KEY) and ls.include:
            if i == insert_at and not composite_placed:
                out.append(composite)
                composite_placed = True
            continue
        out.append(ls)
    return out


def composite_material_props() -> MaterialProps:
    return MATERIALS[MOLDING_COMPOSITE_KEY]


def composite_e_gpa(
    T_c: float,
    die_area_fraction: float,
    *,
    layer_E_gpa: dict[str, float] | None,
    emc_E_gpa: float | None,
    effective_young_modulus,
) -> float:
    f = max(0.0, min(1.0, die_area_fraction))
    mat_die = MATERIALS[DIE_KEY]
    mat_emc = MATERIALS[EMC_KEY]
    e_die = effective_young_modulus(
        mat_die, T_c, E_override_gpa=(layer_E_gpa or {}).get(DIE_KEY)
    )
    e_emc = effective_young_modulus(
        mat_emc,
        T_c,
        E_override_gpa=emc_E_gpa if emc_E_gpa is not None else (layer_E_gpa or {}).get(EMC_KEY),
    )
    return rule_of_mixtures(e_die, e_emc, f)


def composite_alpha_ppm(
    T_c: float,
    die_area_fraction: float,
    *,
    emc_alpha1: float | None,
    emc_alpha2: float | None,
    effective_alpha,
) -> float:
    f = max(0.0, min(1.0, die_area_fraction))
    mat_die = MATERIALS[DIE_KEY]
    mat_emc = MATERIALS[EMC_KEY]
    a_die = effective_alpha(mat_die, T_c)
    a_emc = effective_alpha(mat_emc, T_c, emc_alpha1, emc_alpha2, None)
    return rule_of_mixtures(a_die, a_emc, f)
