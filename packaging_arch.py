"""Packaging architecture presets, process stations, layer stacks, and mechanics routing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from composite_layer import collapse_emc_die_coplanar
from constants import DEFAULT_THICKNESS, LayerStack, ProcessStep


class PackagingArchitecture(str, Enum):
    FOWLP_INFO = "FOWLP (Fan-Out Wafer Level Packaging)"
    COWOS_S = "2.5D Silicon Interposer"


# Legacy UI strings → enum (session state / saved configs)
_ARCHITECTURE_UI_ALIASES: dict[str, PackagingArchitecture] = {
    "FOWLP (InFO)": PackagingArchitecture.FOWLP_INFO,
    "2.5D CoWoS-S": PackagingArchitecture.COWOS_S,
}


# Blue info-box copy (main panel)
ARCHITECTURE_PROCESS_DESCRIPTION: dict[PackagingArchitecture, str] = {
    PackagingArchitecture.FOWLP_INFO: (
        "FOWLP (Fan-Out Wafer Level Packaging) · Chip-First：Carrier 放片 → Molding (灌注 EMC) → "
        "移除 Carrier 翻面 → 長 Front-Side RDL。"
    ),
    PackagingArchitecture.COWOS_S: (
        "2.5D Silicon Interposer：Interposer 穿孔與前段 RDL → Die 貼合 → "
        "Molding (灌注 EMC) → 背磨薄化 → 結合 Substrate。"
    ),
}

# Architecture-specific process timelines (sidebar + warpage chart X-axis)
# Chip-First FOWLP physical time axis (bottom of flow → end of flow)
STEPS_FOWLP: list[str] = [
    "Die on Carrier",
    "Molding",
    "Carrier Release",
    "Front-Side RDL",
    "Wafer Bumping",
]

STEPS_COWOS: list[str] = [
    "Interposer RDL",
    "Die on Interposer",
    "EMC Molding",
    "Back Grinding",
    "C4 Bumping",
    "Substrate Integration",
]

SOLDER_BUMPS_KEY = "Solder Bumps"

# Timeline: EMC mold cure activates thermal stress memory (ΔT vs stress-free T)
CURING_STATION_LABELS: frozenset[str] = frozenset({"Molding", "EMC Molding"})


def process_temperature_for_step(
    tstep: TimelineStepConfig,
    stress_free_T_c: float,
    T_profile: dict | None = None,
) -> float:
    """Resolve in-situ Process T; cure stations use stress-free (oven) temperature."""
    if T_profile and tstep.internal_step in T_profile:
        return float(T_profile[tstep.internal_step])
    if tstep.label in CURING_STATION_LABELS:
        return float(stress_free_T_c)
    return float(tstep.temperature_c)


def thermal_delta_for_timeline_step(
    is_cured: bool,
    process_T_c: float,
    stress_free_T_c: float,
) -> float:
    """Dynamic thermal history: no ΔT before cure; afterward ΔT = Process T − T_stress-free."""
    if not is_cured:
        return 0.0
    return float(process_T_c) - float(stress_free_T_c)


# Strict per-station layer on/off for 2.5D CoWoS-S (single source of truth)
COWOS_STATION_LAYER_ACTIVE: dict[str, dict[str, bool]] = {
    "Interposer RDL": {
        "Silicon Substrate": False,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": False,
        "EMC (Molding)": False,
        SOLDER_BUMPS_KEY: False,
    },
    "Die on Interposer": {
        "Silicon Substrate": False,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": True,
        "EMC (Molding)": False,
        SOLDER_BUMPS_KEY: False,
    },
    "EMC Molding": {
        "Silicon Substrate": False,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": True,
        "EMC (Molding)": True,
        SOLDER_BUMPS_KEY: False,
    },
    "Back Grinding": {
        "Silicon Substrate": False,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": True,
        "EMC (Molding)": True,
        SOLDER_BUMPS_KEY: False,
    },
    "C4 Bumping": {
        "Silicon Substrate": False,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": True,
        "EMC (Molding)": True,
        SOLDER_BUMPS_KEY: True,
    },
    "Substrate Integration": {
        "Silicon Substrate": True,
        "Silicon Interposer": True,
        "RDL (Polyimide)": True,
        "Die/Chip": True,
        "EMC (Molding)": True,
        "Solder Bumps": False,
    },
}

PROCESS_STATIONS: dict[PackagingArchitecture, list[str]] = {
    PackagingArchitecture.FOWLP_INFO: STEPS_FOWLP,
    PackagingArchitecture.COWOS_S: STEPS_COWOS,
}


@dataclass(frozen=True)
class TimelineStepConfig:
    """One station on the architecture-specific warpage timeline."""

    label: str
    internal_step: ProcessStep
    temperature_c: float
    layer_active: dict[str, bool]
    carrier_constrained: bool | None = None
    release_carrier: bool = False


def _timeline_fowlp() -> list[TimelineStepConfig]:
    """Chip-First: steps 1–2 with carrier constraint; carrier inactive from step 3 onward."""
    return [
        TimelineStepConfig(
            "Die on Carrier",
            ProcessStep.MOLDING,
            25.0,
            {
                "Carrier Glass": True,
                "EMC (Molding)": False,
                "Die/Chip": True,
                "RDL (Polyimide)": False,
                SOLDER_BUMPS_KEY: False,
            },
            carrier_constrained=True,
        ),
        TimelineStepConfig(
            "Molding",
            ProcessStep.MOLDING,
            175.0,
            {
                "Carrier Glass": True,
                "EMC (Molding)": True,
                "Die/Chip": True,
                "RDL (Polyimide)": False,
                SOLDER_BUMPS_KEY: False,
            },
            carrier_constrained=True,
        ),
        TimelineStepConfig(
            "Carrier Release",
            ProcessStep.FS_RDL,
            25.0,
            {
                "Carrier Glass": False,
                "EMC (Molding)": True,
                "Die/Chip": True,
                "RDL (Polyimide)": False,
                SOLDER_BUMPS_KEY: False,
            },
            carrier_constrained=False,
            release_carrier=True,
        ),
        TimelineStepConfig(
            "Front-Side RDL",
            ProcessStep.FS_RDL,
            200.0,
            {
                "Carrier Glass": False,
                "EMC (Molding)": True,
                "Die/Chip": True,
                "RDL (Polyimide)": True,
                SOLDER_BUMPS_KEY: False,
            },
            carrier_constrained=False,
        ),
        TimelineStepConfig(
            "Wafer Bumping",
            ProcessStep.BS_RDL,
            25.0,
            {
                "Carrier Glass": False,
                "EMC (Molding)": True,
                "Die/Chip": True,
                "RDL (Polyimide)": True,
                SOLDER_BUMPS_KEY: True,
            },
            carrier_constrained=False,
        ),
    ]


def _timeline_cowos() -> list[TimelineStepConfig]:
    specs: list[tuple[str, ProcessStep, float, bool | None]] = [
        ("Interposer RDL", ProcessStep.FS_RDL, 25.0, False),
        ("Die on Interposer", ProcessStep.TOP_RDL, 25.0, False),
        ("EMC Molding", ProcessStep.MOLDING, 175.0, False),
        ("Back Grinding", ProcessStep.BACK_GRINDING, 25.0, False),
        ("C4 Bumping", ProcessStep.BS_RDL, 260.0, False),
        ("Substrate Integration", ProcessStep.CARRIER_ATTACH, 25.0, True),
    ]
    return [
        TimelineStepConfig(
            label,
            step,
            temp_c,
            dict(COWOS_STATION_LAYER_ACTIVE[label]),
            carrier_constrained=constrained,
        )
        for label, step, temp_c, constrained in specs
    ]


ARCHITECTURE_TIMELINE: dict[PackagingArchitecture, list[TimelineStepConfig]] = {
    PackagingArchitecture.FOWLP_INFO: _timeline_fowlp(),
    PackagingArchitecture.COWOS_S: _timeline_cowos(),
}

# Sidebar Active presets = timeline station layer sets
STATION_LAYER_ACTIVE: dict[tuple[PackagingArchitecture, str], dict[str, bool]] = {
    (arch, step.label): dict(step.layer_active)
    for arch, steps in ARCHITECTURE_TIMELINE.items()
    for step in steps
}


@dataclass(frozen=True)
class ArchitectureLayerUI:
    material_key: str
    display_name: str
    default_active: bool
    show_split_cte: bool = False


# Bottom → top order in sidebar & 1D plate stack
ARCHITECTURE_LAYERS: dict[PackagingArchitecture, list[ArchitectureLayerUI]] = {
    PackagingArchitecture.FOWLP_INFO: [
        ArchitectureLayerUI("Carrier Glass", "Carrier Glass (底)", True),
        ArchitectureLayerUI("EMC (Molding)", "EMC Molding", True, show_split_cte=True),
        ArchitectureLayerUI("Die/Chip", "Die", True),
        ArchitectureLayerUI("RDL (Polyimide)", "Front RDL", True, show_split_cte=True),
        ArchitectureLayerUI(SOLDER_BUMPS_KEY, "Solder Bumps (植球層)", False),
    ],
    PackagingArchitecture.COWOS_S: [
        ArchitectureLayerUI("Silicon Substrate", "Substrate (底)", True),
        ArchitectureLayerUI("Silicon Interposer", "Silicon Interposer (INT)", True),
        ArchitectureLayerUI("RDL (Polyimide)", "RDL (INT 上)", True, show_split_cte=True),
        ArchitectureLayerUI("Die/Chip", "Die / HBM", True),
        ArchitectureLayerUI("EMC (Molding)", "EMC Molding", True, show_split_cte=True),
        ArchitectureLayerUI(SOLDER_BUMPS_KEY, "Solder Bumps (植球層)", False),
    ],
}

DW_DEFAULT_PARAMS: dict[PackagingArchitecture, tuple[str, str]] = {
    PackagingArchitecture.FOWLP_INFO: ("t:EMC (Molding)", "emc_alpha1"),
    PackagingArchitecture.COWOS_S: ("t:Silicon Interposer", "emc_alpha1"),
}

INTERPOSER_NA_STIFFNESS_WEIGHT = 4.0
FOWLP_COMPLIANT_LAYER_TYPES = frozenset({"emc", "rdl", "carrier"})


def parse_architecture(value: str) -> PackagingArchitecture:
    if value in _ARCHITECTURE_UI_ALIASES:
        return _ARCHITECTURE_UI_ALIASES[value]
    for arch in PackagingArchitecture:
        if arch.value == value:
            return arch
    return PackagingArchitecture.FOWLP_INFO


def architecture_process_description(arch: PackagingArchitecture | str) -> str:
    a = parse_architecture(arch) if isinstance(arch, str) else arch
    return ARCHITECTURE_PROCESS_DESCRIPTION[a]


def timeline_steps_for_architecture(arch: PackagingArchitecture | str) -> list[TimelineStepConfig]:
    a = parse_architecture(arch) if isinstance(arch, str) else arch
    return list(ARCHITECTURE_TIMELINE[a])


def timeline_labels_for_architecture(arch: PackagingArchitecture | str) -> list[str]:
    return [s.label for s in timeline_steps_for_architecture(arch)]


def final_timeline_label(arch: PackagingArchitecture | str) -> str:
    labels = timeline_labels_for_architecture(arch)
    return labels[-1] if labels else "Final"


def timeline_step_by_label(
    arch: PackagingArchitecture | str,
    label: str,
) -> TimelineStepConfig | None:
    for step in timeline_steps_for_architecture(arch):
        if step.label == label:
            return step
    return None


def stations_for_architecture(arch: PackagingArchitecture | str) -> list[str]:
    a = parse_architecture(arch) if isinstance(arch, str) else arch
    return list(PROCESS_STATIONS[a])


def default_process_station(arch: PackagingArchitecture | str) -> str:
    return stations_for_architecture(arch)[0]


def layer_active_for_station(
    arch: PackagingArchitecture | str,
    station: str,
) -> dict[str, bool]:
    """Strict station → layer Active map (CoWoS-S uses COWOS_STATION_LAYER_ACTIVE)."""
    a = parse_architecture(arch) if isinstance(arch, str) else arch
    if a == PackagingArchitecture.COWOS_S and station in COWOS_STATION_LAYER_ACTIVE:
        return dict(COWOS_STATION_LAYER_ACTIVE[station])
    preset = STATION_LAYER_ACTIVE.get((a, station))
    if preset:
        return dict(preset)
    return {row.material_key: row.default_active for row in ARCHITECTURE_LAYERS[a]}


def timeline_layer_active(tstep: TimelineStepConfig, arch: PackagingArchitecture | str) -> dict[str, bool]:
    """Per-timeline-station layer on/off (physics + UI); FOWLP carrier overrides."""
    layer_active = layer_active_for_station(arch, tstep.label)
    a = parse_architecture(arch) if isinstance(arch, str) else arch
    if a == PackagingArchitecture.FOWLP_INFO:
        if tstep.label in ("Die on Carrier", "Molding"):
            layer_active["Carrier Glass"] = True
        elif tstep.label in ("Carrier Release", "Front-Side RDL", "Wafer Bumping"):
            layer_active["Carrier Glass"] = False
    return layer_active


def process_step_for_station(
    arch: PackagingArchitecture | str,
    station: str,
) -> ProcessStep:
    cfg = timeline_step_by_label(arch, station)
    return cfg.internal_step if cfg else ProcessStep.MOLDING


def default_layer_active(arch: PackagingArchitecture) -> dict[str, bool]:
    """All visible layers Active by default (sidebar material input)."""
    return {row.material_key: True for row in ARCHITECTURE_LAYERS[arch]}


def visible_material_keys(arch: PackagingArchitecture) -> list[str]:
    return [row.material_key for row in ARCHITECTURE_LAYERS[arch]]


def layer_display_name(arch: PackagingArchitecture, material_key: str) -> str:
    for row in ARCHITECTURE_LAYERS[arch]:
        if row.material_key == material_key:
            return row.display_name
    return material_key


def _t(
    layer_thickness_um: dict[str, float] | None,
    key: str,
    fallback: float | None = None,
) -> float:
    if layer_thickness_um and key in layer_thickness_um:
        return layer_thickness_um[key]
    if fallback is not None:
        return fallback
    return DEFAULT_THICKNESS[key]


def _layer(
    key: str,
    thickness_um: float,
    layer_active: dict[str, bool] | None,
    *,
    default_active: bool = True,
) -> LayerStack:
    active = layer_active or {}
    include = active.get(key, default_active)
    return LayerStack(key, thickness_um, include=include)


def _stack_from_ordered(ordered: list[LayerStack]) -> list[LayerStack]:
    """Return bottom→top plate layers; EMC+Die collapse to one coplanar composite when both active."""
    included = [ls for ls in ordered if ls.include]
    return collapse_emc_die_coplanar(included)


def build_stack_for_step(
    step: ProcessStep,
    *,
    packaging_architecture: str = PackagingArchitecture.FOWLP_INFO.value,
    substrate_thickness_um: float | None = None,
    layer_thickness_um: dict[str, float] | None = None,
    layer_active: dict[str, bool] | None = None,
) -> tuple[list[LayerStack], bool]:
    arch = parse_architecture(packaging_architecture)
    if arch == PackagingArchitecture.FOWLP_INFO:
        return _build_fowlp_stack(step, layer_thickness_um, layer_active)
    return _build_cowos_stack(
        step,
        substrate_thickness_um=substrate_thickness_um,
        layer_thickness_um=layer_thickness_um,
        layer_active=layer_active,
    )


def _fowlp_ordered_layers(
    lt: dict[str, float] | None,
    layer_active: dict[str, bool] | None,
) -> list[LayerStack]:
    """Bottom → top: Carrier → EMC → Die → RDL."""
    return [
        _layer("Carrier Glass", _t(lt, "Carrier Glass"), layer_active, default_active=True),
        _layer("EMC (Molding)", _t(lt, "EMC (Molding)"), layer_active, default_active=True),
        _layer("Die/Chip", _t(lt, "Die/Chip"), layer_active, default_active=True),
        _layer("RDL (Polyimide)", _t(lt, "RDL (Polyimide)"), layer_active, default_active=False),
        _layer(SOLDER_BUMPS_KEY, _t(lt, SOLDER_BUMPS_KEY), layer_active, default_active=False),
    ]


def _build_fowlp_stack(
    step: ProcessStep,
    lt: dict[str, float] | None,
    layer_active: dict[str, bool] | None,
) -> tuple[list[LayerStack], bool]:
    ordered = _fowlp_ordered_layers(lt, layer_active)
    stack = _stack_from_ordered(ordered)
    carrier_on = (layer_active or {}).get("Carrier Glass", True)

    constrained = step in (ProcessStep.CARRIER_ATTACH, ProcessStep.BACK_GRINDING, ProcessStep.BS_RDL) and carrier_on
    if step == ProcessStep.MOLDING:
        constrained = carrier_on
    return stack, constrained


def _cowos_ordered_layers(
    lt: dict[str, float] | None,
    layer_active: dict[str, bool] | None,
    *,
    substrate_thickness_um: float | None = None,
    thin_substrate: bool = False,
) -> list[LayerStack]:
    """Bottom → top: Substrate → INT → RDL → Die → EMC."""
    t_sub = 100.0 if thin_substrate else (
        substrate_thickness_um if substrate_thickness_um is not None else _t(lt, "Silicon Substrate")
    )
    return [
        _layer("Silicon Substrate", t_sub, layer_active, default_active=False),
        _layer("Silicon Interposer", _t(lt, "Silicon Interposer"), layer_active, default_active=True),
        _layer("RDL (Polyimide)", _t(lt, "RDL (Polyimide)"), layer_active, default_active=True),
        _layer("Die/Chip", _t(lt, "Die/Chip"), layer_active, default_active=True),
        _layer("EMC (Molding)", _t(lt, "EMC (Molding)"), layer_active, default_active=True),
        _layer(SOLDER_BUMPS_KEY, _t(lt, SOLDER_BUMPS_KEY), layer_active, default_active=False),
    ]


def _build_cowos_stack(
    step: ProcessStep,
    *,
    substrate_thickness_um: float | None,
    layer_thickness_um: dict[str, float] | None,
    layer_active: dict[str, bool] | None,
) -> tuple[list[LayerStack], bool]:
    lt = layer_thickness_um or {}
    thin = step in (ProcessStep.BACK_GRINDING, ProcessStep.BS_RDL, ProcessStep.DEBONDING)
    ordered = _cowos_ordered_layers(
        lt,
        layer_active,
        substrate_thickness_um=substrate_thickness_um,
        thin_substrate=thin,
    )
    stack = _stack_from_ordered(ordered)

    constrained = step == ProcessStep.CARRIER_ATTACH and (layer_active or {}).get("Silicon Substrate", False)
    return stack, constrained
