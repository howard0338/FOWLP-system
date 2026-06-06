"""Material properties and process definitions for 2.5D packaging warpage simulation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LayerType(str, Enum):
    SILICON = "silicon"
    INTERPOSER = "interposer"
    EMC = "emc"
    DIE = "die"
    RDL = "rdl"
    CARRIER = "carrier"
    SOLDER = "solder"


@dataclass(frozen=True)
class MaterialProps:
    name: str
    E_gpa: float
    nu: float
    alpha_ppm: float
    layer_type: LayerType
    alpha2_ppm: float | None = None
    Tg_c: float | None = None
    alpha_sub_ref: bool = False


# Baseline material database (GPa, ppm/K)
MATERIALS: dict[str, MaterialProps] = {
    "Silicon Substrate": MaterialProps(
        name="Silicon Substrate",
        E_gpa=130.0,
        nu=0.22,
        alpha_ppm=2.6,
        layer_type=LayerType.SILICON,
        alpha_sub_ref=True,
    ),
    "Silicon Interposer": MaterialProps(
        name="Silicon Interposer",
        E_gpa=130.0,
        nu=0.22,
        alpha_ppm=2.6,
        layer_type=LayerType.INTERPOSER,
    ),
    "EMC (Molding)": MaterialProps(
        name="EMC (Molding)",
        E_gpa=12.0,
        nu=0.25,
        alpha_ppm=12.0,
        alpha2_ppm=40.0,
        layer_type=LayerType.EMC,
    ),
    "Die/Chip": MaterialProps(
        name="Die/Chip",
        E_gpa=130.0,
        nu=0.22,
        alpha_ppm=2.6,
        layer_type=LayerType.DIE,
    ),
    "Molding (EMC+Die)": MaterialProps(
        name="Molding (EMC+Die)",
        E_gpa=12.0,
        nu=0.25,
        alpha_ppm=12.0,
        alpha2_ppm=40.0,
        layer_type=LayerType.EMC,
    ),
    "RDL (Polyimide)": MaterialProps(
        name="RDL (Polyimide)",
        E_gpa=3.5,
        nu=0.34,
        alpha_ppm=25.0,
        alpha2_ppm=45.0,
        Tg_c=300.0,
        layer_type=LayerType.RDL,
    ),
    "Carrier Glass": MaterialProps(
        name="Carrier Glass",
        E_gpa=70.0,
        nu=0.22,
        alpha_ppm=5.0,
        layer_type=LayerType.CARRIER,
    ),
    "Solder Bumps": MaterialProps(
        name="Solder Bumps",
        E_gpa=0.01,
        nu=0.35,
        alpha_ppm=22.0,
        layer_type=LayerType.SOLDER,
    ),
}


# Temperature-dependent EMC parameters
EMC_T_SOFTEN_START_C = 150.0
EMC_T_SOFTEN_END_C = 220.0
EMC_E_MIN_FRACTION = 0.05


@dataclass
class LayerStack:
    """Single layer in the multi-layer plate stack."""

    material_key: str
    thickness_um: float
    include: bool = True

    @property
    def material(self) -> MaterialProps:
        return MATERIALS[self.material_key]


class ProcessStep(str, Enum):
    """Chip-last FO-WLP sequence: front RDL → molding → top RDL → back-end."""

    FS_RDL = "Front-Side RDL"
    MOLDING = "Molding"
    TOP_RDL = "Top-RDL"
    CARRIER_ATTACH = "Carrier Attach"
    BACK_GRINDING = "Back Grinding"
    BS_RDL = "Back-Side RDL"
    DEBONDING = "Debonding"


# Explicit thermal / mechanical sequence (do not rely on Enum definition order alone).
PROCESS_ORDER: list[ProcessStep] = [
    ProcessStep.FS_RDL,
    ProcessStep.MOLDING,
    ProcessStep.TOP_RDL,
    ProcessStep.CARRIER_ATTACH,
    ProcessStep.BACK_GRINDING,
    ProcessStep.BS_RDL,
    ProcessStep.DEBONDING,
]


@dataclass
class ProcessState:
    step: ProcessStep
    temperature_c: float
    T_ref_c: float
    layers: list[LayerStack]
    carrier_constrained: bool
    description: str = ""


# Default layer thicknesses (µm)
DEFAULT_THICKNESS = {
    "Silicon Substrate": 775.0,
    "Silicon Interposer": 150.0,
    "EMC (Molding)": 200.0,
    "Die/Chip": 100.0,
    "RDL (Polyimide)": 15.0,
    "Carrier Glass": 500.0,
    "Solder Bumps": 80.0,
}

# Process temperature profile per step (°C)
PROCESS_TEMPERATURES: dict[ProcessStep, float] = {
    ProcessStep.MOLDING: 175.0,
    ProcessStep.FS_RDL: 200.0,
    ProcessStep.TOP_RDL: 220.0,
    ProcessStep.CARRIER_ATTACH: 180.0,
    ProcessStep.BACK_GRINDING: 25.0,
    ProcessStep.BS_RDL: 200.0,
    ProcessStep.DEBONDING: 25.0,
}

PROCESS_T_REF_C = 25.0
DEFAULT_MEASUREMENT_T_C = 25.0  # measurement / room-temperature warpage assessment (°C)
DEFAULT_EVALUATION_T_C = DEFAULT_MEASUREMENT_T_C  # legacy alias
DEFAULT_STRESS_FREE_T_C = 175.0  # molding cure / stress-free temperature (°C)
WAFER_RADIUS_MM = 50.0  # wafer radius default (mm)
WARPAGE_CRITICAL_MM = 1.5  # solver / compare threshold (mm)

# UI length & warpage display (micrometers)
UM_PER_MM = 1000.0
WARPAGE_UNIT = "µm"


def mm_to_um(x_mm: float) -> float:
    return float(x_mm) * UM_PER_MM


def um_to_mm(x_um: float) -> float:
    return float(x_um) / UM_PER_MM


WARPAGE_CRITICAL_UM = mm_to_um(WARPAGE_CRITICAL_MM)

@dataclass(frozen=True)
class LayerInputSpec:
    """Intrinsic layer properties — user Input (not solver Output)."""

    material_key: str
    display_name: str
    thickness_um_default: float
    thickness_um_range: tuple[float, float]
    E_gpa_default: float
    E_gpa_range: tuple[float, float]
    cte_display: str
    alpha1_ppm_default: float
    alpha2_ppm_default: float | None = None
    Tg_c_default: float | None = None
    has_split_cte: bool = False
    note: str = ""


# Unified material input catalog (multi-layer plate theory — intrinsic properties)
INPUT_LAYER_SPECS: list[LayerInputSpec] = [
    LayerInputSpec(
        material_key="Silicon Substrate",
        display_name="Silicon Substrate",
        thickness_um_default=775.0,
        thickness_um_range=(100.0, 800.0),
        E_gpa_default=130.0,
        E_gpa_range=(100.0, 180.0),
        cte_display="2.6",
        alpha1_ppm_default=2.6,
        note="剛性基準",
    ),
    LayerInputSpec(
        material_key="Silicon Interposer",
        display_name="Silicon Interposer",
        thickness_um_default=150.0,
        thickness_um_range=(100.0, 200.0),
        E_gpa_default=130.0,
        E_gpa_range=(100.0, 180.0),
        cte_display="2.6",
        alpha1_ppm_default=2.6,
        note="矽 Interposer · 強約束翹曲",
    ),
    LayerInputSpec(
        material_key="EMC (Molding)",
        display_name="EMC (Molding)",
        thickness_um_default=200.0,
        thickness_um_range=(50.0, 400.0),
        E_gpa_default=12.0,
        E_gpa_range=(5.0, 30.0),
        cte_display="12 / 40",
        alpha1_ppm_default=12.0,
        alpha2_ppm_default=40.0,
        has_split_cte=True,
        note="翹曲驅動源 (Tg 前/後)",
    ),
    LayerInputSpec(
        material_key="Die/Chip",
        display_name="Die/Chip",
        thickness_um_default=100.0,
        thickness_um_range=(50.0, 200.0),
        E_gpa_default=130.0,
        E_gpa_range=(100.0, 180.0),
        cte_display="2.6",
        alpha1_ppm_default=2.6,
        note="剛性填料",
    ),
    LayerInputSpec(
        material_key="RDL (Polyimide)",
        display_name="RDL (Polyimide)",
        thickness_um_default=15.0,
        thickness_um_range=(5.0, 40.0),
        E_gpa_default=3.5,
        E_gpa_range=(2.0, 8.0),
        cte_display="25 / 45",
        alpha1_ppm_default=25.0,
        alpha2_ppm_default=45.0,
        Tg_c_default=300.0,
        has_split_cte=True,
        note="不對稱應力源",
    ),
    LayerInputSpec(
        material_key="Carrier Glass",
        display_name="Carrier Glass",
        thickness_um_default=500.0,
        thickness_um_range=(200.0, 800.0),
        E_gpa_default=70.0,
        E_gpa_range=(50.0, 90.0),
        cte_display="5.0",
        alpha1_ppm_default=5.0,
        note="剛性載板",
    ),
    LayerInputSpec(
        material_key="Solder Bumps",
        display_name="Solder Bumps (植球層)",
        thickness_um_default=80.0,
        thickness_um_range=(20.0, 150.0),
        E_gpa_default=0.01,
        E_gpa_range=(0.001, 1.0),
        cte_display="22.0",
        alpha1_ppm_default=22.0,
        note="Wafer/C4 bumping · discrete solder",
    ),
]

PARAM_RANGES = {
    "measurement_T": (-40.0, 125.0, 25.0),
    "die_area_fraction": (0.05, 0.70, 0.30),
    "stress_free_T": (150.0, 200.0, 175.0),
    "emc_alpha1": (8.0, 15.0, 12.0),
    "emc_alpha2": (30.0, 60.0, 40.0),
    "rdl_alpha1": (15.0, 40.0, 25.0),
    "rdl_alpha2": (30.0, 60.0, 45.0),
    "rdl_Tg": (250.0, 350.0, 300.0),
}

# Sequential process model (paper-like: residual stress + debond release)
# T_stress_free tracks the last stress-free temperature (partial relaxation each step).
STRESS_FREE_RELAXATION = 0.28
RESIDUAL_MOMENT_GAIN = 0.22
STEP_BUILDUP_GAIN = 0.11  # cumulative packaging build per process index
# Mild warpage jump when carrier / substrate constraint is released (dimensionless ratio on |w|)
DEBOND_RELEASE_RATIO = 1.35
# Legacy hook kept at unity — warpage is computed in SI; do not apply empirical mm scaling here
WARPAGE_CALIBRATION = 1.0

# Solder bump: near-zero flexural rigidity (discrete spheres — excluded from bending moment)
SOLDER_MOMENT_FACTOR = 0.0
E_MIN_SOLDER_GPA = 1e-6
E_REF_PA_FLOOR = 1e6

# 2.5D stack: rigid Si interposer vs compliant EMC (moment weighting in plate model)
INTERPOSER_MOMENT_GAIN = 1.38
EMC_MOMENT_FACTOR_WITH_INTERPOSER = 0.40


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


def build_stack_for_step(
    step: ProcessStep,
    *,
    packaging_architecture: str | None = None,
    substrate_thickness_um: float | None = None,
    layer_thickness_um: dict[str, float] | None = None,
    layer_active: dict[str, bool] | None = None,
) -> tuple[list[LayerStack], bool]:
    """Delegate to packaging_arch (FOWLP InFO vs 2.5D CoWoS-S layer build-up)."""
    from packaging_arch import build_stack_for_step as _build

    return _build(
        step,
        packaging_architecture=packaging_architecture or "FOWLP (Fan-Out Wafer Level Packaging)",
        substrate_thickness_um=substrate_thickness_um,
        layer_thickness_um=layer_thickness_um,
        layer_active=layer_active,
    )
