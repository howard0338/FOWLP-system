"""Multi-layer stack schematic — flexbox layout, coplanar EMC+Die molding band."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from composite_layer import (
    COPLANAR_NOTE,
    DEFAULT_DIE_AREA_FRACTION,
    DIE_KEY,
    EMC_KEY,
    MOLDING_COMPOSITE_KEY,
    arch_has_molding_pair,
    rule_of_mixtures,
)
from constants import ProcessStep, build_stack_for_step
from packaging_arch import parse_architecture, visible_material_keys

if TYPE_CHECKING:
    from layer_inputs import MaterialInputConfig

_LAYER_STYLES: dict[str, tuple[str, str, str]] = {
    "Silicon Substrate": ("Silicon Substrate", "SUB", "fowlp-mat--sub"),
    "Silicon Interposer": ("Silicon Interposer", "INT", "fowlp-mat--int"),
    "Die/Chip": ("Die / Chip", "DIE", "fowlp-mat--die"),
    "RDL (Polyimide)": ("RDL (PI)", "RDL", "fowlp-mat--rdl"),
    "EMC (Molding)": ("EMC Molding", "EMC", "fowlp-mat--emc"),
    "Carrier Glass": ("Carrier Glass", "CAR", "fowlp-mat--car"),
    "Solder Bumps": ("Solder Bumps (植球層)", "SLD", "fowlp-mat--sld"),
}

_MATERIAL_CSS_DEFAULT = "fowlp-mat--default"


def _mat_class(material_key: str) -> str:
    return _LAYER_STYLES.get(material_key, (material_key, "?", _MATERIAL_CSS_DEFAULT))[2]

_STACK_BAR_PX = 200


def _plate_thickness_um(cfg: MaterialInputConfig, step: ProcessStep, active_map: dict[str, bool]) -> float:
    stacks, _ = build_stack_for_step(
        step,
        packaging_architecture=cfg.packaging_architecture,
        substrate_thickness_um=cfg.thickness_um.get("Silicon Substrate"),
        layer_thickness_um=cfg.thickness_um,
        layer_active=active_map,
    )
    return sum(ls.thickness_um for ls in stacks if ls.include)


def _simple_band(
    key: str,
    cfg: MaterialInputConfig,
    active_map: dict[str, bool],
) -> dict[str, Any]:
    title, tag, mat_cls = _LAYER_STYLES.get(
        key, (key, key[:3].upper(), _MATERIAL_CSS_DEFAULT)
    )
    on = bool(active_map.get(key, False))
    return {
        "kind": "layer",
        "key": key,
        "title": title,
        "tag": tag,
        "mat_cls": mat_cls,
        "t_um": float(cfg.thickness_um.get(key, 0.0)),
        "e_gpa": float(cfg.E_gpa.get(key, 0.0)),
        "active": on,
    }


def _composite_band(cfg: MaterialInputConfig, active_map: dict[str, bool]) -> dict[str, Any]:
    f_die = max(0.05, min(0.95, getattr(cfg, "die_area_fraction", DEFAULT_DIE_AREA_FRACTION)))
    t_emc = float(cfg.thickness_um.get(EMC_KEY, 200.0))
    e_die = float(cfg.E_gpa.get(DIE_KEY, 130.0))
    e_emc = float(cfg.E_gpa.get(EMC_KEY, 12.0))
    die_on = bool(active_map.get(DIE_KEY, False))
    emc_on = bool(active_map.get(EMC_KEY, False))
    return {
        "kind": "composite",
        "key": "molding_composite",
        "title": "Molding Layer (EMC + Die)",
        "t_um": t_emc,
        "e_gpa": rule_of_mixtures(e_die, e_emc, f_die),
        "die_frac": f_die,
        "die_active": die_on,
        "emc_active": emc_on,
        "active": die_on or emc_on,
        "die_mat": "fowlp-mat--die",
        "emc_mat": "fowlp-mat--emc",
    }


def _molding_zone_visible(active_map: dict[str, bool]) -> bool:
    """Coplanar molding UI only when BOTH EMC and Die are active (matches solver collapse)."""
    return bool(active_map.get(DIE_KEY, False) and active_map.get(EMC_KEY, False))


def _bands_bottom_to_top(
    cfg: MaterialInputConfig,
    *,
    layer_active: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Build stack bands bottom → top (plate order)."""
    active_map = dict(layer_active if layer_active is not None else cfg.layer_active)
    arch = parse_architecture(cfg.packaging_architecture)
    keys = visible_material_keys(arch)
    show_composite = arch_has_molding_pair(arch) and _molding_zone_visible(active_map)

    bands: list[dict[str, Any]] = []
    skip: set[str] = {EMC_KEY, DIE_KEY} if show_composite else set()

    i = 0
    while i < len(keys):
        key = keys[i]
        if not show_composite and key in (EMC_KEY, DIE_KEY):
            i += 1
            continue
        if key in skip:
            if not any(b.get("kind") == "composite" for b in bands):
                bands.append(_composite_band(cfg, active_map))
            i += 1
            while i < len(keys) and keys[i] in skip:
                i += 1
            continue
        bands.append(_simple_band(key, cfg, active_map))
        i += 1

    return bands


def _active_legend_bands_top_first(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    active_map: dict[str, bool],
) -> list[dict[str, Any]]:
    """Legend rows = solver plate only (no grey future/inactive layers)."""
    return list(reversed(_solver_plate_segments_bottom_to_top(cfg, step, active_map)))


def _band_is_active(band: dict[str, Any]) -> bool:
    if band["kind"] == "composite":
        return bool(band.get("die_active") or band.get("emc_active"))
    return bool(band.get("active"))


def _segment_from_layer_stack(
    ls,
    cfg: MaterialInputConfig,
    active_map: dict[str, bool],
) -> dict[str, Any]:
    """One plate segment from solver LayerStack (bottom→top order)."""
    key = ls.material_key
    if key == MOLDING_COMPOSITE_KEY:
        band = _composite_band(cfg, active_map)
        band = {**band, "t_um": float(ls.thickness_um), "active": True}
        return band
    title, tag, mat_cls = _LAYER_STYLES.get(key, (key, key[:3].upper(), _MATERIAL_CSS_DEFAULT))
    return {
        "kind": "layer",
        "key": key,
        "title": title,
        "tag": tag,
        "mat_cls": mat_cls,
        "t_um": float(ls.thickness_um),
        "e_gpa": float(cfg.E_gpa.get(key, 0.0)),
        "active": True,
    }


def _solver_plate_segments_bottom_to_top(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    active_map: dict[str, bool],
) -> list[dict[str, Any]]:
    """Plate geometry = exact layers in warpage solver (single source of truth)."""
    stacks, _ = build_stack_for_step(
        step,
        packaging_architecture=cfg.packaging_architecture,
        substrate_thickness_um=cfg.thickness_um.get("Silicon Substrate"),
        layer_thickness_um=cfg.thickness_um,
        layer_active=active_map,
    )
    return [_segment_from_layer_stack(ls, cfg, active_map) for ls in stacks if ls.include]


def _segment_heights_px(segments_bottom_to_top: list[dict[str, Any]]) -> list[int]:
    """Partition _STACK_BAR_PX by physical thickness (no inter-band DOM gaps)."""
    total_t = sum(b["t_um"] for b in segments_bottom_to_top) or 1.0
    raw = [max(2, int(_STACK_BAR_PX * b["t_um"] / total_t)) for b in segments_bottom_to_top]
    drift = _STACK_BAR_PX - sum(raw)
    if drift and raw:
        raw[-1] = max(2, raw[-1] + drift)
    return raw


def _na_bottom_percent(z_na_um: float, total_t_um: float) -> float:
    """CSS bottom % for z_NA line (0 = stack bottom, 100 = stack top)."""
    if total_t_um <= 0:
        return 0.0
    ratio = max(0.0, min(1.0, float(z_na_um) / float(total_t_um)))
    return ratio * 100.0


def _na_marker_bottom_px(
    z_na_um: float,
    segments_bottom_to_top: list[dict[str, Any]],
    seg_heights_px: list[int],
) -> float:
    """Pixel offset from plate bottom (matches segment rounding, sub-layer accurate)."""
    total_t = sum(s["t_um"] for s in segments_bottom_to_top) or 1.0
    z = max(0.0, min(float(z_na_um), total_t))
    cum_t = 0.0
    cum_px = 0.0
    for seg, h_px in zip(segments_bottom_to_top, seg_heights_px):
        t_um = float(seg["t_um"])
        if z <= cum_t + t_um + 1e-9:
            frac = (z - cum_t) / t_um if t_um > 0 else 0.0
            return cum_px + frac * float(h_px)
        cum_t += t_um
        cum_px += float(h_px)
    return max(0.0, min(cum_px, float(_STACK_BAR_PX)))


def _html_pill(
    label: str,
    *,
    active: bool,
    mat_cls: str,
    flex_pct: float,
) -> str:
    state = "fowlp-pill--on" if active else "fowlp-pill--off"
    pct = max(8.0, min(92.0, flex_pct))
    return (
        f'<div class="fowlp-pill {state} {mat_cls}" style="flex: {pct:.2f} 1 0;" '
        f'title="{html.escape(label)}">'
        f'<span class="fowlp-pill-label">{html.escape(label)}</span>'
        f"</div>"
    )


def _html_plate_segment_layer(band: dict[str, Any], seg_px: int) -> str:
    active = band["active"]
    bar_state = "fowlp-mat-bar--active" if active else "fowlp-mat-bar--inactive"
    mat_cls = band.get("mat_cls", _MATERIAL_CSS_DEFAULT)
    return (
        f'<div class="fowlp-plate-segment" style="height:{seg_px}px" '
        f'title="{html.escape(band["title"])}">'
        f'<div class="fowlp-mat-bar fowlp-plate-bar {mat_cls} {bar_state}">'
        f'<span class="fowlp-band-tag">{html.escape(band["tag"])}</span>'
        f"</div></div>"
    )


def _html_plate_segment_composite(band: dict[str, Any], seg_px: int) -> str:
    f = float(band["die_frac"])
    die_pct = f * 100.0
    emc_pct = 100.0 - die_pct
    any_on = band["die_active"] or band["emc_active"]
    row_cls = "fowlp-composite-row fowlp-composite-row--on" if any_on else "fowlp-composite-row fowlp-composite-row--off"
    die_pill = _html_pill("DIE", active=band["die_active"], mat_cls=band["die_mat"], flex_pct=die_pct)
    emc_pill = _html_pill("EMC", active=band["emc_active"], mat_cls=band["emc_mat"], flex_pct=emc_pct)
    return (
        f'<div class="fowlp-plate-segment fowlp-plate-segment--composite" style="height:{seg_px}px">'
        f'<div class="{row_cls} fowlp-plate-bar">{die_pill}'
        f'<div class="fowlp-coplanar-divider" aria-hidden="true"></div>{emc_pill}</div></div>'
    )


def _html_legend_row(band: dict[str, Any]) -> str:
    if band["kind"] == "composite":
        active = band["die_active"] or band["emc_active"]
        band_cls = "fowlp-legend fowlp-legend--on" if active else "fowlp-legend fowlp-legend--off"
        die_pct = float(band["die_frac"]) * 100.0
        meta = (
            f"t = {band['t_um']:.0f} µm (EMC envelope) · Die {die_pct:.0f}% · "
            f"E_eff = {band['e_gpa']:.1f} GPa"
        )
        extra = f'<div class="fowlp-coplanar-note">{html.escape(COPLANAR_NOTE)}</div>'
    else:
        active = band["active"]
        band_cls = "fowlp-legend fowlp-legend--on" if active else "fowlp-legend fowlp-legend--off"
        meta = f"t = {band['t_um']:.0f} µm · E = {band['e_gpa']:.1f} GPa"
        extra = ""
    return f"""
<div class="{band_cls}">
  <div class="fowlp-legend-title">{html.escape(band['title'])}</div>
  {extra}
  <div class="fowlp-band-meta">{meta}</div>
</div>
"""


def _stack_visual_css() -> str:
    """Modern engineering UI: gradients, depth, glass edges, z_NA glow."""
    return """
/* --- Card shell --- */
.fowlp-stack-root {
  font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
  color: #0F172A;
  background: linear-gradient(165deg, #FFFFFF 0%, #F1F5F9 42%, #E8EEF4 100%);
  border: 1px solid rgba(255, 255, 255, 0.65);
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08), 0 2px 6px rgba(15, 23, 42, 0.04);
  overflow: hidden;
  width: 100%;
}
.fowlp-stack-header {
  padding: 18px 20px 14px;
  border-bottom: 1px solid rgba(226, 232, 240, 0.9);
  background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(248,250,252,0.85) 100%);
}
.fowlp-stack-header h3 {
  margin: 0 0 4px;
  font-size: 15px;
  font-weight: 700;
  letter-spacing: 0.02em;
}
.fowlp-stack-header p { margin: 0 0 10px; font-size: 12px; color: #64748B; }
.fowlp-badge-row { display: flex; flex-wrap: wrap; gap: 8px; }
.fowlp-badge {
  padding: 4px 11px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.02em;
  border: 1px solid rgba(255, 255, 255, 0.35);
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.06);
}
.fowlp-badge--step {
  background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
  color: #1D4ED8;
  border-color: rgba(191, 219, 254, 0.8);
}
.fowlp-badge--temp {
  background: linear-gradient(135deg, #FFF7ED 0%, #FFEDD5 100%);
  color: #C2410C;
  border-color: rgba(254, 215, 170, 0.8);
}
.fowlp-badge--na {
  background: linear-gradient(135deg, #FEF2F2 0%, #FECACA 100%);
  color: #B91C1C;
  border-color: rgba(254, 202, 202, 0.9);
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.2);
}
.fowlp-stack-axis {
  display: flex;
  justify-content: space-between;
  padding: 8px 20px 4px;
  font-size: 10px;
  color: #94A3B8;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.fowlp-stack-body {
  padding: 8px 16px 12px;
  background: linear-gradient(180deg, rgba(248,250,252,0.4) 0%, rgba(241,245,249,0.2) 100%);
}
.fowlp-stack-layout {
  display: grid;
  grid-template-columns: minmax(108px, 34%) 1fr;
  gap: 14px;
  align-items: stretch;
}

/* Continuous physical plate (no gaps between layers) */
.fowlp-stack-plate {
  position: relative;
  width: 100%;
  height: 200px;
  display: flex;
  flex-direction: column-reverse;
  gap: 0;
  border-radius: 10px;
  overflow: hidden;
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12), inset 0 1px 0 rgba(255, 255, 255, 0.35);
  border: 1px solid rgba(148, 163, 184, 0.35);
  background: rgba(241, 245, 249, 0.5);
}
.fowlp-plate-segment {
  flex: 0 0 auto;
  width: 100%;
  min-height: 2px;
  box-sizing: border-box;
}
.fowlp-plate-segment--composite .fowlp-composite-row {
  height: 100%;
  min-height: 100%;
}
.fowlp-plate-bar,
.fowlp-mat-bar.fowlp-plate-bar {
  position: relative;
  width: 100%;
  height: 100%;
  min-height: 100%;
  border-radius: 0;
  display: flex;
  align-items: center;
  overflow: hidden;
  box-sizing: border-box;
}
.fowlp-stack-legends {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  gap: 6px;
  min-height: 200px;
}
.fowlp-legend {
  padding: 8px 10px;
  border-radius: 8px;
  box-sizing: border-box;
}
.fowlp-legend--on {
  background: rgba(255, 255, 255, 0.72);
  border: 1px solid rgba(255, 255, 255, 0.5);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}
.fowlp-legend--off {
  background: rgba(248, 250, 252, 0.4);
  border: 1px dashed rgba(148, 163, 184, 0.45);
  opacity: 0.75;
}
.fowlp-legend-title {
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
  color: #0F172A;
}

/* Material bars — gradients @ 135deg */
.fowlp-mat-bar {
  position: relative;
  width: 100%;
  display: flex;
  align-items: center;
  overflow: hidden;
  box-sizing: border-box;
}
.fowlp-mat-bar--active {
  border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.35);
  opacity: 0.92;
}
.fowlp-mat-bar--inactive {
  background: rgba(226, 232, 240, 0.45) !important;
  border: 1px dashed rgba(148, 163, 184, 0.55) !important;
  box-shadow: none !important;
  opacity: 0.55;
}
.fowlp-mat--sub {
  background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 55%, #3B82F6 100%);
}
.fowlp-mat--int {
  background: linear-gradient(135deg, #0F766E 0%, #0D9488 50%, #14B8A6 100%);
}
.fowlp-mat--die {
  background: linear-gradient(135deg, #1E293B 0%, #334155 48%, #475569 100%);
}
.fowlp-mat--rdl {
  background: linear-gradient(135deg, #9A3412 0%, #C2410C 45%, #FB923C 100%);
}
.fowlp-mat--emc {
  background: linear-gradient(135deg, #064E3B 0%, #047857 42%, #0D9488 100%);
}
.fowlp-mat--car {
  background: linear-gradient(135deg, #292524 0%, #44403C 50%, #78716C 100%);
}
.fowlp-mat--sld {
  background: linear-gradient(135deg, #92400E 0%, #D97706 50%, #FCD34D 100%);
}
.fowlp-mat--default {
  background: linear-gradient(135deg, #64748B 0%, #94A3B8 100%);
}
.fowlp-band-tag {
  margin-left: 14px;
  font-size: 12px;
  font-weight: 700;
  color: #FFFFFF;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  letter-spacing: 0.04em;
  z-index: 1;
}
.fowlp-mat-bar--inactive .fowlp-band-tag,
.fowlp-pill--off .fowlp-pill-label {
  color: #94A3B8;
  text-shadow: none;
}

/* Composite molding row */
.fowlp-composite-row {
  display: flex;
  flex-direction: row;
  align-items: stretch;
  width: 100%;
  border-radius: 10px;
  overflow: hidden;
  box-sizing: border-box;
}
.fowlp-composite-row--on {
  border: 1px solid rgba(255, 255, 255, 0.35);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.25);
}
.fowlp-composite-row--off {
  border: 1px dashed rgba(148, 163, 184, 0.5);
  box-shadow: none;
  opacity: 0.65;
}
.fowlp-pill {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 0;
  overflow: hidden;
  border-radius: 0;
}
.fowlp-pill--on {
  border: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.28);
}
.fowlp-pill--off {
  background: rgba(226, 232, 240, 0.4) !important;
  opacity: 0.5;
}
.fowlp-pill--on .fowlp-pill-label {
  color: #FFFFFF;
  font-size: 12px;
  font-weight: 700;
  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);
  letter-spacing: 0.05em;
}
.fowlp-coplanar-divider {
  width: 2px;
  flex: 0 0 2px;
  background: repeating-linear-gradient(
    180deg,
    rgba(220, 38, 38, 0.9) 0,
    rgba(220, 38, 38, 0.9) 4px,
    transparent 4px,
    transparent 7px
  );
  box-shadow: 0 0 6px rgba(239, 68, 68, 0.35);
}

.fowlp-band-meta {
  font-size: 11px;
  color: #64748B;
  line-height: 1.4;
}
.fowlp-coplanar-note {
  font-size: 10px;
  font-weight: 600;
  color: #DC2626;
  line-height: 1.25;
}
.fowlp-status {
  font-size: 10px;
  font-style: italic;
  color: #94A3B8;
  line-height: 1.2;
  margin-top: 2px;
}

/* z_NA — absolute overlay on continuous plate (physics z from bottom) */
.fowlp-zna {
  position: absolute;
  left: 0;
  right: 0;
  pointer-events: none;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 4px;
  transform: translateY(1px);
}
.fowlp-zna-line {
  flex: 1;
  height: 0;
  border: none;
  border-top: 2px dashed #EF4444;
  box-shadow: 0 0 10px rgba(239, 68, 68, 0.65), 0 0 18px rgba(239, 68, 68, 0.28);
  filter: drop-shadow(0 0 3px rgba(239, 68, 68, 0.55));
}
.fowlp-zna-label {
  flex: 0 0 auto;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.05em;
  color: #FEF2F2;
  text-transform: uppercase;
  padding: 3px 8px;
  border-radius: 6px;
  background: rgba(185, 28, 28, 0.88);
  border: 1px solid rgba(254, 202, 202, 0.9);
  box-shadow: 0 0 12px rgba(239, 68, 68, 0.45), 0 2px 6px rgba(0, 0, 0, 0.25);
  text-shadow: 0 0 6px rgba(0, 0, 0, 0.85), 0 1px 2px rgba(0, 0, 0, 0.9),
    0 0 2px #FFFFFF;
}
.fowlp-stack-footer {
  padding: 12px 20px 16px;
  border-top: 1px solid rgba(226, 232, 240, 0.9);
  font-size: 11px;
  color: #64748B;
  display: flex;
  justify-content: space-between;
  background: rgba(255, 255, 255, 0.5);
}
"""


def build_stack_schematic_html(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    step_label: str,
    *,
    layer_active: dict[str, bool] | None = None,
    z_na_um: float | None = None,
    temperature_c: float | None = None,
) -> str:
    active_map = dict(layer_active if layer_active is not None else cfg.layer_active)
    plate_segments = _solver_plate_segments_bottom_to_top(cfg, step, active_map)
    if not plate_segments:
        return "<p style='color:#64748B;font-family:system-ui'>No layer data</p>"

    seg_heights = _segment_heights_px(plate_segments)
    total_t = sum(s["t_um"] for s in plate_segments) or _plate_thickness_um(cfg, step, active_map)
    legend_bands = _active_legend_bands_top_first(cfg, step, active_map)

    plate_parts: list[str] = []
    for band, seg_px in zip(plate_segments, seg_heights):
        if band["kind"] == "composite":
            plate_parts.append(_html_plate_segment_composite(band, seg_px))
        else:
            plate_parts.append(_html_plate_segment_layer(band, seg_px))

    legend_parts = [_html_legend_row(b) for b in legend_bands]

    z_marker = ""
    if z_na_um is not None and total_t > 0 and plate_segments:
        na_bottom_px = _na_marker_bottom_px(z_na_um, plate_segments, seg_heights)
        na_bottom_pct = _na_bottom_percent(z_na_um, total_t)
        z_marker = (
            f'<div class="fowlp-zna" style="bottom:{na_bottom_px:.2f}px;" '
            f'title="z_NA = {z_na_um:.1f} µm / Σt = {total_t:.1f} µm '
            f'({na_bottom_pct:.1f}% from stack bottom)">'
            f'<span class="fowlp-zna-line" aria-hidden="true"></span>'
            f'<span class="fowlp-zna-label">z_NA · {z_na_um:.1f} µm</span></div>'
        )

    step_esc = html.escape(step_label)
    temp_badge = (
        f'<span class="fowlp-badge fowlp-badge--temp">{temperature_c:.0f} °C</span>'
        if temperature_c is not None
        else ""
    )
    z_badge = (
        f'<span class="fowlp-badge fowlp-badge--na">z_NA = {z_na_um:.1f} µm</span>'
        if z_na_um is not None
        else ""
    )
    active_n = len(plate_segments)

    css = _stack_visual_css()
    return f"""
<div class="fowlp-stack-root">
<style>{css}</style>
<div class="fowlp-stack-header">
  <h3>Multi-Layer Stack</h3>
  <p>Top → bottom · coplanar molding · 135° material gradients</p>
  <div class="fowlp-badge-row">
    <span class="fowlp-badge fowlp-badge--step">{step_esc}</span>
    {temp_badge}
    {z_badge}
  </div>
</div>
<div class="fowlp-stack-axis"><span>↑ TOP</span><span>↓ BOTTOM (z=0)</span></div>
<div class="fowlp-stack-body">
  <div class="fowlp-stack-layout">
    <div class="fowlp-stack-plate" style="height:{_STACK_BAR_PX}px">
      {z_marker}
      {''.join(plate_parts)}
    </div>
    <div class="fowlp-stack-legends">
      {''.join(legend_parts)}
    </div>
  </div>
</div>
<div class="fowlp-stack-footer">
  <span>Active elements: {active_n}</span>
  <span>Σt (plate) = {total_t:.0f} µm</span>
</div>
</div>
"""


def schematic_height_px(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    *,
    layer_active: dict[str, bool] | None = None,
) -> int:
    active_map = dict(layer_active if layer_active is not None else cfg.layer_active)
    plate_segments = _solver_plate_segments_bottom_to_top(cfg, step, active_map)
    legend_rows = max(len(plate_segments), 1)
    body_h = max(_STACK_BAR_PX, legend_rows * 52)
    return int(118 + body_h + 40)


def _layer_rows(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    *,
    layer_active: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    active_map = dict(layer_active if layer_active is not None else cfg.layer_active)
    return _active_legend_bands_top_first(cfg, step, active_map)


def render_stack_schematic(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    step_label: str,
    *,
    layer_active: dict[str, bool] | None = None,
    z_na_um: float | None = None,
    temperature_c: float | None = None,
) -> None:
    """Render stack card in iframe (flexbox — coplanar molding row)."""
    import streamlit.components.v1 as components

    html_doc = build_stack_schematic_html(
        cfg,
        step,
        step_label,
        layer_active=layer_active,
        z_na_um=z_na_um,
        temperature_c=temperature_c,
    )
    components.html(
        html_doc,
        height=schematic_height_px(cfg, step, layer_active=layer_active),
        scrolling=False,
    )
