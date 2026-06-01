"""Multi-layer stack schematic — flexbox layout, coplanar EMC+Die molding band."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING, Any

from composite_layer import (
    COPLANAR_NOTE,
    DEFAULT_DIE_AREA_FRACTION,
    DIE_KEY,
    EMC_KEY,
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

# Minimum band chrome (px) — prevents label overlap
_BAND_META_PX = 58
_BAND_GAP_PX = 8
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
    """Show EMC+Die coplanar band only when at least one is in this process step."""
    return bool(active_map.get(DIE_KEY, False) or active_map.get(EMC_KEY, False))


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


def _visual_bands_top_first(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    *,
    layer_active: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    """Display order: top (RDL / EMC side) → bottom (Carrier / Substrate)."""
    _ = step
    return list(reversed(_bands_bottom_to_top(cfg, layer_active=layer_active)))


def _band_heights_px(bands: list[dict[str, Any]]) -> list[int]:
    total_t = sum(b["t_um"] for b in bands) or 1.0
    raw = [max(40, int(_STACK_BAR_PX * b["t_um"] / total_t)) for b in bands]
    scale = sum(raw) / _STACK_BAR_PX
    return [max(40, int(h / scale)) for h in raw]


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


def _html_band_layer(band: dict[str, Any], bar_px: int) -> str:
    active = band["active"]
    band_cls = "fowlp-band fowlp-band--on" if active else "fowlp-band fowlp-band--off"
    bar_state = "fowlp-mat-bar--active" if active else "fowlp-mat-bar--inactive"
    mat_cls = band.get("mat_cls", _MATERIAL_CSS_DEFAULT)
    status = "" if active else '<div class="fowlp-status">not in this step</div>'

    return f"""
<article class="{band_cls}" style="min-height:{bar_px + _BAND_META_PX}px">
  <div class="fowlp-band-bar-wrap">
    <div class="fowlp-band-bar fowlp-mat-bar {mat_cls} {bar_state}" style="height:{bar_px}px">
      <span class="fowlp-band-tag">{html.escape(band['tag'])}</span>
    </div>
  </div>
  <div class="fowlp-band-info">
    <div class="fowlp-band-title">{html.escape(band['title'])}</div>
    <div class="fowlp-band-meta">t = {band['t_um']:.0f} µm · E = {band['e_gpa']:.1f} GPa</div>
    {status}
  </div>
</article>
"""


def _html_band_composite(band: dict[str, Any], bar_px: int) -> str:
    f = float(band["die_frac"])
    die_pct = f * 100.0
    emc_pct = 100.0 - die_pct
    any_on = band["die_active"] or band["emc_active"]
    band_cls = "fowlp-band fowlp-band--on" if any_on else "fowlp-band fowlp-band--off"
    row_cls = "fowlp-composite-row fowlp-composite-row--on" if any_on else "fowlp-composite-row fowlp-composite-row--off"

    die_pill = _html_pill(
        "DIE", active=band["die_active"], mat_cls=band["die_mat"], flex_pct=die_pct
    )
    emc_pill = _html_pill(
        "EMC", active=band["emc_active"], mat_cls=band["emc_mat"], flex_pct=emc_pct
    )

    return f"""
<article class="{band_cls} fowlp-band--composite" style="min-height:{bar_px + _BAND_META_PX + 22}px">
  <div class="fowlp-band-bar-wrap">
    <div class="{row_cls}" style="height:{bar_px}px">
      {die_pill}
      <div class="fowlp-coplanar-divider" aria-hidden="true"></div>
      {emc_pill}
    </div>
  </div>
  <div class="fowlp-band-info">
    <div class="fowlp-band-title">{html.escape(band['title'])}</div>
    <div class="fowlp-coplanar-note">{html.escape(COPLANAR_NOTE)}</div>
    <div class="fowlp-band-meta">t = {band['t_um']:.0f} µm (EMC envelope) · Die {die_pct:.0f}% · E_eff = {band['e_gpa']:.1f} GPa</div>
  </div>
</article>
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
  position: relative;
  padding: 8px 16px 12px;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(248,250,252,0.4) 0%, rgba(241,245,249,0.2) 100%);
}
.fowlp-stack-column {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

/* --- Band cards --- */
.fowlp-band {
  display: grid;
  grid-template-columns: minmax(120px, 38%) 1fr;
  gap: 14px;
  padding: 10px 12px;
  border-radius: 10px;
  overflow: hidden;
  box-sizing: border-box;
  transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.fowlp-band--on {
  background: rgba(255, 255, 255, 0.78);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.45);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08);
}
.fowlp-band--off {
  background: rgba(248, 250, 252, 0.35);
  border: 1px dashed rgba(148, 163, 184, 0.45);
  box-shadow: none;
  opacity: 0.72;
}
.fowlp-band-bar-wrap { display: flex; align-items: stretch; min-width: 0; }

/* Material bars — gradients @ 135deg */
.fowlp-mat-bar {
  position: relative;
  width: 100%;
  border-radius: 10px;
  display: flex;
  align-items: center;
  overflow: hidden;
  box-sizing: border-box;
}
.fowlp-mat-bar--active {
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.35);
}
.fowlp-mat-bar--inactive {
  background: rgba(226, 232, 240, 0.35) !important;
  border: 1px dashed rgba(148, 163, 184, 0.55) !important;
  box-shadow: none !important;
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

/* Info column */
.fowlp-band-info {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
  min-height: 58px;
  overflow: hidden;
  min-width: 0;
}
.fowlp-band-title {
  font-size: 13px;
  font-weight: 600;
  line-height: 1.3;
  color: #0F172A;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.fowlp-band--off .fowlp-band-title { color: #94A3B8; }
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

/* z_NA — glow + badge */
.fowlp-zna {
  position: absolute;
  left: 16px;
  right: 16px;
  pointer-events: none;
  z-index: 4;
  display: flex;
  align-items: center;
  gap: 8px;
}
.fowlp-zna-line {
  flex: 1;
  height: 0;
  border: none;
  border-top: 2px dashed #EF4444;
  box-shadow: 0 0 8px rgba(239, 68, 68, 0.55), 0 0 16px rgba(239, 68, 68, 0.22);
  filter: drop-shadow(0 0 2px rgba(239, 68, 68, 0.4));
}
.fowlp-zna-label {
  flex: 0 0 auto;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: 0.06em;
  color: #991B1B;
  text-transform: uppercase;
  padding: 4px 12px;
  border-radius: 8px;
  background: linear-gradient(135deg, rgba(254, 226, 226, 0.98) 0%, rgba(252, 165, 165, 0.95) 100%);
  border: 1px solid rgba(239, 68, 68, 0.45);
  box-shadow: 0 0 14px rgba(239, 68, 68, 0.35), 0 2px 8px rgba(0, 0, 0, 0.08);
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
    bands = _visual_bands_top_first(cfg, step, layer_active=active_map)
    if not bands:
        return "<p style='color:#64748B;font-family:system-ui'>No layer data</p>"

    bar_heights = _band_heights_px(bands)
    total_t = _plate_thickness_um(cfg, step, active_map)
    stack_body_px = sum(bar_heights) + len(bands) * (_BAND_META_PX + _BAND_GAP_PX)

    parts: list[str] = []
    for band, bar_px in zip(bands, bar_heights):
        if band["kind"] == "composite":
            parts.append(_html_band_composite(band, bar_px))
        else:
            parts.append(_html_band_layer(band, bar_px))

    z_marker = ""
    if z_na_um is not None and total_t > 0:
        z_ratio = max(0.0, min(1.0, z_na_um / total_t))
        z_bottom_pct = (1.0 - z_ratio) * 100.0
        z_marker = (
            f'<div class="fowlp-zna" style="bottom:{z_bottom_pct:.2f}%" '
            f'title="Neutral axis from stack bottom">'
            f'<span class="fowlp-zna-line"></span>'
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
    active_n = sum(
        1
        for b in bands
        if (
            b["kind"] == "composite"
            and (b.get("die_active") or b.get("emc_active"))
        )
        or (b.get("kind") == "layer" and b.get("active"))
    )

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
<div class="fowlp-stack-axis"><span>↑ TOP</span><span>↓ BOTTOM</span></div>
<div class="fowlp-stack-body" style="min-height:{stack_body_px + 12}px">
  {z_marker}
  <div class="fowlp-stack-column">
    {''.join(parts)}
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
    bands = _visual_bands_top_first(cfg, step, layer_active=layer_active)
    bar_heights = _band_heights_px(bands)
    body = sum(bar_heights) + len(bands) * (_BAND_META_PX + _BAND_GAP_PX)
    extra = 22 if any(b["kind"] == "composite" for b in bands) else 0
    return int(120 + body + extra + 36)


def _layer_rows(
    cfg: MaterialInputConfig,
    step: ProcessStep,
    *,
    layer_active: dict[str, bool] | None = None,
) -> list[dict[str, Any]]:
    return _visual_bands_top_first(cfg, step, layer_active=layer_active)


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
