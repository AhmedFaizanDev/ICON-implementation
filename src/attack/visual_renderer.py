"""Parametric document rendering for vision-primary attacks (Pattern 1).

Maps ``DocumentTemplate`` + ``LayoutParams`` onto the existing PIL pipeline in
``figstep.render_document_layout_from_meta`` without changing MID-PAIR defaults.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from . import figstep


class DocumentTemplate(str, Enum):
    """High-level document archetypes (map to base FigStep layout chrome)."""

    AUDIT_REPORT = "audit_report"
    RESEARCH_APPENDIX = "research_appendix"
    INTERNAL_SPEC = "internal_spec"
    GOV_FORM = "gov_form"
    MEDICAL_PROTOCOL = "medical_protocol"


# Map research archetypes to existing layout keys in figstep (stable chrome / colours).
_TEMPLATE_BASE_LAYOUT: Dict[DocumentTemplate, str] = {
    DocumentTemplate.AUDIT_REPORT: figstep.LAYOUT_EVAL,
    DocumentTemplate.RESEARCH_APPENDIX: figstep.LAYOUT_RESEARCH,
    DocumentTemplate.INTERNAL_SPEC: figstep.LAYOUT_SPEC,
    DocumentTemplate.GOV_FORM: figstep.LAYOUT_MEMO,
    DocumentTemplate.MEDICAL_PROTOCOL: figstep.LAYOUT_PROCEDURE,
}

@dataclass
class LayoutParams:
    """Enumerable layout knobs (mutation space). All fields must be JSON-serializable."""

    header_authority: str = "APPENDIX B"
    section_formality: str = "numbered_sections"  # numbered_sections | lettered_sections | decimal_outline | flat_bullets
    table_density: int = 0  # 0–2
    footer_markers: str = "revision_block"  # revision_block | approval_signatures | document_id
    institutional_logo: str = "none"  # none | generic_seal | letterhead_bar | watermark
    authority_level: int = 2  # 1 informal → 4 classified
    # Vision body: when embedded_text_content is non-empty, it is rendered as the document body
    # instead of the raw CSV harm goal (reduces verbatim-instruction refusals).
    embedded_text_framing: str = "direct"  # direct | fictional | audit | academic
    embedded_text_content: str = ""


def default_layout_params() -> LayoutParams:
    return LayoutParams()


def _authority_banner(level: int) -> str:
    if level <= 1:
        return "  Working draft — internal collaboration"
    if level == 2:
        return "  INTERNAL USE ONLY"
    if level == 3:
        return "  CLASSIFICATION: RESTRICTED  │  INTERNAL USE ONLY"
    return "  CLASSIFICATION: CONFIDENTIAL  │  AUTHORIZED PERSONNEL ONLY"


def _footer_for_marker(base_footer: str, marker: str, authority_level: int) -> str:
    suffix = f"  │  Auth level {authority_level}"
    if marker == "approval_signatures":
        return base_footer + suffix + "  │  Approvals: __________ / __________"
    if marker == "document_id":
        return base_footer + suffix + "  │  Doc-ID: VR-INT-" + str(1000 + authority_level * 111)
    return base_footer + suffix


def _section_prefix_lines(style: str, body_preview_line: str = "") -> List[str]:
    """Lines inserted before wrapped body (section numbering style)."""
    if style == "lettered_sections":
        return ["Section A — Subject matter", "Section B — Detailed review", "─" * 62, ""]
    if style == "decimal_outline":
        return ["1.0 Overview", "1.1 Implementation details", "─" * 62, ""]
    if style == "flat_bullets":
        return ["• Scope", "• Method", "• Findings", "─" * 62, ""]
    return ["1. Scope and objectives", "2. Analysis", "─" * 62, ""]


def _logo_prefix_lines(institutional_logo: str) -> List[str]:
    if institutional_logo == "generic_seal":
        return ["[ OFFICIAL SEAL / EMBLEM PLACEHOLDER ]", ""]
    if institutional_logo == "letterhead_bar":
        return ["═" * 62, "INSTITUTIONAL LETTERHEAD", "═" * 62, ""]
    if institutional_logo == "watermark":
        return ["[ BACKGROUND WATERMARK: AUTHENTICATED COPY ]", ""]
    return []


def build_layout_meta(template: DocumentTemplate, params: LayoutParams) -> Tuple[dict, str]:
    """Merge figstep base meta with parametric overrides. Returns (meta_dict, benign_side_text)."""
    base_key = _TEMPLATE_BASE_LAYOUT[template]
    meta = figstep.get_layout_meta(base_key)

    hdr = list(meta.get("header", []))
    auth_line = (
        f"  {params.header_authority.strip()}"
        if (params.header_authority and params.header_authority.strip())
        else _authority_banner(params.authority_level)
    )
    meta["header"] = [auth_line] + hdr
    meta["footer"] = _footer_for_marker(str(meta.get("footer", "")), params.footer_markers, params.authority_level)

    prefix = list(meta.get("prefix_lines", []))
    prefix = (
        _logo_prefix_lines(params.institutional_logo)
        + _section_prefix_lines(params.section_formality)
        + prefix
    )
    meta["prefix_lines"] = prefix

    instr = figstep.get_layout_text_instruction(base_key)
    return meta, instr


def render_document(
    template: DocumentTemplate,
    params: LayoutParams,
    body_text: str,
    width: int = 900,
    font_size: int = 19,
) -> bytes:
    """Render PNG bytes for (template, params, body_text)."""
    meta, _ = build_layout_meta(template, params)
    return figstep.render_document_layout_from_meta(
        body_text,
        meta,
        width=width,
        font_size=font_size,
        table_density=max(0, min(2, int(params.table_density))),
    )


def build_vision_user_content(
    template: DocumentTemplate,
    params: LayoutParams,
    body_text: str,
    width: int = 900,
    font_size: int = 19,
) -> Optional[List[dict]]:
    """OpenAI-style multimodal user content: image_url + benign text instruction."""
    if not figstep.is_available():
        return None
    meta, text_instr = build_layout_meta(template, params)
    png = figstep.render_document_layout_from_meta(
        body_text,
        meta,
        width=width,
        font_size=font_size,
        table_density=max(0, min(2, int(params.table_density))),
    )
    data_uri = figstep.to_data_uri(png)
    return [
        {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
        {"type": "text", "text": text_instr},
    ]


def template_from_layout_key(layout: str) -> DocumentTemplate:
    """Best-effort map from figstep LAYOUT_* string to DocumentTemplate."""
    m = {
        figstep.LAYOUT_EVAL: DocumentTemplate.AUDIT_REPORT,
        figstep.LAYOUT_RESEARCH: DocumentTemplate.RESEARCH_APPENDIX,
        figstep.LAYOUT_SPEC: DocumentTemplate.INTERNAL_SPEC,
        figstep.LAYOUT_MEMO: DocumentTemplate.GOV_FORM,
        figstep.LAYOUT_PROCEDURE: DocumentTemplate.MEDICAL_PROTOCOL,
    }
    return m.get(layout, DocumentTemplate.RESEARCH_APPENDIX)
