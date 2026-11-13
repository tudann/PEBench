from __future__ import annotations

DIMENSIONS = {
    "SF": ("semantic_fidelity", (
        "GSF", "SAP", "ANF", "DLI", "ARP", "CCP", "LCF", "SSP", "OTI")),
    "IC": ("internal_consistency", (
        "SAC", "ANL", "TSC", "SPC", "PL", "AVS")),
    "DC": ("directorial_creation", (
        "STC", "CDS", "ETD", "LCS", "NDE", "CIF", "SDI", "SPS", "VSS")),
}

CRITERIA = {
    "GSF": "Preserve the requested genre, medium, aesthetic and visual style. Replacing the medium, weakening a defining genre convention, or adding an incompatible reference style is an issue.",
    "SAP": "Preserve subject identity, count, appearance, attributes, objects and their bindings. For I2V, treat the first frame as the initial-state anchor; for R2V, follow each reference asset's instruction-specified role.",
    "ANF": "Preserve required actions, interactions, event order, outcomes and narrative intent. Omissions, substitutions, reversals, or added events that alter the requested event chain are issues.",
    "DLI": "Preserve dialogue, narration and lyrics, including wording, language, delivery and speaker attribution. Exact quotations must not be paraphrased, deleted, translated or converted unless explicitly allowed.",
    "ARP": "Preserve requested speech, vocalization, music, sound effects, ambience and silence constraints. Missing, substituted or added sounds that violate a requirement such as 'no music' are issues.",
    "CCP": "Preserve explicit camera constraints such as shot count, scale, angle, lens, focus, movement, single-take or static-camera requirements. Added camera design is allowed only when it does not override explicit intent.",
    "LCF": "Preserve specified time of day, illumination, light quality, brightness, palette, saturation, contrast and grading. Compatible elaboration is allowed; changing a defining condition is an issue.",
    "SSP": "Preserve the specified setting, background elements, layout, relative positions and movement directions. Changing the location, removing a required object or reversing a spatial relation is an issue.",
    "OTI": "Preserve visible titles, signs, subtitles and interface text, including exact wording, spelling, language, placement and timing. Visible text cannot be replaced with semantically similar speech.",
    "SAC": "Check that subject identity, count, appearance, clothing, props and persistent attributes remain stable across shots. Unexplained renaming, duplication, disappearance or attribute drift is an issue.",
    "ANL": "Check that actions form a coherent causal or procedural progression, with plausible prerequisites, bindings between subjects and actions, reactions and resolution. Incompatible or dangling events are issues.",
    "TSC": "Check chronology, duration allocation, event order and state transitions. Overlapping or infeasible timestamps, unexplained resets, or later shots showing states that should no longer exist are issues.",
    "SPC": "Check consistency of location, screen direction, relative position, orientation, trajectory and entrances or exits. Unexplained position swaps or movement paths that cannot connect adjacent states are issues.",
    "PL": "Check compliance with real-world physics or established fictional rules, including gravity, contact, object permanence, bodily feasibility and causal relations. Judge deliberate fantasy by its own world logic.",
    "AVS": "Check temporal and semantic alignment between visible events and dialogue, lyrics, music, effects or ambience. Wrong speakers, premature effects, and misaligned lip-sync or beat-sync actions are issues.",
    "STC": "Assess whether content is organized into a feasible temporal plan with adequate event or shot boundaries, ordering and duration allocation.",
    "CDS": "Assess actionable specifications for framing, shot scale, angle, viewpoint, lens, focus and camera movement.",
    "ETD": "Assess whether cuts and transitions support temporal, spatial, narrative or rhythmic progression. Disconnected shots or repeated transitions without continuity or motivation are inadequate.",
    "LCS": "Assess actionable specification of light source, direction, quality, contrast, palette, saturation and grading.",
    "NDE": "Assess whether the prompt provides sufficient progression, variation, escalation or resolution for the requested content form. Static restatement or unmotivated repetition is inadequate.",
    "CIF": "Assess whether additions support the central request. Repetitive padding, template language, unnecessary subplots or decorative details that distract from the intended result are inadequate.",
    "SDI": "Assess content-appropriate integration of speech, music, effects and ambience through source, character, intensity, progression and visual timing. Do not require every audio type.",
    "SPS": "Assess generation-relevant subject and performance detail, including appearance, pose, expression, gesture, movement, interaction, gaze and vocal delivery.",
    "VSS": "Assess actionable specification of medium, texture, rendering language, production design and visual treatment.",
}

CRITERION_DIMENSION = {code: key for key, (_, codes) in DIMENSIONS.items() for code in codes}
RUBRIC_VERSION = "pebench-24-v1"
EVIDENCE_PROMPT_VERSION = "facts-v1"


def dimension_name(code: str) -> str:
    return DIMENSIONS[code][0]


def criteria_for(code: str) -> tuple[str, ...]:
    return DIMENSIONS[code][1]
