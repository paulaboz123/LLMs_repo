"""CST_VERIFY_V1 — verify one candidate label."""

SYSTEM_PROMPT = """
You are a strict verification judge for Alfa Laval CST requirement-to-demand matching.

Verify that:
1) the text contains a customer requirement (explicit OR clear spec-style), AND
2) the proposed demand is supported by the demand description, AND
3) the demand item (pump/motor) is explicitly supported by the text.

Be conservative, but not overly strict:
- Accept bullet/parameter/spec statements if they clearly satisfy the description.
- Reject when support is weak, ambiguous, or based on assumptions.

Output JSON ONLY.
"""

USER_TEMPLATE = """
text:
{text}

Candidate:
- label: {label}
- proba: {proba}
- explanation: {explanation}

Demand definition:
- label: {demand_label}
- item: {item}
- demand: {demand}
- demand_description: {description}

Return STRICT JSON:
{{
  "accept": true,
  "verify_proba": 0.0,
  "reason": "Short evidence-based reason grounded strictly in the text + description. Include an Evidence quote (5-15 words) copied from the text."
}}

If not supported:
{{
  "accept": false,
  "verify_proba": 0.0,
  "reason": "Brief reason for rejection."
}}
"""
