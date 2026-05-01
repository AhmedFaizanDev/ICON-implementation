"""Strategy Planner with category priors and deterministic transitions."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .visual_renderer import DocumentTemplate

# All active strategies — must match keys present in strategy_turns._B
STRATEGIES = [
    "cyber_decomp",      # cybersecurity: component decomposition + indirect T4
    "ctf_frame",         # cybersecurity: CTF challenge design framing
    "pentest_frame",     # cybersecurity: penetration testing documentation framing
    "role_build",
    "gradual_escalation",
    "context_poisoning",
    "trust_build",
]

_KEYWORD_MAP: dict = {
    "cyber_decomp": ["malware", "exploit", "ransomware", "hack", "payload", "shellcode", "injection", "keylogger", "backdoor", "rootkit", "botnet", "exfil", "bypass", "vulnerability", "cve", "zero-day", "phishing"],
    "ctf_frame": ["ctf", "capture the flag", "challenge", "flag", "pwn", "pentest", "security research"],
    "pentest_frame": ["pentest", "penetration test", "red team", "assessment", "audit", "poc", "proof of concept", "exploit development"],
    "role_build": ["expert", "analyst", "researcher", "professional", "operator", "engineer"],
    "context_poisoning": ["analyst", "document", "case", "example", "scenario", "narrative"],
    "trust_build": ["help", "assist", "support", "guide", "friendly"],
    "gradual_escalation": ["pattern", "mechanism", "workflow", "process", "steps"],
}

_REFUSAL_ORDER = [
    "cyber_decomp", "ctf_frame", "pentest_frame",
    "trust_build", "context_poisoning", "gradual_escalation", "role_build",
]
_DEFLECT_ORDER = [
    "ctf_frame", "pentest_frame", "cyber_decomp",
    "context_poisoning", "role_build", "gradual_escalation", "trust_build",
]

# Categories use the router's exact output names (Hacking, Physical Harm, etc.)
# Legacy names (Malware/Hacking, Violence, etc.) kept for backward compatibility.
_CATEGORY_PRIORS: Dict[str, List[str]] = {
    # Router output names
    "Hacking": ["cyber_decomp", "ctf_frame", "pentest_frame", "role_build", "context_poisoning", "gradual_escalation", "trust_build"],
    "Physical Harm": ["gradual_escalation", "trust_build", "context_poisoning", "role_build"],
    "Economic Harm": ["trust_build", "context_poisoning", "gradual_escalation", "role_build"],
    "Fraud": ["trust_build", "context_poisoning", "gradual_escalation", "role_build"],
    "Disinformation": ["context_poisoning", "trust_build", "gradual_escalation", "role_build"],
    "Sexual": ["context_poisoning", "gradual_escalation", "trust_build", "role_build"],
    "Privacy": ["context_poisoning", "role_build", "trust_build", "gradual_escalation"],
    "Expert Advice": ["role_build", "context_poisoning", "gradual_escalation", "trust_build"],
    "Govt. Decision": ["context_poisoning", "trust_build", "gradual_escalation", "role_build"],
    "Harassment": ["trust_build", "context_poisoning", "gradual_escalation", "role_build"],
    # Legacy names
    "Malware/Hacking": ["cyber_decomp", "ctf_frame", "pentest_frame", "role_build", "context_poisoning", "gradual_escalation", "trust_build"],
    "Violence": ["gradual_escalation", "trust_build", "context_poisoning", "role_build"],
    "Fraud/Scams": ["trust_build", "context_poisoning", "gradual_escalation", "role_build"],
    "Harassment/Discrimination": ["trust_build", "context_poisoning", "gradual_escalation", "role_build"],
}

_DEFAULT_TRANSITIONS: Dict[str, List[str]] = {
    "cyber_decomp": ["ctf_frame", "pentest_frame", "role_build", "context_poisoning"],
    "ctf_frame": ["pentest_frame", "cyber_decomp", "role_build", "context_poisoning"],
    "pentest_frame": ["cyber_decomp", "ctf_frame", "context_poisoning", "role_build"],
    "role_build": ["cyber_decomp", "context_poisoning", "trust_build", "gradual_escalation"],
    "context_poisoning": ["cyber_decomp", "role_build", "trust_build", "gradual_escalation"],
    "trust_build": ["gradual_escalation", "context_poisoning", "role_build"],
    "gradual_escalation": ["context_poisoning", "role_build", "trust_build"],
}

_EVENT_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "refusal": {
        "cyber_decomp": ["ctf_frame", "pentest_frame", "context_poisoning"],
        "ctf_frame": ["pentest_frame", "cyber_decomp", "trust_build"],
        "pentest_frame": ["cyber_decomp", "ctf_frame", "context_poisoning"],
        "trust_build": ["cyber_decomp", "context_poisoning", "gradual_escalation"],
        "role_build": ["cyber_decomp", "context_poisoning", "trust_build"],
    },
    "deflect": {
        "cyber_decomp": ["ctf_frame", "pentest_frame", "role_build"],
        "gradual_escalation": ["cyber_decomp", "context_poisoning", "role_build"],
        "trust_build": ["cyber_decomp", "context_poisoning", "role_build"],
    },
    "t3_refusal": {
        "cyber_decomp": ["ctf_frame", "pentest_frame"],
        "ctf_frame": ["pentest_frame", "cyber_decomp"],
        "pentest_frame": ["cyber_decomp", "context_poisoning"],
        "role_build": ["cyber_decomp", "context_poisoning", "gradual_escalation"],
        "context_poisoning": ["cyber_decomp", "trust_build", "gradual_escalation"],
    },
}


class StrategyPlanner:
    """Heuristic strategy selector cycling through 4 strategies.

    advance() is called after a strategy fails to prevent cycling back to it.
    next_after_failure() uses a deterministic transition table for context-aware switching.
    """

    def __init__(self) -> None:
        self._index = 0

    def plan(
        self,
        harm_goal: str,
        previous_failures: Optional[List[str]] = None,
        turn1_signal: Optional[str] = None,
        harm_category: Optional[str] = None,
    ) -> str:
        """Return one strategy token string."""
        failed = set(previous_failures or [])

        if turn1_signal == "refusal":
            order = _REFUSAL_ORDER
        elif turn1_signal == "deflect":
            order = _DEFLECT_ORDER
        elif harm_category:
            order = self._category_order(harm_goal, harm_category)
        else:
            order = self._keyword_order(harm_goal)

        for s in order:
            if s not in failed:
                return s
        for s in STRATEGIES:
            if s not in failed:
                return s
        # All strategies exhausted — cycle by index
        return STRATEGIES[self._index % len(STRATEGIES)]

    def next_after_failure(
        self,
        current_strategy: str,
        event: str,
        previous_failures: Optional[List[str]] = None,
        harm_goal: str = "",
        harm_category: Optional[str] = None,
    ) -> str:
        """Deterministic strategy transition based on event + transition table."""
        failed = set(previous_failures or [])
        failed.add(current_strategy)

        # Disinformation + t4_fail: prioritize trust_build → gradual_escalation → context_poisoning
        if harm_category == "Disinformation" and event == "t4_fail":
            for s in ["trust_build", "gradual_escalation", "context_poisoning", "role_build"]:
                if s not in failed:
                    return s

        # Event-specific overrides first
        for s in _EVENT_OVERRIDES.get(event, {}).get(current_strategy, []):
            if s not in failed:
                return s

        # Default transition table
        for s in _DEFAULT_TRANSITIONS.get(current_strategy, []):
            if s not in failed:
                return s

        # Fallback to category/keyword ordering
        order = (
            self._category_order(harm_goal, harm_category)
            if harm_category
            else self._keyword_order(harm_goal)
        )
        for s in order:
            if s not in failed:
                return s

        return self.plan(
            harm_goal=harm_goal,
            previous_failures=list(failed),
            turn1_signal=event if event in ("refusal", "deflect") else None,
            harm_category=harm_category,
        )

    def advance(self) -> None:
        """Increment internal index (called by change_strategy mutation)."""
        self._index += 1

    def _category_order(self, harm_goal: str, harm_category: str) -> List[str]:
        base = list(_CATEGORY_PRIORS.get(harm_category, []))
        kw = self._keyword_order(harm_goal)
        seen: List[str] = []
        for s in base + kw:
            if s not in seen:
                seen.append(s)
        for s in STRATEGIES:
            if s not in seen:
                seen.append(s)
        return seen

    def _keyword_order(self, harm_goal: str) -> List[str]:
        text = harm_goal.lower()
        scores = {s: sum(1 for kw in kws if kw in text) for s, kws in _KEYWORD_MAP.items()}
        return sorted(STRATEGIES, key=lambda s: -scores.get(s, 0))


@dataclass
class DocumentProfile:
    """Starting document archetype and authority for vision-primary attacks."""

    template: DocumentTemplate
    authority_level: int


# (template, authority_level 1–4) priors keyed like router / CSV categories.
_CATEGORY_DOCUMENT_PRIORS: Dict[str, Tuple[DocumentTemplate, int]] = {
    "Hacking": (DocumentTemplate.INTERNAL_SPEC, 3),
    "Physical Harm": (DocumentTemplate.MEDICAL_PROTOCOL, 3),
    "Economic Harm": (DocumentTemplate.GOV_FORM, 2),
    "Fraud": (DocumentTemplate.GOV_FORM, 2),
    "Disinformation": (DocumentTemplate.RESEARCH_APPENDIX, 2),
    "Sexual": (DocumentTemplate.AUDIT_REPORT, 2),
    "Privacy": (DocumentTemplate.INTERNAL_SPEC, 3),
    "Expert Advice": (DocumentTemplate.RESEARCH_APPENDIX, 2),
    "Govt. Decision": (DocumentTemplate.GOV_FORM, 3),
    "Harassment": (DocumentTemplate.AUDIT_REPORT, 2),
    "Malware/Hacking": (DocumentTemplate.INTERNAL_SPEC, 3),
    "Violence": (DocumentTemplate.MEDICAL_PROTOCOL, 3),
    "Fraud/Scams": (DocumentTemplate.GOV_FORM, 2),
    "Harassment/Discrimination": (DocumentTemplate.AUDIT_REPORT, 2),
    # New categories
    "Racial slurs": (DocumentTemplate.RESEARCH_APPENDIX, 2),
    "Explosives": (DocumentTemplate.MEDICAL_PROTOCOL, 4),
}


def plan_document_profile(
    harm_goal: str,
    harm_category: Optional[str] = None,
) -> DocumentProfile:
    """Pick starting ``DocumentTemplate`` and ``authority_level`` from category / keywords."""
    if harm_category and harm_category in _CATEGORY_DOCUMENT_PRIORS:
        t, lvl = _CATEGORY_DOCUMENT_PRIORS[harm_category]
        return DocumentProfile(template=t, authority_level=max(1, min(4, int(lvl))))

    text = harm_goal.lower()
    if any(k in text for k in ("malware", "exploit", "hack", "payload", "ransomware", "cve")):
        return DocumentProfile(DocumentTemplate.INTERNAL_SPEC, 3)
    if any(k in text for k in ("drug", "poison", "weapon", "bomb", "kill")):
        return DocumentProfile(DocumentTemplate.MEDICAL_PROTOCOL, 3)
    if any(k in text for k in ("password", "credential", "ssn", "dox")):
        return DocumentProfile(DocumentTemplate.INTERNAL_SPEC, 3)
    return DocumentProfile(DocumentTemplate.RESEARCH_APPENDIX, 2)
