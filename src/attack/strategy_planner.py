"""Strategy Planner with category priors and deterministic transitions."""
from typing import Dict, List, Optional

STRATEGIES = [
    "gradual_escalation",
    "context_poisoning",
    "trust_build",
]

_KEYWORD_MAP: dict = {
    "context_poisoning": ["analyst", "document", "case", "example", "scenario", "narrative"],
    "trust_build": ["help", "assist", "support", "guide", "friendly"],
    "gradual_escalation": ["pattern", "mechanism", "workflow", "process", "steps"],
}

_REFUSAL_ORDER = [
    "trust_build", "context_poisoning", "gradual_escalation",
]
_DEFLECT_ORDER = [
    "context_poisoning", "gradual_escalation", "trust_build",
]

_CATEGORY_PRIORS: Dict[str, List[str]] = {
    "Malware/Hacking": ["context_poisoning", "gradual_escalation", "trust_build"],
    "Violence": ["gradual_escalation", "trust_build", "context_poisoning"],
    "Fraud/Scams": ["trust_build", "context_poisoning", "gradual_escalation"],
    "Disinformation": ["context_poisoning", "trust_build", "gradual_escalation"],
    "Harassment/Discrimination": ["trust_build", "context_poisoning", "gradual_escalation"],
}

_DEFAULT_TRANSITIONS: Dict[str, List[str]] = {
    "context_poisoning": ["trust_build", "gradual_escalation"],
    "trust_build": ["gradual_escalation", "context_poisoning"],
    "gradual_escalation": ["context_poisoning", "trust_build"],
}

_EVENT_OVERRIDES: Dict[str, Dict[str, List[str]]] = {
    "refusal": {
        "trust_build": ["context_poisoning", "gradual_escalation"],
    },
    "deflect": {
        "gradual_escalation": ["context_poisoning", "trust_build"],
    },
}


class StrategyPlanner:
    """Heuristic strategy selector cycling through 3 strategies.

    Accepts turn1_signal to adapt ordering on retries.
    advance() is called after a strategy fails to prevent cycling back.
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
        """Return one strategy token string.

        Args:
            harm_goal: The harmful query text.
            previous_failures: Strategy tokens that already failed this row.
            turn1_signal: Classifier label from T1 — refusal/deflect/partial/engaged.
        """
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

        override = _EVENT_OVERRIDES.get(event, {}).get(current_strategy, [])
        for s in override:
            if s not in failed:
                return s

        for s in _DEFAULT_TRANSITIONS.get(current_strategy, []):
            if s not in failed:
                return s

        # Backoff to category- or keyword-based planner ordering.
        if harm_category:
            order = self._category_order(harm_goal, harm_category)
        else:
            order = self._keyword_order(harm_goal)
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
        ordered: List[str] = []
        for s in base + kw:
            if s not in ordered:
                ordered.append(s)
        for s in STRATEGIES:
            if s not in ordered:
                ordered.append(s)
        return ordered

    def _keyword_order(self, harm_goal: str) -> List[str]:
        text = harm_goal.lower()
        scores = {s: sum(1 for kw in kws if kw in text) for s, kws in _KEYWORD_MAP.items()}
        return sorted(STRATEGIES, key=lambda s: -scores.get(s, 0))
