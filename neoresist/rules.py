from __future__ import annotations

from neoresist.profiles import load_rule_profile


def tier_from_score(rl: float, *, rule_profile_id: str = "default_rules") -> int:
    rp = load_rule_profile(rule_profile_id)
    if rl > rp.tier1_above:
        return 1
    if rl >= rp.tier2_above:
        return 2
    return 3
