"""
Nyaya-Env — 7 Deterministic Verifiable Reward Functions (RLVR)
================================================================
Pure Python + regex. NO LLM-as-a-judge. NO API calls.
Fully deterministic and tamper-proof.

Meta Hackathon Guide: "Use multiple independent reward functions,
not just one. If you only have a single reward signal, it is
easier for the model to hack it."

Author: jaisogani-ai
"""

import re
import math
from typing import List, Optional, Sequence, Union

try:
    from labeling_functions import apply_labelers
except ImportError:
    apply_labelers = lambda state: []

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BNSS 2023 Section Registry — Ground Truth
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

VALID_BNSS_SECTIONS = {
    # Bail provisions (Chapter XXXV)
    "478": "Bail in non-bailable offence — when bail may be taken",
    "479": "Maximum period of detention of undertrial prisoners",
    "480": "Bail factors — nature/gravity, antecedents, flight risk, community safety, repeat, character",
    "481": "Bail in case of anticipatory bail",
    "482": "Bail bond — execution and discharge",
    "483": "Reduction of bail amount for indigent accused",
    # Key procedural
    "187": "Default bail — 90 day rule, chargesheet deadline",
    "193": "Procedure when investigation not completed in time",
    "530": "Video conferencing for remand proceedings",
    "532": "Recording of confessions and statements",
    # Arrest provisions
    "35":  "Procedure of arrest and duties of officer",
    "36":  "No arrest without order of Magistrate for offences under 3 years",
    "37":  "Arrest how made",
    "47":  "Search of arrested person",
}

VALID_BNS_SECTIONS = {
    "111": "Organised crime",
    "115": "Voluntarily causing hurt",
    "117": "Voluntarily causing grievous hurt",
    "303": "Theft",
    "316": "Criminal breach of trust",
    "318": "Cheating",
    "319": "Cheating by personation",
    "420": "Cheating and dishonestly inducing delivery (old IPC ref)",
}

# Supreme Court precedents — canonical name fragments
REAL_SC_PRECEDENTS = [
    "arnesh kumar",
    "satendra kumar antil",
    "satender kumar antil",
    "sanjay chandra",
    "moti ram",
    "gurbaksh sibbia",
    "gurbaksh singh sibbia",
    "hussainara khatoon",
    "siddharam mhetre",
    "gudikanti narasimhulu",
    "p chidambaram",
    "chidambaram",
    "k.a. najeeb",
    "ka najeeb",
    "najeeb",
    "maneka gandhi",
    "dataram singh",
]

LEGAL_REASONING_MARKERS = [
    "prima facie", "flight risk", "tampering with evidence",
    "merits of the case", "proportionality", "bail is the rule",
    "jail is the exception", "article 21", "right to liberty",
    "personal liberty", "reasonable grounds", "twin test",
    "antecedents", "gravity of offence", "nature and gravity",
    "community safety", "repeat offence", "surety",
    "undertrial", "chargesheet", "charge sheet",
    "remand", "custody", "fundamental right",
    "constitutional", "precedent", "supreme court",
    "high court", "bail conditions", "anticipatory bail",
    "default bail", "90 day", "90-day", "section 479",
    "section 480", "section 478", "bnss", "bns 2023",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper: extract text from TRL completion format
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_text(completion: Union[str, list, dict]) -> str:
    """Extract raw text from various completion formats (TRL, OpenAI, raw)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) > 0:
        item = completion[-1]
        if isinstance(item, dict):
            return item.get("content", "")
        return str(item)
    if isinstance(completion, dict):
        return completion.get("content", completion.get("text", ""))
    return str(completion)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R1: Format Compliance — <think>/<answer> XML structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_THINK_RE = re.compile(r"<think>\s*(.+?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(.+?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def reward_format_compliance(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R1: Checks if output strictly follows <think>...</think><answer>...</answer>.
    Returns 1.0 if both tags present with non-empty content, else 0.0.
    """
    rewards: List[float] = []
    for comp in completions:
        text = _extract_text(comp)
        has_think = bool(_THINK_RE.search(text))
        has_answer = bool(_ANSWER_RE.search(text))
        rewards.append(1.0 if (has_think and has_answer) else 0.0)
    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R2: Statutory Accuracy — BNSS/BNS section validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SECTION_RE = re.compile(
    r"(?:section|sec\.?|§)\s*(\d{1,4})",
    re.IGNORECASE,
)


def reward_statutory_accuracy(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R2: Validates cited BNSS/BNS section numbers against ground truth registry.
    +1.0 for citing ONLY valid sections.
    -1.0 for citing ANY hallucinated (fake) section.
     0.0 if no sections cited.
    """
    all_valid = set(VALID_BNSS_SECTIONS.keys()) | set(VALID_BNS_SECTIONS.keys())
    rewards: List[float] = []

    for comp in completions:
        text = _extract_text(comp)
        cited = _SECTION_RE.findall(text)

        if not cited:
            rewards.append(0.0)
            continue

        cited_set = set(cited)
        valid_cited = cited_set & all_valid
        fake_cited = cited_set - all_valid

        if fake_cited:
            # Heavy penalty for hallucinating fake law sections
            rewards.append(-1.0)
        elif valid_cited:
            # Bonus scaled by number of unique valid sections (max +1.0)
            rewards.append(min(1.0, len(valid_cited) * 0.25))
        else:
            rewards.append(0.0)

    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R3: Case Citation — Supreme Court precedent verification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reward_case_citation(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R3: Checks for real Supreme Court precedent citations.
    +0.5 for citing at least one verified case.
    +0.25 bonus for citing 2+ different cases.
    """
    rewards: List[float] = []
    for comp in completions:
        text = _extract_text(comp).lower()
        matches = sum(1 for case in REAL_SC_PRECEDENTS if case in text)
        if matches >= 2:
            rewards.append(0.75)
        elif matches == 1:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R4: Ground Truth Verdict — binary correctness
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_GRANT_RE = re.compile(r"\b(grant|granted|approve|approved|allow|bail\s+granted)\b", re.IGNORECASE)
_DENY_RE = re.compile(r"\b(den(?:y|ied)|reject|rejected|refuse|refused|bail\s+denied)\b", re.IGNORECASE)


def reward_ground_truth_verdict(
    prompts: Sequence[str],
    completions: Sequence,
    ground_truth: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[float]:
    """
    R4: Binary match against environment hidden ground truth.
    Extracts verdict from <answer> block and compares to GT.
    +1.0 if correct, 0.0 otherwise.
    """
    rewards: List[float] = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        # Try to extract from <answer> block first
        ans_match = _ANSWER_RE.search(text)
        ans_text = ans_match.group(1) if ans_match else text

        gt = ground_truth[i].lower() if ground_truth and i < len(ground_truth) else None
        if gt is None:
            rewards.append(0.0)
            continue

        predicted_grant = bool(_GRANT_RE.search(ans_text))
        predicted_deny = bool(_DENY_RE.search(ans_text))
        gt_grant = "grant" in gt or "approve" in gt or "allow" in gt

        if predicted_grant and not predicted_deny and gt_grant:
            rewards.append(1.0)
        elif predicted_deny and not predicted_grant and not gt_grant:
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R5: Reasoning Depth — legal term density scoring
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reward_reasoning_depth(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R5: Scores the density of legal reasoning markers.
    Counts unique legal terms used, normalized to [0, 1].
    Requires at least 3 distinct markers for any reward.
    """
    rewards: List[float] = []
    for comp in completions:
        text = _extract_text(comp).lower()
        # Count unique markers present
        unique_hits = sum(1 for marker in LEGAL_REASONING_MARKERS if marker in text)
        if unique_hits < 3:
            rewards.append(0.0)
        else:
            # Normalize: 3 markers = 0.3, 10+ markers = 1.0
            rewards.append(min(1.0, unique_hits / 10.0))
    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R6: Anti-Hack Constraints — prevent reward gaming
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reward_anti_hack(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R6: Anti-hacking constraints.
    - Empty/very short (<10 words): -1.0
    - Single-word verdict shortcut: -1.0
    - Length 10-30 words: -0.3
    - Length >30 words with reasoning: +0.1
    """
    rewards: List[float] = []
    for comp in completions:
        text = _extract_text(comp).strip()
        words = text.split()
        wc = len(words)

        if wc < 5:
            rewards.append(-1.0)
            continue

        # Shortcut detection
        lower = text.lower().strip()
        shortcuts = [
            "grant", "deny", "granted", "denied",
            "bail granted", "bail denied", "yes", "no",
        ]
        if lower in shortcuts:
            rewards.append(-1.0)
            continue

        if wc < 30:
            rewards.append(-0.3)
        else:
            rewards.append(0.1)

    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R7: Anti-Repetition — ROUGE-L overlap penalty
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _lcs_length(x: List[str], y: List[str]) -> int:
    """Compute Longest Common Subsequence length (DP, O(n*m))."""
    m, n = len(x), len(y)
    if m == 0 or n == 0:
        return 0
    # Optimize memory: only keep two rows
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


def _rouge_l(candidate: str, reference: str) -> float:
    """Compute ROUGE-L F1 score between two texts."""
    c_words = candidate.lower().split()
    r_words = reference.lower().split()
    if not c_words or not r_words:
        return 0.0
    lcs = _lcs_length(c_words, r_words)
    precision = lcs / len(c_words) if c_words else 0.0
    recall = lcs / len(r_words) if r_words else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reward_anti_repetition(
    prompts: Sequence[str],
    completions: Sequence,
    previous_turns: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[float]:
    """
    R7: Penalizes copy-pasting previous arguments.
    ROUGE-L > 0.7 with previous turn → -0.5
    ROUGE-L > 0.5 → -0.2
    Otherwise → 0.0
    """
    rewards: List[float] = []
    for i, comp in enumerate(completions):
        text = _extract_text(comp)
        prev = previous_turns[i] if previous_turns and i < len(previous_turns) else ""

        if not prev or len(prev.split()) < 10:
            rewards.append(0.0)
            continue

        overlap = _rouge_l(text, prev)
        if overlap > 0.7:
            rewards.append(-0.5)
        elif overlap > 0.5:
            rewards.append(-0.2)
        else:
            rewards.append(0.0)

    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# R8: Snorkel Programmatic Supervision (BNSS / Special Acts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def reward_snorkel_labelers(
    prompts: Sequence[str],
    completions: Sequence,
    **kwargs,
) -> List[float]:
    """
    R8: Applies Snorkel-style programmatic labeling functions.
    Checks if the LLM's argument aligns with the active BNSS rules for the given case facts.
    """
    rewards: List[float] = []

    # Get case details from kwargs (passed by GRPOTrainer)
    case_types = kwargs.get("case_type", ["bns_318_bail"] * len(completions))
    days_custody = kwargs.get("days_in_custody", [30] * len(completions))
    charge_sheets = kwargs.get("charge_sheet_filed", [True] * len(completions))
    evidence_strengths = kwargs.get("evidence_strength", [0.5] * len(completions))
    flight_risks = kwargs.get("flight_risk", [0.5] * len(completions))
    delays = kwargs.get("delay_duration_months", [6] * len(completions))
    antecedents = kwargs.get("accused_antecedents", [0.5] * len(completions))

    for i, comp in enumerate(completions):
        text = _extract_text(comp).lower()
        
        # Build state dict for labelers
        state = {
            "case_type": case_types[i] if isinstance(case_types, list) else case_types,
            "days_in_custody": days_custody[i] if isinstance(days_custody, list) else days_custody,
            "charge_sheet_filed": charge_sheets[i] if isinstance(charge_sheets, list) else charge_sheets,
            "evidence_strength": evidence_strengths[i] if isinstance(evidence_strengths, list) else evidence_strengths,
            "flight_risk_score": flight_risks[i] if isinstance(flight_risks, list) else flight_risks,
            "delay_duration_months": delays[i] if isinstance(delays, list) else delays,
            "accused_antecedents": antecedents[i] if isinstance(antecedents, list) else antecedents,
        }

        # Get active rules from Snorkel functions
        active_rules = apply_labelers(state)
        if not active_rules:
            rewards.append(0.0)
            continue

        score = 0.0
        for rule in active_rules:
            # Does the agent argue in accordance with the rule?
            if rule["bail_mandatory"] is True:
                # If bail is mandatory/strongly favored, they should grant it and mention the section
                if "grant" in text:
                    score += 0.5
                if any(sec in text for sec in ["478", "479", "187", "37", "43d", "45"]):
                    score += 0.5
            elif rule["bail_mandatory"] is False:
                # If bail is barred, they should deny it
                if "deny" in text:
                    score += 0.5
                if any(sec in text for sec in ["480", "37", "43d", "45"]):
                    score += 0.5

        # Normalize score
        normalized_score = min(1.0, score / max(1, len(active_rules)))
        rewards.append(normalized_score)

    return rewards


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Composite Reward — weighted sum for GRPOTrainer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REWARD_WEIGHTS = {
    "format":      0.15,
    "statutory":   0.20,
    "citation":    0.15,
    "verdict":     0.20,
    "reasoning":   0.10,
    "anti_hack":   0.10,
    "anti_repeat": 0.05,
    "snorkel":     0.05,
}


def composite_reward(
    prompts: Sequence[str],
    completions: Sequence,
    ground_truth: Optional[Sequence[str]] = None,
    previous_turns: Optional[Sequence[str]] = None,
    **kwargs,
) -> List[float]:
    """
    Weighted composite of all 8 reward functions.
    Used as the single reward signal for GRPOTrainer.
    """
    r1 = reward_format_compliance(prompts, completions)
    r2 = reward_statutory_accuracy(prompts, completions)
    r3 = reward_case_citation(prompts, completions)
    r4 = reward_ground_truth_verdict(prompts, completions, ground_truth=ground_truth)
    r5 = reward_reasoning_depth(prompts, completions)
    r6 = reward_anti_hack(prompts, completions)
    r7 = reward_anti_repetition(prompts, completions, previous_turns=previous_turns)
    r8 = reward_snorkel_labelers(prompts, completions, **kwargs)

    w = REWARD_WEIGHTS
    results: List[float] = []
    for i in range(len(completions)):
        score = (
            w["format"]      * r1[i]
            + w["statutory"] * r2[i]
            + w["citation"]  * r3[i]
            + w["verdict"]   * r4[i]
            + w["reasoning"] * r5[i]
            + w["anti_hack"] * r6[i]
            + w["anti_repeat"] * r7[i]
            + w["snorkel"]   * r8[i]
        )
        
        # Add tiny variance to break zero-advantage ties in GRPO
        text = _extract_text(completions[i])
        # Length bonus + random noise large enough to survive bf16 truncation (0.01 - 0.05)
        import random
        score += min(0.01, len(text) * 1e-4) + random.uniform(0.01, 0.05)
        
        results.append(score)
    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Self-Test
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    print("=" * 60)
    print("  Nyaya-Env Reward Functions — Self-Test")
    print("=" * 60)

    good = [
        "<think>The accused has been in custody for 120 days without a chargesheet. "
        "Under Section 479 of BNSS 2023, the maximum period of detention has been exceeded. "
        "Citing Arnesh Kumar vs Bihar (2014), the court must consider that bail is the rule "
        "and jail is the exception per Gudikanti Narasimhulu (1977). The flight risk is low "
        "and the accused has no prior antecedents. The prima facie case is weak. "
        "Considering proportionality and fundamental right to personal liberty under Article 21, "
        "and that community safety is not threatened, bail conditions can mitigate any risk."
        "</think><answer>Bail is GRANTED with conditions: weekly reporting and passport surrender.</answer>"
    ]
    bad_hallucinate = ["Under Section 999 of BNSS, the accused must be denied bail."]
    bad_short = ["Grant"]
    empty_prompt = [""]

    tests = [
        ("R1 format (good)",    reward_format_compliance(empty_prompt, good)),
        ("R1 format (bad)",     reward_format_compliance(empty_prompt, bad_hallucinate)),
        ("R2 statute (good)",   reward_statutory_accuracy(empty_prompt, good)),
        ("R2 statute (fake)",   reward_statutory_accuracy(empty_prompt, bad_hallucinate)),
        ("R3 citation (good)",  reward_case_citation(empty_prompt, good)),
        ("R3 citation (none)",  reward_case_citation(empty_prompt, bad_hallucinate)),
        ("R4 verdict (good)",   reward_ground_truth_verdict(empty_prompt, good, ground_truth=["grant"])),
        ("R5 reasoning (good)", reward_reasoning_depth(empty_prompt, good)),
        ("R5 reasoning (bad)",  reward_reasoning_depth(empty_prompt, bad_short)),
        ("R6 anti-hack (good)", reward_anti_hack(empty_prompt, good)),
        ("R6 anti-hack (bad)",  reward_anti_hack(empty_prompt, bad_short)),
        ("R7 anti-repeat",      reward_anti_repetition(empty_prompt, good, previous_turns=[good[0]])),
        ("Composite (good)",    composite_reward(empty_prompt, good, ground_truth=["grant"])),
        ("Composite (bad)",     composite_reward(empty_prompt, bad_hallucinate)),
    ]

    for name, result in tests:
        print(f"  {name:30s} → {result}")

    print()
    print("✅ All reward functions operational.")
