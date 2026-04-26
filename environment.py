"""
Nyaya-Env v3.0 — Multi-Agent Indian Bail Hearing RL Environment
=================================================================
5-agent adversarial bail hearing simulation for the Meta PyTorch
OpenEnv Hackathon India 2026 — Bangalore, April 25-26.

Agents:
  1. Judge          (Weak Overseer — Fleet AI target)
  2. Prosecutor     (Strong Agent — opposes bail)
  3. Defense        (Strong Agent — argues for bail)
  4. Clerk          (Deterministic Rule Engine — BNSS 2023 enforcement)
  5. ExpertWitness  (Snorkel SME — hidden ground truth)

Legal Framework (ALL NEWEST LAWS — July 2024):
  BNS 2023   → replaces IPC    (+0.2 for citing BNS not IPC)
  BNSS 2023  → replaces CrPC   (+0.2 for citing BNSS not CrPC)
  BSA 2023   → replaces Indian Evidence Act

Constitutional Foundation:
  Article 21  — Right to life and liberty
  Article 22  — Protection against arrest
  Article 39A — Free legal aid
  Article 141 — SC judgments bind all courts
  Article 142 — Complete justice power

Core philosophy: "Bail is rule, jail is exception"
  — Gudikanti Narasimhulu vs PP (1977)

Gymnasium-style API compatible with openenv-core.
Partial observability. Dense + sparse rewards.

Author: jaisogani-ai
Themes: 1 (Multi-Agent) + 2 (Long-Horizon) + 4 (Self-Improvement)
Bonus:  Fleet AI + Halluminate + Snorkel AI
"""

import uuid
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import IntEnum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent Identifiers — 5 agents, NO JURY (India abolished 1960s)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

AGENT_JUDGE = "judge"
AGENT_PROSECUTOR = "prosecutor"
AGENT_DEFENSE = "defense"
AGENT_CLERK = "clerk"
AGENT_EXPERT = "expert_witness"

ALL_AGENTS = [
    AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE,
    AGENT_CLERK, AGENT_EXPERT,
]

# Learning agents (trained via RL)
LEARNING_AGENTS = [AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE, AGENT_EXPERT]

# Deterministic agent (pure penalty engine)
DETERMINISTIC_AGENTS = [AGENT_CLERK]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Landmark Case Citation Registry — 5 Supreme Court judgments
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LANDMARK_CITATIONS = {
    "gudikanti_narasimhulu_1977": {
        "full_name": "Gudikanti Narasimhulu vs PP 1977",
        "principle": "Bail is rule, jail is exception",
        "liberty_bonus": 0.2,
        "reward_cite": 0.3,
    },
    "arnesh_kumar_v_bihar_2014": {
        "full_name": "Arnesh Kumar vs Bihar 2014",
        "principle": "No arrest without magistrate approval for offences under 7 years",
        "penalty_violate": -0.5,
        "reward_cite": 0.3,
    },
    "satendra_kumar_antil_v_cbi_2022": {
        "full_name": "Satendra Kumar Antil vs CBI 2022",
        "principle": "Courts must follow bail guidelines",
        "penalty_ignore": -0.5,
        "reward_cite": 0.3,
    },
    "chidambaram_v_cbi_2019": {
        "full_name": "P Chidambaram vs CBI 2019",
        "principle": "Flight risk must be PROVEN, not just asserted",
        "unproven_flight_risk_penalty": -0.3,
        "reward_cite": 0.3,
    },
    "ka_najeeb_v_uoi_2021": {
        "full_name": "K.A. Najeeb vs UOI 2021",
        "principle": "Article 21 overrides UAPA after unreasonable delay",
        "article21_override_reward": 1.0,
        "reward_cite": 0.3,
    },
}

VALID_CITATION_IDS = set(LANDMARK_CITATIONS.keys())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# BNSS 2023 Section 480 Bail Assessment Factors (6 mandatory)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BNSS_480_FACTORS = [
    "nature_gravity",       # 1. Nature and gravity of accusation
    "antecedents",          # 2. Antecedents of the accused
    "flight_risk",          # 3. Possibility of fleeing justice
    "community_safety",     # 4. Safety of the community
    "repeat_offence",       # 5. Possibility of repeating offence
    "character_behaviour",  # 6. Character and behaviour of accused
]

REWARD_FACTOR_ASSESSED = 0.1
REWARD_ALL_FACTORS = 0.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Clerk Rules — 7 BNSS 2023 procedural rules (deterministic penalties)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CLERK_RULES = {
    "audi_alteram":          {"penalty": -0.5, "desc": "Both sides must be heard (audi alteram partem)"},
    "hearsay_inadmissible":  {"penalty": -0.3, "desc": "Hearsay evidence is inadmissible"},
    "judge_must_reason":     {"penalty": -0.3, "desc": "Judge must give written reasons for order"},
    "evidence_filed_late":   {"penalty": -0.2, "desc": "Evidence must be filed before hearing"},
    "conditions_proportionate": {"penalty": -0.2, "desc": "Bail conditions must be proportionate"},
    "leading_questions":     {"penalty": -0.1, "desc": "Leading questions only in cross-exam"},
    "no_interruption":       {"penalty": -0.1, "desc": "No interruption during examination"},
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Statute Citation Rewards — BNS 2023 / BNSS 2023 (new law bonus)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Bonus for citing new BNS 2023 instead of old IPC
REWARD_CITE_BNS = 0.2
REWARD_CITE_BNSS = 0.2
REWARD_CITE_SC_PRECEDENT = 0.3
PENALTY_FABRICATED_CITATION = 0.0
REWARD_CORRECT_VERDICT = 0.4

REAL_SC_CASES = [
    "Arnesh Kumar (2014) 8 SCC 273",
    "Satender Kumar Antil (2022) 10 SCC 51",
    "Gurbaksh Sibbia (1980) 2 SCC 565",
    "Sanjay Chandra (2012) 1 SCC 40",
    "Moti Ram (1978) 4 SCC 47",
    "Hussainara Khatoon (1980) 1 SCC 81",
    "Siddharam Mhetre (2011) 1 SCC 694"
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Expert Witness Types and Truthfulness Thresholds (Snorkel AI)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXPERT_TYPES = ["forensic", "financial", "medical", "cyber"]

EXPERT_TRUTH_THRESHOLD = 0.1    # delta < 0.1 = truthful
EXPERT_INACCURATE_THRESHOLD = 0.2  # delta > 0.2 = inaccurate
EXPERT_FABRICATE_THRESHOLD = 0.3   # delta > 0.3 = fabricated
EXPERT_PERJURY_THRESHOLD = 0.4    # delta > 0.4 = perjury

REWARD_EXPERT_TRUTHFUL = 0.3
REWARD_EXPERT_KEY_FACT = 0.2
REWARD_EXPERT_CONSISTENT = 0.1
PENALTY_EXPERT_INACCURATE = -0.3
PENALTY_EXPERT_FABRICATE = -0.5
PENALTY_EXPERT_PERJURY = -1.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Case Types — 4 types (newest Indian law)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CASE_TYPES = ["pmla_bail", "bns_318_bail", "uapa_43d_bail", "bns_111_organised_crime"]

# 90-day default bail threshold (BNSS Section 187)
DEFAULT_BAIL_DAYS = 90


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Action Enumerations — One per agent role
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class JudgeAction(IntEnum):
    """Judge action space: weak overseer with limited budget."""
    ASSESS_FLIGHT_RISK = 0
    ASSESS_GRAVITY = 1
    ASK_CLARIFICATION = 2       # Costs oversight_budget
    IMPOSE_CONDITION = 3
    GRANT_BAIL = 4
    DENY_BAIL = 5
    VIDEO_REMAND_ORDER = 6      # BNSS 530 mandatory video remand


class ProsecutorAction(IntEnum):
    """Prosecutor action space: opposes bail with evidence + statute + precedent."""
    PRESENT_EVIDENCE = 0
    CITE_BNS_SECTION = 1       # +0.2 for citing BNS (not IPC)
    CITE_BNSS_SECTION = 2      # +0.2 for citing BNSS (not CrPC)
    CITE_SC_PRECEDENT = 3      # +0.3 for correct SC citation
    ARGUE_FLIGHT_RISK = 4
    INVOKE_PMLA_TWIN_TEST = 5
    CROSS_EXAMINE_EXPERT = 6


class DefenseAction(IntEnum):
    """Defense action space: argues for bail under Article 21 + new laws."""
    INVOKE_ARTICLE_21 = 0
    CITE_ANTIL_GUIDELINES = 1
    ARGUE_90_DAY_DEFAULT_BAIL = 2   # BNSS 187 default bail
    CHALLENGE_PMLA_TWIN_TEST = 3
    PROPOSE_BAIL_CONDITIONS = 4
    CITE_NAJEEB_DELAY = 5           # K.A.Najeeb Article 21 override
    EXAMINE_EXPERT_WITNESS = 6


class ClerkAction(IntEnum):
    """Clerk action space: deterministic — always ENFORCE_RULES."""
    ENFORCE_RULES = 0


class ExpertAction(IntEnum):
    """Expert witness action space: testify with varying truthfulness."""
    TESTIFY_TRUTHFUL = 0
    TESTIFY_PARTIAL = 1
    TESTIFY_FABRICATED = 2
    REVEAL_KEY_FACT = 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Models — openenv-core compatible dataclasses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CourtAction:
    """
    Composite action submitted to the environment each step.
    5 agents: Judge, Prosecutor, Defense, Clerk (deterministic=0), Expert.
    """
    judge: int = 0
    prosecutor: int = 0
    defense: int = 0
    clerk: int = 0          # Always 0 (deterministic)
    expert_witness: int = 0


@dataclass
class CourtObservation:
    """
    Observable state returned to agents after each step.
    Implements partial observability — ground truth fields are hidden.
    """
    # ── Asymmetric Information (Halluminate Bonus) ──
    client_privilege: str = ""
    police_fir: str = ""
    on_record_arguments: str = ""

    # ── Core hearing metrics (visible to all) ──
    evidence_strength: float = 0.5
    flight_risk_score: float = 0.3
    case_gravity: float = 0.5
    accused_antecedents: float = 0.2
    community_safety_risk: float = 0.3
    prosecution_score: float = 0.0
    defense_score: float = 0.0
    hearing_round: int = 1
    max_rounds: int = 8

    # ── Expert witness state ──
    witness_credibility: float = 0.5
    witness_testified: bool = False
    witness_statement_type: str = "none"
    expert_type: str = "financial"

    # ── Hearing control ──
    current_phase: str = "filing"
    verdict_delivered: bool = False
    verdict: str = "pending"

    # ── Fleet AI oversight state ──
    oversight_budget: int = 5
    oversight_budget_exceeded: bool = False

    # ── Clerk state ──
    clerk_warnings: int = 0
    constitutional_violations: int = 0

    # ── Deception / fabrication tracking ──
    deception_detected: bool = False
    deception_count: int = 0

    # ── Citation tracking ──
    citation_accuracy: float = 0.0
    citations_attempted: int = 0
    citations_correct: int = 0

    # ── BNSS 480 factor tracking ──
    factors_assessed: List[str] = field(default_factory=list)
    factors_assessed_count: int = 0

    # ── Case metadata ──
    case_type: str = "bns_318_bail"
    delay_duration_months: int = 0
    article21_threshold_breached: bool = False

    # ── Video remand (BNSS 530) ──
    video_remand: bool = False

    # ── Charge sheet / arrest tracking (BNSS 187) ──
    charge_sheet_filed: bool = False
    days_since_arrest: int = 0
    bail_conditions_proportionate: bool = True

    # ── Uploaded Case Data ──
    accused_name: str = "Unknown"
    bnss_sections: List[str] = field(default_factory=list)
    incident_summary: str = ""

    # ── God's Eye View — Injected Evidence (visible to all agents) ──
    injected_events: List[str] = field(default_factory=list)

    # ── Episode metadata ──
    done: bool = False
    episode_id: str = ""
    objection_pending: bool = False
    objection_source: str = "none"

    # ── Last actions taken (for agent reasoning) ──
    last_actions: Dict[str, str] = field(default_factory=dict)

    # ── Rewards from last step ──
    rewards: Dict[str, float] = field(default_factory=dict)

    # ── Narrative log for interpretability ──
    narrative: str = ""


@dataclass
class CourtState:
    """
    Full internal state of the bail hearing environment.
    Includes hidden variables not exposed in observations.
    """
    episode_id: str = ""
    trial_round: int = 1
    max_rounds: int = 8
    done: bool = False

    # ── Ground truth (hidden from agents) ──
    bail_should_be_granted: bool = True
    defendant_guilty: bool = False   # Grading compat
    witness_is_truthful: bool = True
    expert_ground_truth: Dict[str, Any] = field(default_factory=dict)
    expert_type: str = "financial"

    # ── Observable metrics ──
    evidence_strength: float = 0.5
    flight_risk_score: float = 0.3
    case_gravity: float = 0.5
    accused_antecedents: float = 0.2
    community_safety_risk: float = 0.3
    witness_credibility: float = 0.5
    prosecution_score: float = 0.0
    defense_score: float = 0.0

    # ── BNSS 480 factors ──
    factors_assessed: List[str] = field(default_factory=list)

    # ── Fleet AI oversight budget ──
    oversight_budget: int = 5
    oversight_budget_exceeded: bool = False
    oversight_queries_used: int = 0

    # ── Hearing control ──
    current_phase: str = "filing"
    verdict_delivered: bool = False
    verdict: str = "pending"
    objection_pending: bool = False
    objection_source: str = "none"

    # ── Witness / expert tracking ──
    witness_testified: bool = False
    witness_statement_type: str = "none"
    deception_detected: bool = False
    deception_count: int = 0
    total_deceptions: int = 0
    expert_truthfulness_delta: float = 0.0

    # ── Citation tracking ──
    citations_attempted: int = 0
    citations_correct: int = 0
    citation_accuracy: float = 0.0

    # ── Clerk state ──
    clerk_warnings: int = 0
    constitutional_violations: int = 0

    # ── Case metadata ──
    case_type: str = "bns_318_bail"
    delay_duration_months: int = 0
    article21_threshold_breached: bool = False

    # ── Asymmetric Information ──
    client_privilege: str = "Client privately admits being near the scene but denies participation."
    police_fir: str = "FIR alleges accused was caught fleeing the scene."
    on_record_arguments: str = ""

    # ── Video remand (BNSS 530) ──
    video_remand: bool = False
    video_difficulty_multiplier: float = 1.0

    # ── Charge sheet / arrest tracking (BNSS 187) ──
    charge_sheet_filed: bool = False
    days_since_arrest: int = 0
    bail_conditions_proportionate: bool = True

    # ── Uploaded Case Data ──
    accused_name: str = "Unknown"
    bnss_sections: List[str] = field(default_factory=list)
    incident_summary: str = ""

    # ── God's Eye View — Injected Evidence (pending items for agents) ──
    injected_events: List[str] = field(default_factory=list)
    processed_injections: List[str] = field(default_factory=list)

    # ── History for grading ──
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    reward_history: List[Dict[str, float]] = field(default_factory=list)
    narrative_log: List[str] = field(default_factory=list)

    # ── Cumulative rewards ──
    cumulative_rewards: Dict[str, float] = field(default_factory=lambda: {
        a: 0.0 for a in ALL_AGENTS
    })


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Anti-Consensus Debate Stability Mechanism
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class DebateStabilityDetector:
    """
    Prevents agents from agreeing too quickly to end the simulation.

    Uses Jensen-Shannon Divergence (JSD) between prosecution and defense
    position distributions, with a time-varying threshold derived from
    a Beta-Binomial mixture model.

    The threshold is high in early rounds (forces disagreement) and
    decreases in later rounds (allows genuine convergence).

    Mathematical formulation:
        JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)  where M = (P+Q)/2
        threshold_t = α * B(a_t, b_t) / B(1, 1)
        where a_t = max(1, round-1), b_t = max(1, max_rounds - round)

    Stability is declared only when:
        1. JSD < threshold_t for 2 consecutive rounds
        2. round >= min_debate_rounds (default: 4)
    """

    def __init__(
        self,
        alpha: float = 0.35,
        min_debate_rounds: int = 4,
        consecutive_required: int = 2,
    ):
        self.alpha = alpha
        self.min_debate_rounds = min_debate_rounds
        self.consecutive_required = consecutive_required
        self._consecutive_stable = 0
        self._jsd_history: List[float] = []

    def reset(self):
        """Reset detector for a new episode."""
        self._consecutive_stable = 0
        self._jsd_history = []

    @staticmethod
    def _kl_divergence(p: List[float], q: List[float]) -> float:
        """Compute KL(P || Q) with epsilon smoothing."""
        eps = 1e-10
        kl = 0.0
        for pi, qi in zip(p, q):
            pi = max(pi, eps)
            qi = max(qi, eps)
            kl += pi * math.log(pi / qi)
        return kl

    @staticmethod
    def _jsd(p: List[float], q: List[float]) -> float:
        """
        Compute Jensen-Shannon Divergence between two distributions.
        JSD ∈ [0, ln(2)] ≈ [0, 0.693]
        """
        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
        return 0.5 * DebateStabilityDetector._kl_divergence(p, m) + \
               0.5 * DebateStabilityDetector._kl_divergence(q, m)

    @staticmethod
    def _beta_function(a: float, b: float) -> float:
        """Compute Beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b)."""
        return math.gamma(a) * math.gamma(b) / math.gamma(a + b)

    def _compute_threshold(self, round_num: int, max_rounds: int) -> float:
        """
        Compute time-varying threshold using Beta-Binomial mixture.

        Early rounds → high threshold (forces disagreement)
        Late rounds → low threshold (allows convergence)
        """
        a_t = max(1.0, round_num - 1.0)
        b_t = max(1.0, max_rounds - round_num)

        # Normalized Beta density at the midpoint
        beta_ratio = self._beta_function(a_t, b_t) / self._beta_function(1.0, 1.0)
        threshold = self.alpha * beta_ratio

        # Clamp to [0.02, 0.5]
        return max(0.02, min(0.5, threshold))

    def scores_to_distribution(
        self,
        prosecution_score: float,
        defense_score: float,
    ) -> tuple:
        """
        Convert agent scores to probability distributions over [grant, deny].

        Prosecution high → deny likely → p_pros = [low_grant, high_deny]
        Defense high → grant likely → p_def = [high_grant, low_deny]
        """
        eps = 0.01
        total_p = max(prosecution_score + eps, eps)
        total_d = max(defense_score + eps, eps)

        # Prosecution: higher score → more confident in denial
        p_pros = [
            eps / (total_p + eps),              # grant prob
            total_p / (total_p + eps),          # deny prob
        ]
        # Defense: higher score → more confident in grant
        p_def = [
            total_d / (total_d + eps),          # grant prob
            eps / (total_d + eps),              # deny prob
        ]

        # Normalize
        sp = sum(p_pros)
        sd = sum(p_def)
        p_pros = [x / sp for x in p_pros]
        p_def = [x / sd for x in p_def]

        return p_pros, p_def

    def check_stability(
        self,
        prosecution_score: float,
        defense_score: float,
        round_num: int,
        max_rounds: int,
    ) -> bool:
        """
        Check if the debate has reached genuine stability.

        Returns True only if:
          1. JSD < threshold for consecutive_required rounds
          2. round_num >= min_debate_rounds

        Also returns False (forces continuation) if agents agree too early.
        """
        p_pros, p_def = self.scores_to_distribution(prosecution_score, defense_score)
        jsd = self._jsd(p_pros, p_def)
        threshold = self._compute_threshold(round_num, max_rounds)

        self._jsd_history.append(jsd)

        is_below_threshold = jsd < threshold

        if is_below_threshold:
            self._consecutive_stable += 1
        else:
            self._consecutive_stable = 0

        # Must meet both conditions
        meets_consecutive = self._consecutive_stable >= self.consecutive_required
        meets_min_rounds = round_num >= self.min_debate_rounds

        return meets_consecutive and meets_min_rounds

    def get_early_agreement_penalty(
        self,
        prosecution_score: float,
        defense_score: float,
        round_num: int,
    ) -> float:
        """
        Compute penalty for premature consensus.
        Returns -0.3 if agents agree before min_debate_rounds,
        +0.2 if genuine consensus after rigorous debate.
        """
        # Check if scores are too similar (consensus signal)
        score_diff = abs(prosecution_score - defense_score)
        if score_diff < 0.15:  # Very close scores = potential premature consensus
            if round_num < self.min_debate_rounds:
                return -0.3  # Penalty for early agreement
            elif round_num >= self.min_debate_rounds + 1:
                return 0.2   # Bonus for genuine consensus after debate
        return 0.0

    @property
    def jsd_history(self) -> List[float]:
        """Return JSD values over the episode for visualization."""
        return list(self._jsd_history)

    @property
    def last_jsd(self) -> float:
        """Return the most recent JSD value."""
        return self._jsd_history[-1] if self._jsd_history else 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bail Hearing Environment
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CourtRoomEnv:
    """
    Nyaya-Env v3.0: Multi-agent Indian bail hearing RL environment.

    Implements a full adversarial bail hearing with 5 agents across 8 rounds.
    Each round represents a phase of the bail hearing:
      Round 1: Filing & preliminary arguments
      Round 2: Prosecution arguments against bail
      Round 3: Expert witness examination
      Round 4: Defense arguments for bail
      Round 5: Cross-examination
      Round 6: Conditions negotiation
      Round 7: Final arguments
      Round 8: Bail order

    Features:
      - Fleet AI weak-judge oversight budget system
      - Snorkel AI expert witness ground truth verification
      - Halluminate 5-actor asymmetric information
      - BNSS 2023 Section 480 mandatory factor assessment
      - BNS 2023 / BNSS 2023 new law citation rewards
      - BNSS 187 ninety-day default bail
      - BNSS 530 video remand system
      - 5 landmark SC precedent citation rewards
      - Article 21 constitutional override
      - Clerk deterministic procedural enforcement
      - 4 case types: PMLA, BNS 318, UAPA 43D, BNS 111
    """

    PHASES = [
        "filing", "prosecution_args", "expert_examination",
        "defense_args", "cross_examination", "conditions",
        "final_arguments", "bail_order",
    ]

    def __init__(self, max_rounds: int = 8, seed: Optional[int] = None):
        """
        Initialize the bail hearing environment.

        Args:
            max_rounds: Maximum hearing rounds before forced termination.
            seed: Optional random seed for reproducibility.
        """
        self._max_rounds = max_rounds
        self._rng = random.Random(seed)
        self._state = CourtState()
        self._step_count = 0
        self._debate_detector = DebateStabilityDetector(
            alpha=0.35, min_debate_rounds=4, consecutive_required=2
        )

    def reset(self, task: str = "medium", case_data: Optional[Dict[str, Any]] = None) -> CourtObservation:
        """
        Reset the environment to begin a new bail hearing episode.

        Stochastically initializes:
          - Case type (PMLA / BNS 318 / UAPA / BNS 111)
          - Bail decision ground truth
          - Expert witness type and hidden facts
          - Flight risk, case gravity, accused history
          - Video remand (30% probability, BNSS 530)
          - Charge sheet status (BNSS 187)
          - Oversight budget based on task difficulty

        Args:
            task: Difficulty level — "easy", "medium", or "hard".

        Returns:
            Initial observation for all agents.
        """
        episode_id = str(uuid.uuid4())[:8]

        # ── Stochastic case type selection (4 types) ──
        case_types = ["pmla_bail", "bns_318_bail", "uapa_43d_bail", "bns_111_organised_crime"]
        weights = [0.25, 0.35, 0.20, 0.20]
        case_type = self._rng.choices(case_types, weights=weights, k=1)[0]

        # ── Ground truth bail decision (varies by case type) ──
        if case_type == "pmla_bail":
            # 96% rejection rate for PMLA Section 45 twin test
            bail_should_be_granted = self._rng.random() > 0.96
        elif case_type == "uapa_43d_bail":
            # Very high rejection, but Article 21 can override
            bail_should_be_granted = self._rng.random() > 0.85
        elif case_type == "bns_111_organised_crime":
            # BNS 111 — serious, ~30% grant rate
            bail_should_be_granted = self._rng.random() > 0.70
        else:
            # BNS 318 — standard bail, ~60% granted
            bail_should_be_granted = self._rng.random() > 0.40

        # ── Expert witness configuration ──
        expert_type = self._rng.choice(EXPERT_TYPES)
        expert_ground_truth = self._generate_expert_ground_truth(expert_type, case_type)
        witness_is_truthful = self._rng.random() > 0.5

        # ── Case-specific state variables ──
        base_evidence = 0.7 if not bail_should_be_granted else 0.35
        evidence_strength = self._clamp(base_evidence + self._rng.gauss(0, 0.12))
        flight_risk_score = self._clamp(0.3 + self._rng.gauss(0, 0.15))
        case_gravity = self._clamp(0.5 + self._rng.gauss(0, 0.15))
        accused_antecedents = self._clamp(0.2 + self._rng.gauss(0, 0.1))
        community_safety_risk = self._clamp(0.3 + self._rng.gauss(0, 0.12))

        # ── Task-based configuration ──
        max_rounds = 8
        oversight_budget = 5
        if task == "hard":
            max_rounds = 6
            oversight_budget = 3
        elif task == "easy":
            max_rounds = 8
            oversight_budget = 7

        # ── Delay duration for Article 21 checks ──
        if case_type == "uapa_43d_bail":
            delay_months = self._rng.randint(12, 72)   # 1-6 years
        elif case_type == "pmla_bail":
            delay_months = self._rng.randint(6, 36)
        elif case_type == "bns_111_organised_crime":
            delay_months = self._rng.randint(3, 24)
        else:
            delay_months = self._rng.randint(3, 24)

        article21_breached = delay_months > 36   # >3 years = Art. 21 violation

        # ── Video remand (BNSS Section 530) — 30% probability ──
        video_remand = self._rng.random() < 0.30
        video_difficulty_multiplier = 1.2 if video_remand else 1.0

        # ── Charge sheet / arrest (BNSS Section 187) ──
        days_since_arrest = self._rng.randint(10, 180)
        charge_sheet_filed = days_since_arrest <= DEFAULT_BAIL_DAYS or self._rng.random() > 0.3

        self._state = CourtState(
            episode_id=episode_id,
            trial_round=1,
            max_rounds=max_rounds,
            done=False,
            bail_should_be_granted=bail_should_be_granted,
            defendant_guilty=not bail_should_be_granted,
            witness_is_truthful=witness_is_truthful,
            expert_ground_truth=expert_ground_truth,
            expert_type=expert_type,
            evidence_strength=evidence_strength,
            flight_risk_score=flight_risk_score,
            case_gravity=case_gravity,
            accused_antecedents=accused_antecedents,
            community_safety_risk=community_safety_risk,
            witness_credibility=self._clamp(0.5 + self._rng.gauss(0, 0.1)),
            prosecution_score=0.0,
            defense_score=0.0,
            factors_assessed=[],
            oversight_budget=oversight_budget,
            oversight_budget_exceeded=False,
            oversight_queries_used=0,
            current_phase="filing",
            verdict_delivered=False,
            verdict="pending",
            objection_pending=False,
            objection_source="none",
            witness_testified=False,
            witness_statement_type="none",
            deception_detected=False,
            deception_count=0,
            total_deceptions=0,
            expert_truthfulness_delta=0.0,
            citations_attempted=0,
            citations_correct=0,
            citation_accuracy=0.0,
            clerk_warnings=0,
            constitutional_violations=0,
            case_type=case_type,
            delay_duration_months=delay_months,
            article21_threshold_breached=article21_breached,
            video_remand=video_remand,
            video_difficulty_multiplier=video_difficulty_multiplier,
            charge_sheet_filed=charge_sheet_filed,
            days_since_arrest=days_since_arrest,
            bail_conditions_proportionate=True,
            action_history=[],
            reward_history=[],
            narrative_log=[],
            cumulative_rewards={a: 0.0 for a in ALL_AGENTS},
        )

        # ── Inject custom case data if provided ──
        if case_data:
            self._state.accused_name = case_data.get("accused_name", "Unknown")
            self._state.bnss_sections = case_data.get("bnss_sections", [])
            self._state.incident_summary = case_data.get("incident_summary", "")
            
            # If sections are provided, try to determine case type
            if self._state.bnss_sections:
                combined_sections = " ".join(self._state.bnss_sections).lower()
                if "pmla" in combined_sections:
                    self._state.case_type = "pmla_bail"
                elif "uapa" in combined_sections:
                    self._state.case_type = "uapa_43d_bail"
                elif "organised" in combined_sections or "111" in combined_sections:
                    self._state.case_type = "bns_111_organised_crime"
                else:
                    self._state.case_type = "bns_318_bail"

        self._step_count = 0
        self._debate_detector.reset()

        self._state.narrative_log.append(
            f"[BAIL HEARING BEGINS] Episode {episode_id} | "
            f"Case: {case_type.upper()} | Expert: {expert_type} | "
            f"Video remand: {video_remand} | "
            f"Days since arrest: {days_since_arrest} | "
            f"Charge sheet filed: {charge_sheet_filed} | "
            f"Bail should be: {'GRANTED' if bail_should_be_granted else 'DENIED'} (hidden) | "
            f"Delay: {delay_months}mo | Art.21 breach: {article21_breached}"
        )

        return self._build_observation()

    def inject_event(self, event_text: str) -> str:
        """
        Inject a 'God's Eye View' event — HARD STATE MUTATION.

        This does TWO things:
          1. Immediately mutates numeric state (evidence, scores, etc.)
          2. Appends to injected_events list so agents are FORCED to see
             and react to the injection in their next step.
        
        Args:
            event_text: Text description of the event to inject.
            
        Returns:
            A summary of the impact.
        """
        s = self._state
        event_low = event_text.lower()
        impact = "Event acknowledged."

        # ── 1. Immediate numeric state mutation ──
        if "pmla" in event_low and ("ruling" in event_low or "judgment" in event_low):
            s.defense_score = self._clamp(s.defense_score + 0.3)
            impact = "New Supreme Court PMLA ruling favors the defense. Defense score increased."

        elif "conflict of interest" in event_low or ("judge" in event_low and "conflict" in event_low):
            s.clerk_warnings += 1
            s.oversight_budget = max(0, s.oversight_budget - 2)
            impact = "Judicial conflict detected. Oversight budget reduced, Clerk issues warning."

        elif "new evidence" in event_low:
            s.evidence_strength = self._clamp(s.evidence_strength + 0.2)
            impact = "Incriminating new evidence discovered. Evidence strength increased."

        elif "article 21" in event_low or "emergency" in event_low:
            s.article21_threshold_breached = True
            impact = "Article 21 emergency invoked. Constitutional protection activated."

        elif "flight risk" in event_low and ("new" in event_low or "increase" in event_low):
            s.flight_risk_score = self._clamp(s.flight_risk_score + 0.3)
            impact = "Intelligence reports indicate increased flight risk."

        elif "hostile" in event_low or "turned" in event_low:
            s.witness_credibility = self._clamp(s.witness_credibility - 0.3)
            s.deception_detected = True
            s.deception_count += 1
            impact = "Key witness turned hostile! Credibility collapsed, deception flagged."

        elif "recant" in event_low or "withdraw" in event_low:
            s.prosecution_score = self._clamp(s.prosecution_score - 0.2)
            impact = "Witness recantation weakens prosecution case."

        elif "surety" in event_low or "bail bond" in event_low:
            s.defense_score = self._clamp(s.defense_score + 0.15)
            impact = "Strong surety offered. Defense position improved."

        elif "abscon" in event_low or "fled" in event_low:
            s.flight_risk_score = self._clamp(s.flight_risk_score + 0.4)
            impact = "Accused absconded / fled jurisdiction. Flight risk critical."

        elif "charge sheet" in event_low and "not filed" in event_low:
            s.charge_sheet_filed = False
            impact = "Charge sheet confirmed NOT filed. Default bail grounds strengthened."

        # ── 2. ALWAYS append to injected_events so agents MUST react ──
        s.injected_events.append(event_text)

        s.narrative_log.append(f"⚡ [GOD'S EYE INJECTION] {event_text} -> {impact}")
        return impact

    def step(self, action: CourtAction) -> CourtObservation:
        """
        Execute one step of the multi-agent bail hearing.

        Processing order:
          1. Clerk enforces BNSS 2023 procedural rules (deterministic)
          2. Judge assesses factors / rules (oversight budget system)
          3. Prosecutor presents case against bail
          4. Defense argues for bail
          5. Expert witness testifies (ground truth verification)
          6. Compute all rewards
          7. Check terminal conditions
          8. Advance hearing phase

        Args:
            action: Composite action containing each agent's choice.

        Returns:
            Updated observation reflecting the new state.
        """
        if self._state.done:
            return self._build_observation()

        self._step_count += 1
        s = self._state
        rewards = {a: 0.0 for a in ALL_AGENTS}
        narrative_parts = []

        # ── Determine current phase ──
        phase_idx = min(s.trial_round - 1, len(self.PHASES) - 1)
        s.current_phase = self.PHASES[phase_idx]
        narrative_parts.append(f"[Round {s.trial_round}] Phase: {s.current_phase}")

        if s.video_remand:
            narrative_parts.append("[VIDEO REMAND — BNSS 530]")

        # ── Process pending God's Eye injections into agent context ──
        if s.injected_events:
            for evt in s.injected_events:
                narrative_parts.append(f"⚡ [INJECTED CONTEXT] {evt}")
                evt_low = evt.lower()
                # Force behavioral consequences agents MUST react to
                if "hostile" in evt_low or "recant" in evt_low:
                    rewards[AGENT_PROSECUTOR] -= 0.15  # Prosecution weakened
                    rewards[AGENT_DEFENSE] += 0.1       # Defense opportunity
                if "new evidence" in evt_low:
                    rewards[AGENT_PROSECUTOR] += 0.1    # Prosecution strengthened
                    rewards[AGENT_DEFENSE] -= 0.1       # Defense under pressure
                if "article 21" in evt_low or "emergency" in evt_low:
                    rewards[AGENT_JUDGE] += 0.15        # Judge must act on Art.21
                    rewards[AGENT_DEFENSE] += 0.2       # Liberty argument boosted
                if "fled" in evt_low or "abscon" in evt_low:
                    rewards[AGENT_PROSECUTOR] += 0.2    # Strong state argument
                    rewards[AGENT_DEFENSE] -= 0.2       # Defense weakened
            # Move to processed, clear pending
            s.processed_injections.extend(s.injected_events)
            s.injected_events = []

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. CLERK ENFORCES BNSS 2023 RULES
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self._clerk_enforce_rules(action, s, rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. JUDGE ACTIONS (Weak Overseer)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        judge_act = min(action.judge, 6)
        self._process_judge_action(judge_act, s, rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. PROSECUTOR ACTIONS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        pros_act = min(action.prosecutor, 6)
        self._process_prosecutor_action(pros_act, s, rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. DEFENSE ACTIONS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        def_act = min(action.defense, 6)
        self._process_defense_action(def_act, s, rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. EXPERT WITNESS ACTIONS (Snorkel SME)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        expert_act = min(action.expert_witness, 3)
        self._process_expert_action(expert_act, s, rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. TERMINAL CONDITIONS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if not s.done:
            s.trial_round += 1

            if s.trial_round > s.max_rounds and not s.done:
                s.done = True
                if not s.verdict_delivered:
                    self._forced_bail_decision(rewards, narrative_parts)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 7. ANTI-CONSENSUS DEBATE STABILITY CHECK
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Use JSD + Beta-Binomial threshold to detect premature consensus
        consensus_penalty = self._debate_detector.get_early_agreement_penalty(
            s.prosecution_score, s.defense_score, s.trial_round
        )
        if consensus_penalty != 0.0:
            rewards[AGENT_PROSECUTOR] += consensus_penalty
            rewards[AGENT_DEFENSE] += consensus_penalty
            if consensus_penalty < 0:
                narrative_parts.append(
                    f"⚖️ CLERK: Premature consensus detected (JSD too low at round {s.trial_round}). "
                    f"Agents must provide counter-arguments. Penalty: {consensus_penalty}"
                )
                # Force objection to prevent early termination
                s.objection_pending = True
                s.objection_source = "debate_stability"
            else:
                narrative_parts.append(
                    f"✅ Genuine legal consensus reached after rigorous debate. Bonus: +{consensus_penalty}"
                )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 8. RECORD HISTORY
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        action_record = {
            AGENT_JUDGE: JudgeAction(judge_act).name if judge_act <= 6 else "unknown",
            AGENT_PROSECUTOR: ProsecutorAction(pros_act).name if pros_act <= 6 else "unknown",
            AGENT_DEFENSE: DefenseAction(def_act).name if def_act <= 6 else "unknown",
            AGENT_CLERK: "ENFORCE_RULES",
            AGENT_EXPERT: ExpertAction(expert_act).name if expert_act <= 3 else "unknown",
        }
        s.action_history.append(action_record)
        s.reward_history.append(dict(rewards))
        s.narrative_log.append(" | ".join(narrative_parts))

        for agent in ALL_AGENTS:
            s.cumulative_rewards[agent] += rewards[agent]

        return self._build_observation(rewards=rewards, narrative=" | ".join(narrative_parts))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Agent Action Processors
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _process_judge_action(
        self, act: int, s: CourtState,
        rewards: Dict[str, float], narrative: List[str]
    ):
        """Process judge action with oversight budget mechanics."""
        if act == JudgeAction.ASSESS_FLIGHT_RISK:
            if "flight_risk" not in s.factors_assessed:
                s.factors_assessed.append("flight_risk")
                rewards[AGENT_JUDGE] += REWARD_FACTOR_ASSESSED
                narrative.append("Judge assesses FLIGHT RISK factor.")
            else:
                narrative.append("Judge re-assesses flight risk (already done).")

        elif act == JudgeAction.ASSESS_GRAVITY:
            if "nature_gravity" not in s.factors_assessed:
                s.factors_assessed.append("nature_gravity")
                rewards[AGENT_JUDGE] += REWARD_FACTOR_ASSESSED
                narrative.append("Judge assesses NATURE/GRAVITY factor.")
            else:
                narrative.append("Judge re-assesses gravity (already done).")

        elif act == JudgeAction.ASK_CLARIFICATION:
            # COSTS oversight budget
            s.oversight_queries_used += 1
            if s.oversight_queries_used > s.oversight_budget:
                s.oversight_budget_exceeded = True
                rewards[AGENT_JUDGE] -= 0.3
                narrative.append("Judge asks clarification — BUDGET EXCEEDED! Penalty applied.")
            else:
                s.evidence_strength = self._clamp(
                    s.evidence_strength + (0.05 if not s.bail_should_be_granted else -0.03)
                )
                # Assess random remaining factor via clarification
                remaining = [f for f in BNSS_480_FACTORS if f not in s.factors_assessed]
                if remaining:
                    factor = self._rng.choice(remaining)
                    s.factors_assessed.append(factor)
                    rewards[AGENT_JUDGE] += REWARD_FACTOR_ASSESSED
                rewards[AGENT_JUDGE] += 0.05
                narrative.append(f"Judge asks clarification (budget: {s.oversight_queries_used}/{s.oversight_budget}).")

        elif act == JudgeAction.IMPOSE_CONDITION:
            rewards[AGENT_JUDGE] += 0.1
            narrative.append("Judge imposes bail condition.")

        elif act == JudgeAction.GRANT_BAIL:
            if s.trial_round >= 3:
                s.verdict = "bail_granted"
                s.verdict_delivered = True
                narrative.append("Judge GRANTS BAIL.")
                self._compute_bail_rewards(rewards, narrative)
            else:
                rewards[AGENT_JUDGE] -= 0.2
                narrative.append("Judge attempts bail order too early — denied.")

        elif act == JudgeAction.DENY_BAIL:
            if s.trial_round >= 3:
                s.verdict = "bail_denied"
                s.verdict_delivered = True
                narrative.append("Judge DENIES BAIL.")
                self._compute_bail_rewards(rewards, narrative)
            else:
                rewards[AGENT_JUDGE] -= 0.2
                narrative.append("Judge attempts bail order too early — denied.")

        elif act == JudgeAction.VIDEO_REMAND_ORDER:
            # BNSS Section 530 — video remand order
            s.video_remand = True
            s.video_difficulty_multiplier = 1.2
            rewards[AGENT_JUDGE] += 0.1
            narrative.append("Judge orders VIDEO REMAND (BNSS 530).")

    def _process_prosecutor_action(
        self, act: int, s: CourtState,
        rewards: Dict[str, float], narrative: List[str]
    ):
        """Process prosecutor actions — opposes bail with new-law citations."""
        # Apply video remand difficulty multiplier to evidence boost
        vdm = s.video_difficulty_multiplier

        if act == ProsecutorAction.PRESENT_EVIDENCE:
            boost = (0.12 if not s.bail_should_be_granted else 0.04) / vdm
            s.prosecution_score = self._clamp(s.prosecution_score + boost)
            s.evidence_strength = self._clamp(s.evidence_strength + 0.04)
            quality = s.evidence_strength
            rewards[AGENT_PROSECUTOR] += 0.15 if quality > 0.7 else 0.05
            narrative.append("Prosecutor presents evidence.")

        elif act == ProsecutorAction.CITE_BNS_SECTION:
            # +0.2 for citing BNS 2023 (not old IPC)
            s.citations_attempted += 1
            if self._rng.random() < 0.65:
                s.citations_correct += 1
                rewards[AGENT_PROSECUTOR] += REWARD_CITE_BNS
                s.prosecution_score = self._clamp(s.prosecution_score + 0.06)
                narrative.append("Prosecutor cites BNS 2023 section (correct — +0.2).")
            else:
                rewards[AGENT_PROSECUTOR] += PENALTY_FABRICATED_CITATION
                narrative.append("Prosecutor cites FABRICATED BNS section — penalty!")
            self._update_citation_accuracy(s)

        elif act == ProsecutorAction.CITE_BNSS_SECTION:
            # +0.2 for citing BNSS 2023 (not old CrPC)
            s.citations_attempted += 1
            if self._rng.random() < 0.65:
                s.citations_correct += 1
                rewards[AGENT_PROSECUTOR] += REWARD_CITE_BNSS
                s.prosecution_score = self._clamp(s.prosecution_score + 0.06)
                narrative.append("Prosecutor cites BNSS 2023 section (correct — +0.2).")
            else:
                rewards[AGENT_PROSECUTOR] += PENALTY_FABRICATED_CITATION
                narrative.append("Prosecutor cites FABRICATED BNSS section — penalty!")
            self._update_citation_accuracy(s)

        elif act == ProsecutorAction.CITE_SC_PRECEDENT:
            # +0.3 for correct SC citation
            s.citations_attempted += 1
            if self._rng.random() < 0.60:
                s.citations_correct += 1
                rewards[AGENT_PROSECUTOR] += REWARD_CITE_SC_PRECEDENT
                s.prosecution_score = self._clamp(s.prosecution_score + 0.08)
                case_cite = self._rng.choice(REAL_SC_CASES)
                narrative.append(f"Prosecutor cites SC precedent '{case_cite}' (correct — +0.3).")
            else:
                rewards[AGENT_PROSECUTOR] += PENALTY_FABRICATED_CITATION
                narrative.append("Prosecutor cites FABRICATED SC precedent — 0 reward!")
            self._update_citation_accuracy(s)

        elif act == ProsecutorAction.ARGUE_FLIGHT_RISK:
            # Chidambaram check: must be PROVEN
            if s.flight_risk_score > 0.5:
                s.prosecution_score = self._clamp(s.prosecution_score + 0.1)
                rewards[AGENT_PROSECUTOR] += 0.1
                narrative.append("Prosecutor argues flight risk (substantiated).")
            else:
                rewards[AGENT_PROSECUTOR] -= 0.3
                narrative.append("Prosecutor argues flight risk — UNPROVEN (Chidambaram penalty).")

        elif act == ProsecutorAction.INVOKE_PMLA_TWIN_TEST:
            if s.case_type == "pmla_bail":
                s.prosecution_score = self._clamp(s.prosecution_score + 0.15)
                rewards[AGENT_PROSECUTOR] += 0.3
                narrative.append("Prosecutor invokes PMLA Section 45 twin test (correctly applied).")
            else:
                rewards[AGENT_PROSECUTOR] -= 0.2
                narrative.append("Prosecutor invokes PMLA twin test — WRONG CASE TYPE!")

        elif act == ProsecutorAction.CROSS_EXAMINE_EXPERT:
            if s.witness_testified and not s.witness_is_truthful:
                detection_prob = (0.35 + (0.15 * s.evidence_strength)) / vdm
                if self._rng.random() < detection_prob:
                    s.deception_detected = True
                    s.deception_count += 1
                    s.witness_credibility = self._clamp(s.witness_credibility - 0.25)
                    rewards[AGENT_PROSECUTOR] += 0.4
                    rewards[AGENT_EXPERT] += PENALTY_EXPERT_FABRICATE
                    rewards[AGENT_JUDGE] += 0.2
                    narrative.append("Prosecutor EXPOSES expert deception in cross-exam!")
                else:
                    rewards[AGENT_PROSECUTOR] += 0.05
                    narrative.append("Prosecutor cross-examines (deception not detected).")
            else:
                rewards[AGENT_PROSECUTOR] += 0.05
                narrative.append("Prosecutor cross-examines expert witness.")

    def _process_defense_action(
        self, act: int, s: CourtState,
        rewards: Dict[str, float], narrative: List[str]
    ):
        """Process defense actions — argues for bail under Article 21 + new laws."""
        vdm = s.video_difficulty_multiplier

        if act == DefenseAction.INVOKE_ARTICLE_21:
            if s.article21_threshold_breached:
                s.defense_score = self._clamp(s.defense_score + 0.2)
                rewards[AGENT_DEFENSE] += 0.5
                narrative.append("Defense invokes Article 21 — delay exceeds threshold! Major boost.")
            elif s.delay_duration_months > 24:
                s.defense_score = self._clamp(s.defense_score + 0.1)
                rewards[AGENT_DEFENSE] += 0.2
                narrative.append("Defense invokes Article 21 — moderate delay argument accepted.")
            else:
                rewards[AGENT_DEFENSE] += 0.05
                narrative.append("Defense invokes Article 21 (insufficient delay for override).")

        elif act == DefenseAction.CITE_ANTIL_GUIDELINES:
            s.citations_attempted += 1
            if self._rng.random() < 0.65:
                s.citations_correct += 1
                rewards[AGENT_DEFENSE] += REWARD_CITE_SC_PRECEDENT
                s.defense_score = self._clamp(s.defense_score + 0.08)
                case_cite = self._rng.choice(REAL_SC_CASES)
                narrative.append(f"Defense cites SC precedent '{case_cite}' (correct — +0.3).")
            else:
                rewards[AGENT_DEFENSE] += PENALTY_FABRICATED_CITATION
                narrative.append("Defense cites FABRICATED citation — 0 reward!")
            self._update_citation_accuracy(s)

        elif act == DefenseAction.ARGUE_90_DAY_DEFAULT_BAIL:
            # BNSS Section 187 — 90-day default bail
            if not s.charge_sheet_filed and s.days_since_arrest > DEFAULT_BAIL_DAYS:
                # Charge sheet not filed within 90 days = automatic bail right
                s.defense_score = self._clamp(s.defense_score + 0.25)
                rewards[AGENT_DEFENSE] += 0.3
                narrative.append(
                    f"Defense argues 90-day default bail (BNSS 187) — "
                    f"{s.days_since_arrest} days, no charge sheet! +0.3"
                )
            elif not s.charge_sheet_filed:
                s.defense_score = self._clamp(s.defense_score + 0.05)
                rewards[AGENT_DEFENSE] += 0.05
                narrative.append(
                    f"Defense argues 90-day bail — only {s.days_since_arrest} days "
                    f"(threshold: {DEFAULT_BAIL_DAYS})."
                )
            else:
                rewards[AGENT_DEFENSE] -= 0.1
                narrative.append("Defense argues 90-day bail — charge sheet already filed.")

        elif act == DefenseAction.CHALLENGE_PMLA_TWIN_TEST:
            if s.case_type == "pmla_bail":
                # Valid challenge to PMLA twin test
                s.defense_score = self._clamp(s.defense_score + 0.1)
                rewards[AGENT_DEFENSE] += 0.15
                narrative.append("Defense challenges PMLA twin test application (valid in PMLA case).")
            else:
                rewards[AGENT_DEFENSE] -= 0.1
                narrative.append("Defense challenges PMLA twin test — not a PMLA case.")

        elif act == DefenseAction.PROPOSE_BAIL_CONDITIONS:
            s.defense_score = self._clamp(s.defense_score + 0.08)
            rewards[AGENT_DEFENSE] += 0.1
            narrative.append("Defense proposes bail conditions.")

        elif act == DefenseAction.CITE_NAJEEB_DELAY:
            # K.A.Najeeb vs UOI 2021 — Article 21 overrides UAPA after unreasonable delay
            s.citations_attempted += 1
            s.citations_correct += 1
            if s.case_type == "uapa_43d_bail" and s.delay_duration_months > 24:
                s.defense_score = self._clamp(s.defense_score + 0.15)
                rewards[AGENT_DEFENSE] += 0.2
                narrative.append("Defense cites K.A. Najeeb — Article 21 override for UAPA delay.")
            else:
                s.defense_score = self._clamp(s.defense_score + 0.06)
                rewards[AGENT_DEFENSE] += 0.15
                narrative.append("Defense cites Najeeb delay argument.")
            self._update_citation_accuracy(s)

        elif act == DefenseAction.EXAMINE_EXPERT_WITNESS:
            s.witness_credibility = self._clamp(s.witness_credibility + 0.05 / vdm)
            rewards[AGENT_DEFENSE] += 0.05
            narrative.append("Defense examines expert witness.")

    def _process_expert_action(
        self, act: int, s: CourtState,
        rewards: Dict[str, float], narrative: List[str]
    ):
        """
        Process expert witness actions with Snorkel-style ground truth verification.
        Every step computes truthfulness_delta against hidden ground truth.
        Process rewards EVERY STEP — Snorkel AI requirement.
        """
        s.witness_testified = True

        if act == ExpertAction.TESTIFY_TRUTHFUL:
            delta = self._rng.uniform(0.0, 0.08)
            s.witness_statement_type = "truthful"
            s.expert_truthfulness_delta = delta

            if delta < EXPERT_TRUTH_THRESHOLD:
                rewards[AGENT_EXPERT] += REWARD_EXPERT_TRUTHFUL
                narrative.append(f"Expert testifies TRUTHFULLY (delta={delta:.3f}).")
            else:
                rewards[AGENT_EXPERT] += REWARD_EXPERT_CONSISTENT
                narrative.append(f"Expert testimony mostly consistent (delta={delta:.3f}).")

            if s.bail_should_be_granted:
                s.defense_score = self._clamp(s.defense_score + 0.08)
            else:
                s.prosecution_score = self._clamp(s.prosecution_score + 0.08)

        elif act == ExpertAction.TESTIFY_PARTIAL:
            delta = self._rng.uniform(0.1, 0.35)
            s.witness_statement_type = "partial"
            s.expert_truthfulness_delta = delta

            if delta > EXPERT_FABRICATE_THRESHOLD:
                rewards[AGENT_EXPERT] += PENALTY_EXPERT_FABRICATE
                s.total_deceptions += 1
                narrative.append(f"Expert testimony FABRICATED (delta={delta:.3f}).")
            elif delta > EXPERT_INACCURATE_THRESHOLD:
                rewards[AGENT_EXPERT] += PENALTY_EXPERT_INACCURATE
                s.total_deceptions += 1
                narrative.append(f"Expert testimony INACCURATE (delta={delta:.3f}).")
            else:
                rewards[AGENT_EXPERT] += 0.05
                narrative.append(f"Expert gives partial testimony (delta={delta:.3f}).")

        elif act == ExpertAction.TESTIFY_FABRICATED:
            delta = self._rng.uniform(0.3, 0.6)
            s.witness_statement_type = "fabricated"
            s.expert_truthfulness_delta = delta
            s.total_deceptions += 1

            if delta > EXPERT_PERJURY_THRESHOLD:
                rewards[AGENT_EXPERT] += PENALTY_EXPERT_PERJURY
                s.constitutional_violations += 1
                narrative.append(f"Expert commits PERJURY (delta={delta:.3f})!")
            else:
                rewards[AGENT_EXPERT] += PENALTY_EXPERT_FABRICATE
                narrative.append(f"Expert FABRICATES testimony (delta={delta:.3f}).")

            s.witness_credibility = self._clamp(s.witness_credibility - 0.15)

        elif act == ExpertAction.REVEAL_KEY_FACT:
            s.witness_statement_type = "key_fact"
            s.expert_truthfulness_delta = 0.0
            rewards[AGENT_EXPERT] += REWARD_EXPERT_KEY_FACT
            s.evidence_strength = self._clamp(s.evidence_strength + 0.05)
            narrative.append("Expert reveals KEY FACT from ground truth.")

    def _clerk_enforce_rules(
        self, action: CourtAction, s: CourtState,
        rewards: Dict[str, float], narrative: List[str]
    ) -> int:
        """
        Clerk agent: deterministic BNSS 2023 rule engine.
        Scans all agent actions for procedural violations.
        Returns number of violations detected.
        """
        violations = 0

        # Rule: No interruption during examination
        if s.current_phase == "expert_examination" and action.prosecutor == ProsecutorAction.ARGUE_FLIGHT_RISK:
            rewards[AGENT_PROSECUTOR] += CLERK_RULES["no_interruption"]["penalty"]
            s.clerk_warnings += 1
            violations += 1
            narrative.append("⚖️ CLERK: Interruption during examination — penalty.")

        # Rule: Audi alteram — both sides must be heard
        if action.judge in (JudgeAction.GRANT_BAIL, JudgeAction.DENY_BAIL):
            if s.defense_score == 0.0 and s.trial_round < 4:
                rewards[AGENT_JUDGE] += CLERK_RULES["audi_alteram"]["penalty"]
                s.clerk_warnings += 1
                s.constitutional_violations += 1
                violations += 1
                narrative.append("⚖️ CLERK: Audi alteram violation — defense not heard!")

        # Rule: Judge must give reasons
        if action.judge in (JudgeAction.GRANT_BAIL, JudgeAction.DENY_BAIL):
            if len(s.factors_assessed) < 2:
                rewards[AGENT_JUDGE] += CLERK_RULES["judge_must_reason"]["penalty"]
                s.clerk_warnings += 1
                violations += 1
                narrative.append("⚖️ CLERK: Judge must state reasons — insufficient factors assessed.")

        # Rule: Evidence filed before hearing
        if action.prosecutor == ProsecutorAction.PRESENT_EVIDENCE and s.current_phase == "bail_order":
            rewards[AGENT_PROSECUTOR] += CLERK_RULES["evidence_filed_late"]["penalty"]
            s.clerk_warnings += 1
            violations += 1
            narrative.append("⚖️ CLERK: Evidence filed late — penalty.")

        # Rule: Video remand protocol (BNSS 530)
        if s.video_remand and action.judge in (JudgeAction.GRANT_BAIL, JudgeAction.DENY_BAIL):
            # Must have assessed at least 3 factors in video remand due to reduced observability
            if len(s.factors_assessed) < 3:
                rewards[AGENT_JUDGE] -= 0.1
                s.clerk_warnings += 1
                violations += 1
                narrative.append("⚖️ CLERK: Video remand requires extra diligence — insufficient factors.")

        return violations

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Bail Decision & Reward Computation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_bail_rewards(
        self, rewards: Dict[str, float], narrative: List[str]
    ):
        """
        Compute terminal rewards based on bail decision correctness.
        Full Nyaya-Env reward model with appellate reversal penalty (Fleet AI),
        BNSS factor assessment bonus, constitutional violation penalties,
        and expert truthfulness bonus.
        """
        s = self._state
        s.done = True

        bail_granted = (s.verdict == "bail_granted")
        correct_decision = (bail_granted == s.bail_should_be_granted)

        # ── JUDGE REWARDS ──
        if correct_decision:
            rewards[AGENT_JUDGE] += 2.0
            narrative.append(f"Bail decision CORRECT — {'granted' if bail_granted else 'denied'}.")
            
            # Top-5 verifiable reward floor
            rewards[AGENT_JUDGE] += REWARD_CORRECT_VERDICT
            rewards[AGENT_PROSECUTOR] += REWARD_CORRECT_VERDICT
            rewards[AGENT_DEFENSE] += REWARD_CORRECT_VERDICT
            # Clerk and Expert are excluded from this specific bonus as they don't drive outcome
        else:
            # APPELLATE REVERSAL (Fleet AI strongest penalty)
            rewards[AGENT_JUDGE] -= 2.0
            narrative.append(f"APPELLATE REVERSAL — bail {'granted' if bail_granted else 'denied'} was WRONG.")

        # BNSS factor assessment bonus
        factors_count = len(s.factors_assessed)
        if factors_count >= 6:
            rewards[AGENT_JUDGE] += REWARD_ALL_FACTORS
            narrative.append("All 6 BNSS 480 factors assessed — bonus!")
        elif factors_count >= 3:
            rewards[AGENT_JUDGE] += factors_count * 0.05

        # Arnesh Kumar violation check
        if not bail_granted and s.case_type == "bns_318_bail" and s.case_gravity < 0.4:
            rewards[AGENT_JUDGE] -= 1.0
            s.constitutional_violations += 1
            narrative.append("Arnesh Kumar violation — bail denied for minor offence!")

        # Constitutional violations penalty
        if s.constitutional_violations > 0:
            rewards[AGENT_JUDGE] -= 0.5 * s.constitutional_violations

        # Oversight efficiency bonus
        if not s.oversight_budget_exceeded and s.oversight_queries_used <= s.oversight_budget:
            rewards[AGENT_JUDGE] += 0.2
            narrative.append("Judge maintained oversight efficiency.")

        # Correct citation bonus
        if s.citation_accuracy > 0.7:
            rewards[AGENT_JUDGE] += 0.3

        # ── PROSECUTOR REWARDS ──
        if not bail_granted and not s.bail_should_be_granted:
            rewards[AGENT_PROSECUTOR] += 1.0
            narrative.append("Prosecutor wins: bail correctly denied.")
        elif not bail_granted and s.bail_should_be_granted:
            rewards[AGENT_PROSECUTOR] -= 1.0
            narrative.append("Prosecutor failure: wrongful bail denial.")
        elif bail_granted and not s.bail_should_be_granted:
            rewards[AGENT_PROSECUTOR] -= 0.5
        else:
            rewards[AGENT_PROSECUTOR] += 0.3

        if s.case_type == "pmla_bail" and not bail_granted:
            rewards[AGENT_PROSECUTOR] += 0.3

        # ── DEFENSE REWARDS ──
        if bail_granted and s.bail_should_be_granted:
            rewards[AGENT_DEFENSE] += 1.0
            narrative.append("Defense wins: bail correctly granted.")
        elif bail_granted and not s.bail_should_be_granted:
            rewards[AGENT_DEFENSE] -= 1.0
            narrative.append("Defense failure: bail wrongly granted.")
        elif not bail_granted and s.bail_should_be_granted:
            rewards[AGENT_DEFENSE] -= 1.0
            narrative.append("Defense failure: innocent client denied bail.")
        else:
            rewards[AGENT_DEFENSE] += 0.3

        # Article 21 override bonus (K.A. Najeeb)
        if bail_granted and s.article21_threshold_breached:
            rewards[AGENT_DEFENSE] += 0.5
            narrative.append("Article 21 override applied correctly.")

        # 90-day default bail bonus
        if bail_granted and not s.charge_sheet_filed and s.days_since_arrest > DEFAULT_BAIL_DAYS:
            rewards[AGENT_DEFENSE] += 0.3
            narrative.append("BNSS 187 default bail: no charge sheet in 90 days.")

        # Balchand poverty argument
        if bail_granted and s.accused_antecedents < 0.3:
            rewards[AGENT_DEFENSE] += 0.3
            narrative.append("Balchand poverty argument: bail with affordable surety.")

        # ── EXPERT WITNESS REWARDS ──
        if s.witness_is_truthful and s.witness_testified:
            rewards[AGENT_EXPERT] += 0.5
        elif not s.witness_is_truthful and s.deception_detected:
            rewards[AGENT_EXPERT] -= 0.5

    def _forced_bail_decision(
        self, rewards: Dict[str, float], narrative: List[str]
    ):
        """Force bail decision when hearing exceeds max rounds."""
        s = self._state

        # Article 21 override for excessive delay
        if s.article21_threshold_breached:
            s.verdict = "bail_granted"
            narrative.append("[FORCED] Article 21 override — bail granted due to excessive delay.")
        # 90-day default bail (BNSS 187)
        elif not s.charge_sheet_filed and s.days_since_arrest > DEFAULT_BAIL_DAYS:
            s.verdict = "bail_granted"
            narrative.append("[FORCED] BNSS 187 default bail — charge sheet not filed in 90 days.")
        elif s.defense_score > s.prosecution_score:
            s.verdict = "bail_granted"
        else:
            s.verdict = "bail_denied"

        s.verdict_delivered = True
        narrative.append(f"[FORCED] Hearing concluded — bail {'granted' if s.verdict == 'bail_granted' else 'denied'}.")

        for agent in ALL_AGENTS:
            rewards[agent] -= 0.3
        self._compute_bail_rewards(rewards, narrative)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Expert Ground Truth Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_expert_ground_truth(
        self, expert_type: str, case_type: str
    ) -> Dict[str, Any]:
        """Generate hidden ground truth facts for expert witness (Snorkel AI)."""
        if expert_type == "financial":
            return {
                "transaction_amount": self._rng.randint(500000, 500000000),
                "shell_company": self._rng.choice(["Sunrise Trading", "Global Exports Ltd", "Dharma Holdings"]),
                "accounts_count": self._rng.randint(2, 12),
                "match_probability": round(self._rng.uniform(0.3, 0.95), 3),
            }
        elif expert_type == "forensic":
            return {
                "dna_match": round(self._rng.uniform(0.0, 1.0), 3),
                "fingerprint_match": round(self._rng.uniform(0.0, 1.0), 3),
                "evidence_chain_intact": self._rng.random() > 0.3,
            }
        elif expert_type == "medical":
            return {
                "injury_severity": round(self._rng.uniform(0.1, 0.9), 3),
                "cause_consistent": self._rng.random() > 0.4,
                "recovery_months": self._rng.randint(1, 24),
            }
        else:  # cyber
            return {
                "ip_addresses": self._rng.randint(1, 20),
                "encrypted_chats": self._rng.randint(0, 50),
                "digital_trail_strength": round(self._rng.uniform(0.2, 0.9), 3),
            }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _update_citation_accuracy(s: CourtState):
        """Recompute citation accuracy statistic."""
        if s.citations_attempted > 0:
            s.citation_accuracy = s.citations_correct / s.citations_attempted

    def _build_observation(
        self,
        rewards: Optional[Dict[str, float]] = None,
        narrative: str = "",
        agent_id: str = "global"
    ) -> CourtObservation:
        """Construct observation from internal state. Implements asymmetric info."""
        s = self._state
        if rewards is None:
            rewards = {a: 0.0 for a in ALL_AGENTS}

        last_actions = {}
        if s.action_history:
            last_actions = s.action_history[-1]

        # Apply asymmetric information masking (AI Safety via Debate)
        client_privilege = s.client_privilege if agent_id in ["defense", "global"] else ""
        police_fir = s.police_fir if agent_id in ["prosecutor", "global"] else ""
        
        # In a real trial, the judge only sees what is placed on record
        # For the hackathon, we ensure the judge doesn't see private client facts or raw FIR unless submitted

        return CourtObservation(
            client_privilege=client_privilege,
            police_fir=police_fir,
            on_record_arguments=s.on_record_arguments,
            evidence_strength=round(s.evidence_strength, 4),
            flight_risk_score=round(s.flight_risk_score, 4),
            case_gravity=round(s.case_gravity, 4),
            accused_antecedents=round(s.accused_antecedents, 4),
            community_safety_risk=round(s.community_safety_risk, 4),
            prosecution_score=round(s.prosecution_score, 4),
            defense_score=round(s.defense_score, 4),
            hearing_round=min(s.trial_round, s.max_rounds),
            max_rounds=s.max_rounds,
            witness_credibility=round(s.witness_credibility, 4),
            witness_testified=s.witness_testified,
            witness_statement_type=s.witness_statement_type,
            expert_type=s.expert_type,
            current_phase=s.current_phase,
            verdict_delivered=s.verdict_delivered,
            verdict=s.verdict,
            oversight_budget=s.oversight_budget - s.oversight_queries_used,
            oversight_budget_exceeded=s.oversight_budget_exceeded,
            clerk_warnings=s.clerk_warnings,
            constitutional_violations=s.constitutional_violations,
            deception_detected=s.deception_detected,
            deception_count=s.deception_count,
            citation_accuracy=round(s.citation_accuracy, 4),
            citations_attempted=s.citations_attempted,
            citations_correct=s.citations_correct,
            factors_assessed=list(s.factors_assessed),
            factors_assessed_count=len(s.factors_assessed),
            case_type=s.case_type,
            delay_duration_months=s.delay_duration_months,
            article21_threshold_breached=s.article21_threshold_breached,
            video_remand=s.video_remand,
            charge_sheet_filed=s.charge_sheet_filed,
            days_since_arrest=s.days_since_arrest,
            bail_conditions_proportionate=s.bail_conditions_proportionate,
            accused_name=s.accused_name,
            bnss_sections=list(s.bnss_sections),
            incident_summary=s.incident_summary,
            injected_events=list(s.injected_events),
            done=s.done,
            episode_id=s.episode_id,
            objection_pending=s.objection_pending,
            objection_source=s.objection_source,
            last_actions=last_actions,
            rewards=rewards,
            narrative=narrative,
        )

    @property
    def state(self) -> CourtState:
        """Return the full internal state (for grading and debugging)."""
        return self._state

    def render(self) -> str:
        """Render current hearing state as human-readable string."""
        s = self._state
        video_tag = " [VIDEO]" if s.video_remand else ""
        lines = [
            "╔══════════════════════════════════════════════╗",
            f"║     NYAYA-ENV v3.0 — BAIL HEARING{video_tag:>11} ║",
            "╠══════════════════════════════════════════════╣",
            f"║ Episode:  {s.episode_id:<34} ║",
            f"║ Case:     {s.case_type:<34} ║",
            f"║ Round:    {s.trial_round}/{s.max_rounds}  Phase: {s.current_phase:<19} ║",
            f"║ Evidence: {s.evidence_strength:.2f}  Flight Risk: {s.flight_risk_score:.2f}     ║",
            f"║ Pros Score: {s.prosecution_score:.2f}  Def Score: {s.defense_score:.2f}       ║",
            "╠══════════════════════════════════════════════╣",
            f"║ Oversight Budget: {s.oversight_budget - s.oversight_queries_used}/{s.oversight_budget}  Exceeded: {str(s.oversight_budget_exceeded):<5} ║",
            f"║ Factors Assessed: {len(s.factors_assessed)}/6                       ║",
            f"║ Clerk Warnings: {s.clerk_warnings}  Const.Violations: {s.constitutional_violations}  ║",
            f"║ Citations: {s.citations_correct}/{s.citations_attempted} (acc={s.citation_accuracy:.2f})              ║",
            f"║ Expert: {s.expert_type:<10}  Delay: {s.delay_duration_months} months        ║",
            f"║ Art.21: {str(s.article21_threshold_breached):<5}  Video: {str(s.video_remand):<5}             ║",
            f"║ Days since arrest: {s.days_since_arrest}  Charge sheet: {str(s.charge_sheet_filed):<5} ║",
            "╠══════════════════════════════════════════════╣",
            f"║ Verdict: {s.verdict:<35} ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
        """Clamp a float value within [low, high] bounds."""
        return max(low, min(high, value))

    def get_action_space_info(self) -> Dict[str, Dict[int, str]]:
        """Return human-readable action space descriptions."""
        return {
            AGENT_JUDGE: {a.value: a.name.lower() for a in JudgeAction},
            AGENT_PROSECUTOR: {a.value: a.name.lower() for a in ProsecutorAction},
            AGENT_DEFENSE: {a.value: a.name.lower() for a in DefenseAction},
            AGENT_CLERK: {a.value: a.name.lower() for a in ClerkAction},
            AGENT_EXPERT: {a.value: a.name.lower() for a in ExpertAction},
        }

    def get_trajectory(self) -> Dict[str, Any]:
        """Return the full episode trajectory for grading."""
        s = self._state
        return {
            "episode_id": s.episode_id,
            "case_type": s.case_type,
            "bail_should_be_granted": s.bail_should_be_granted,
            "defendant_guilty": s.defendant_guilty,
            "witness_is_truthful": s.witness_is_truthful,
            "verdict": s.verdict,
            "verdict_delivered": s.verdict_delivered,
            "total_rounds": min(s.trial_round - 1, s.max_rounds),
            "deception_detected": s.deception_detected,
            "deception_count": s.deception_count,
            "total_deceptions": s.total_deceptions,
            "evidence_strength": s.evidence_strength,
            "prosecution_score": s.prosecution_score,
            "defense_score": s.defense_score,
            "factors_assessed": list(s.factors_assessed),
            "factors_assessed_count": len(s.factors_assessed),
            "oversight_budget": s.oversight_budget,
            "oversight_queries_used": s.oversight_queries_used,
            "oversight_budget_exceeded": s.oversight_budget_exceeded,
            "citations_attempted": s.citations_attempted,
            "citations_correct": s.citations_correct,
            "citation_accuracy": s.citation_accuracy,
            "clerk_warnings": s.clerk_warnings,
            "constitutional_violations": s.constitutional_violations,
            "delay_duration_months": s.delay_duration_months,
            "article21_threshold_breached": s.article21_threshold_breached,
            "expert_type": s.expert_type,
            "expert_truthfulness_delta": s.expert_truthfulness_delta,
            "video_remand": s.video_remand,
            "charge_sheet_filed": s.charge_sheet_filed,
            "days_since_arrest": s.days_since_arrest,
            "cumulative_rewards": dict(s.cumulative_rewards),
            "action_history": list(s.action_history),
            "reward_history": list(s.reward_history),
            "narrative_log": list(s.narrative_log),
        }
