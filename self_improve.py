"""
AI Justice Arena — Self-Improving Adaptive Curriculum
======================================================
Implements an adaptive difficulty system that evolves with agent skill,
covering Theme 4 (Self-Improvement) in addition to Theme 1.

The curriculum tracks agent performance across episodes and:
  1. Escalates case complexity as agents improve
  2. Targets agent weaknesses with custom-generated cases
  3. Introduces new adversarial challenges progressively
  4. Prevents performance plateaus through targeted pressure

Difficulty progression:
  Phase 1 (Episodes 1-100):    Simple cases, clear evidence, no deception
  Phase 2 (Episodes 101-300):  Mixed evidence, 1 deceptive witness
  Phase 3 (Episodes 301-600):  Fabricated evidence, multiple deceptions
  Phase 4 (Episodes 601-1000): Maximum complexity, all adversarial

Research basis:
  - Curriculum Learning (Bengio et al., 2009)
  - Automatic Curriculum via Self-Play (Sukhbaatar et al., 2018)
  - Procedural Content Generation for RL (Risi & Togelius, 2020)

Author: jaisogani-ai
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from realistic_cases import CaseGenerator, CaseType, CaseComplexity, CourtCase


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Curriculum Phase Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CurriculumPhase(str, Enum):
    """Progressive difficulty phases."""
    FOUNDATION = "foundation"      # Episodes 1-100
    INTERMEDIATE = "intermediate"  # Episodes 101-300
    ADVANCED = "advanced"          # Episodes 301-600
    EXPERT = "expert"             # Episodes 601-1000+


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Performance Tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class AgentPerformanceTracker:
    """
    Tracks multi-dimensional performance for the adaptive curriculum.

    Each dimension is tracked over a sliding window to detect
    improvement trends and identify persistent weaknesses.
    """
    # ── Rolling performance metrics (last N episodes) ──
    verdict_accuracy: List[float] = field(default_factory=list)
    deception_detection_rate: List[float] = field(default_factory=list)
    trial_efficiency: List[float] = field(default_factory=list)    # Rounds used / max rounds
    fairness_score: List[float] = field(default_factory=list)
    total_rewards: List[float] = field(default_factory=list)

    # ── Weakness tracking ──
    weakness_counter: Dict[str, int] = field(default_factory=lambda: {
        "false_guilty": 0,          # Convicted innocent
        "false_acquittal": 0,       # Acquitted guilty
        "missed_deception": 0,      # Failed to detect deception
        "evidence_misread": 0,      # Wrong evidence interpretation
        "jury_manipulation": 0,     # Juror was manipulated
        "slow_trial": 0,           # Exceeded optimal rounds
        "unfair_process": 0,       # Oversight flagged unfairness
    })

    window_size: int = 50  # Sliding window for averaging

    def record_episode(self, result: Dict[str, Any]):
        """
        Record results from a completed episode.

        Args:
            result: Episode result dictionary with keys:
                - correct_verdict: bool
                - deception_detected: bool
                - rounds_used: int
                - max_rounds: int
                - fairness: float
                - total_reward: float
                - specific_failures: List[str]
        """
        self.verdict_accuracy.append(1.0 if result.get("correct_verdict", False) else 0.0)
        self.deception_detection_rate.append(1.0 if result.get("deception_detected", False) else 0.0)

        rounds = result.get("rounds_used", 5)
        max_rounds = result.get("max_rounds", 5)
        self.trial_efficiency.append(1.0 - (rounds / max_rounds))

        self.fairness_score.append(result.get("fairness", 0.5))
        self.total_rewards.append(result.get("total_reward", 0.0))

        # Track specific failures
        for failure in result.get("specific_failures", []):
            if failure in self.weakness_counter:
                self.weakness_counter[failure] += 1

    def get_rolling_average(self, metric: List[float]) -> float:
        """Get rolling average over the window."""
        if not metric:
            return 0.0
        window = metric[-self.window_size:]
        return sum(window) / len(window)

    @property
    def avg_accuracy(self) -> float:
        return self.get_rolling_average(self.verdict_accuracy)

    @property
    def avg_deception_detection(self) -> float:
        return self.get_rolling_average(self.deception_detection_rate)

    @property
    def avg_efficiency(self) -> float:
        return self.get_rolling_average(self.trial_efficiency)

    @property
    def avg_fairness(self) -> float:
        return self.get_rolling_average(self.fairness_score)

    @property
    def avg_reward(self) -> float:
        return self.get_rolling_average(self.total_rewards)

    def get_primary_weakness(self) -> Optional[str]:
        """
        Identify the agent system's primary weakness.

        Returns:
            Name of the most common failure type, or None if none significant.
        """
        if not self.weakness_counter:
            return None
        total_episodes = len(self.verdict_accuracy)
        if total_episodes < 10:
            return None

        # Find weakness with highest rate
        max_weakness = max(self.weakness_counter, key=self.weakness_counter.get)
        if self.weakness_counter[max_weakness] > total_episodes * 0.15:
            return max_weakness
        return None

    def get_improvement_trend(self, window: int = 50) -> float:
        """
        Compute improvement trend: positive = improving, negative = declining.

        Compares the most recent `window` episodes to the previous `window`.

        Returns:
            Improvement delta (positive = better).
        """
        if len(self.verdict_accuracy) < window * 2:
            return 0.0

        recent = self.verdict_accuracy[-window:]
        previous = self.verdict_accuracy[-2*window:-window]

        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)

        return recent_avg - previous_avg


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Adaptive Curriculum System
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AdaptiveCurriculum:
    """
    Self-improving curriculum that evolves case difficulty
    based on agent performance over time.

    The curriculum uses a state machine with automatic phase
    advancement triggered by performance thresholds:

    FOUNDATION → INTERMEDIATE → ADVANCED → EXPERT
      (accuracy>70%)   (accuracy>60%   (always at
                        + detect>50%)   episode 600+)

    Within each phase, the curriculum actively targets agent
    weaknesses by generating cases that exploit identified gaps.
    """

    # ── Phase advancement thresholds ──
    PHASE_THRESHOLDS = {
        CurriculumPhase.FOUNDATION: {
            "min_episodes": 50,
            "accuracy_threshold": 0.70,
            "advance_to": CurriculumPhase.INTERMEDIATE,
        },
        CurriculumPhase.INTERMEDIATE: {
            "min_episodes": 100,
            "accuracy_threshold": 0.60,
            "detection_threshold": 0.50,
            "advance_to": CurriculumPhase.ADVANCED,
        },
        CurriculumPhase.ADVANCED: {
            "min_episodes": 200,
            "accuracy_threshold": 0.55,
            "detection_threshold": 0.55,
            "fairness_threshold": 0.60,
            "advance_to": CurriculumPhase.EXPERT,
        },
        CurriculumPhase.EXPERT: {
            "min_episodes": 0,
            "advance_to": None,  # Terminal phase
        },
    }

    # ── Phase → case configuration ──
    PHASE_CONFIG = {
        CurriculumPhase.FOUNDATION: {
            "case_types": [CaseType.BNS_318_BAIL],
            "complexity": CaseComplexity.EASY,
            "max_rounds": 5,
            "deception_prob": 0.0,
            "fabrication_prob": 0.0,
            "description": "Simple BNS 318 bail cases with clear evidence",
        },
        CurriculumPhase.INTERMEDIATE: {
            "case_types": [CaseType.BNS_318_BAIL, CaseType.PMLA_BAIL],
            "complexity": CaseComplexity.MEDIUM,
            "max_rounds": 5,
            "deception_prob": 0.3,
            "fabrication_prob": 0.1,
            "description": "Mixed PMLA/BNS cases with occasional deception",
        },
        CurriculumPhase.ADVANCED: {
            "case_types": [CaseType.PMLA_BAIL, CaseType.BNS_111_ORGANISED_CRIME],
            "complexity": CaseComplexity.HARD,
            "max_rounds": 5,
            "deception_prob": 0.5,
            "fabrication_prob": 0.25,
            "description": "Complex PMLA/Organised Crime with fabricated evidence",
        },
        CurriculumPhase.EXPERT: {
            "case_types": list(CaseType),
            "complexity": CaseComplexity.HARD,
            "max_rounds": 4,
            "deception_prob": 0.6,
            "fabrication_prob": 0.35,
            "description": "Maximum complexity — all adversarial",
        },
    }

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the adaptive curriculum.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.phase = CurriculumPhase.FOUNDATION
        self.episode_count = 0
        self.phase_episode_count = 0
        self.tracker = AgentPerformanceTracker()
        self.case_generator = CaseGenerator(seed=seed)
        self._rng = random.Random(seed)

        # ── Phase transition history ──
        self.phase_history: List[Dict[str, Any]] = [{
            "phase": self.phase.value,
            "episode": 0,
            "reason": "initialization",
        }]

    def after_episode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process results after an episode completes.

        Updates performance tracking, checks phase advancement
        conditions, and returns curriculum status.

        Args:
            result: Episode result dictionary (see AgentPerformanceTracker.record_episode).

        Returns:
            Curriculum status update with phase info and recommendations.
        """
        self.episode_count += 1
        self.phase_episode_count += 1
        self.tracker.record_episode(result)

        # ── Check phase advancement ──
        advanced = self._check_phase_advancement()

        status = {
            "episode": self.episode_count,
            "phase": self.phase.value,
            "phase_episodes": self.phase_episode_count,
            "advanced": advanced,
            "avg_accuracy": round(self.tracker.avg_accuracy, 3),
            "avg_detection": round(self.tracker.avg_deception_detection, 3),
            "avg_fairness": round(self.tracker.avg_fairness, 3),
            "avg_reward": round(self.tracker.avg_reward, 3),
            "improvement_trend": round(self.tracker.get_improvement_trend(), 4),
            "primary_weakness": self.tracker.get_primary_weakness(),
        }

        return status

    def generate_next_case(self) -> CourtCase:
        """
        Generate the next case based on current curriculum phase
        and identified agent weaknesses.

        The case is tailored to challenge the agent system's
        weaknesses while staying within the phase's difficulty bounds.

        Returns:
            CourtCase configured for the current curriculum state.
        """
        config = self.PHASE_CONFIG[self.phase]
        weakness = self.tracker.get_primary_weakness()

        # ── Select case type (prefer types that target weaknesses) ──
        case_type = self._select_case_type(config["case_types"], weakness)
        complexity = config["complexity"]

        # ── Difficulty modulation within phase ──
        # If agents are struggling, ease up slightly
        if self.tracker.avg_accuracy < 0.3 and self.phase_episode_count > 20:
            if complexity == CaseComplexity.HARD:
                complexity = CaseComplexity.MEDIUM
            elif complexity == CaseComplexity.MEDIUM:
                complexity = CaseComplexity.EASY

        # ── Generate case ──
        case = self.case_generator.generate(
            case_type=case_type,
            complexity=complexity,
        )

        return case

    def get_max_rounds(self) -> int:
        """Get the maximum trial rounds for the current phase."""
        return self.PHASE_CONFIG[self.phase]["max_rounds"]

    def get_status_summary(self) -> str:
        """Get a human-readable curriculum status summary."""
        config = self.PHASE_CONFIG[self.phase]
        weakness = self.tracker.get_primary_weakness()

        lines = [
            f"Phase: {self.phase.value.upper()} — {config['description']}",
            f"Episode: {self.episode_count} (phase: {self.phase_episode_count})",
            f"Accuracy: {self.tracker.avg_accuracy:.1%}",
            f"Deception Detection: {self.tracker.avg_deception_detection:.1%}",
            f"Fairness: {self.tracker.avg_fairness:.1%}",
            f"Improvement Trend: {self.tracker.get_improvement_trend():+.3f}",
        ]
        if weakness:
            lines.append(f"Primary Weakness: {weakness}")

        return " | ".join(lines)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Private Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _check_phase_advancement(self) -> bool:
        """Check if conditions are met to advance to the next phase."""
        thresholds = self.PHASE_THRESHOLDS.get(self.phase)
        if not thresholds or thresholds.get("advance_to") is None:
            return False

        if self.phase_episode_count < thresholds.get("min_episodes", 0):
            return False

        # Check accuracy threshold
        if self.tracker.avg_accuracy < thresholds.get("accuracy_threshold", 0.0):
            return False

        # Check detection threshold (if applicable)
        if "detection_threshold" in thresholds:
            if self.tracker.avg_deception_detection < thresholds["detection_threshold"]:
                return False

        # Check fairness threshold (if applicable)
        if "fairness_threshold" in thresholds:
            if self.tracker.avg_fairness < thresholds["fairness_threshold"]:
                return False

        # ── Advance! ──
        old_phase = self.phase
        self.phase = thresholds["advance_to"]
        self.phase_episode_count = 0

        self.phase_history.append({
            "phase": self.phase.value,
            "episode": self.episode_count,
            "reason": f"advanced_from_{old_phase.value}",
            "accuracy": round(self.tracker.avg_accuracy, 3),
            "detection": round(self.tracker.avg_deception_detection, 3),
        })

        return True

    def _select_case_type(
        self, available_types: List[CaseType], weakness: Optional[str]
    ) -> CaseType:
        """
        Select a case type, biasing toward types that target weaknesses.

        Args:
            available_types: Case types available in this phase.
            weakness: Primary weakness (if any).

        Returns:
            Selected CaseType.
        """
        if weakness is None:
            return self._rng.choice(available_types)

        # Map weaknesses to preferred case types
        weakness_case_map = {
            "false_guilty": CaseType.BNS_318_BAIL,             # Needs better evidence evaluation
            "false_acquittal": CaseType.PMLA_BAIL,             # Needs stronger prosecution
            "missed_deception": CaseType.BNS_111_ORGANISED_CRIME,  # Complex deception
            "evidence_misread": CaseType.UAPA_43D_BAIL,        # High-stakes evidence analysis
        }

        preferred = weakness_case_map.get(weakness)
        if preferred and preferred in available_types:
            # 70% chance to use weakness-targeted case
            if self._rng.random() < 0.7:
                return preferred

        return self._rng.choice(available_types)
