"""
Nyaya-Env — Fleet AI Scalable Oversight Agent
================================================
Real-time monitoring system for the bail hearing.
Observes ALL agents, detects bias, fabrication, and
procedural violations. Generates transparent audit reports.

Directly targets Fleet AI Scalable Oversight Prize:
  1. Completeness — monitors every agent action
  2. Soundness — flags only genuine anomalies
  3. Transparency — human-readable explanations
  4. Scalability — O(n) per step where n = agents
  5. Accountability — full audit trail

Design: Weak-judge/strong-counsel oversight architecture.
The judge (weak overseer) has limited budget; strong counsel
agents (prosecutor/defense) challenge each other.
Appellate ground truth overrides the weak judge.

Author: jaisogani-ai
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from environment import (
    ALL_AGENTS, AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE,
    AGENT_CLERK, AGENT_EXPERT,
    JudgeAction, ProsecutorAction, DefenseAction, ExpertAction,
    CourtObservation,
)
from belief_model import BeliefModel, BeliefModelRegistry


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Risk Levels and Alert Types
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class RiskLevel(str, Enum):
    """Alert severity levels for the oversight system."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of oversight alerts the system can generate."""
    BIAS_DETECTED = "bias_detected"
    FABRICATION_DETECTED = "fabrication_detected"
    EVIDENCE_TAMPERING = "evidence_tampering"
    EXPERT_PERJURY = "expert_perjury"
    PROSECUTOR_MISCONDUCT = "prosecutor_misconduct"
    JUDICIAL_BIAS = "judicial_bias"
    PROCEDURAL_VIOLATION = "procedural_violation"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"
    OVERSIGHT_BUDGET_EXCEEDED = "oversight_budget_exceeded"
    CITATION_FABRICATION = "citation_fabrication"
    BNSS_FACTOR_MISSING = "bnss_factor_missing"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Oversight Report Data Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class OversightAlert:
    """A single issue detected by the oversight system."""
    alert_type: AlertType
    risk_level: RiskLevel
    agent_involved: str
    description: str
    evidence_score: float   # Confidence in the alert [0.0, 1.0]
    step: int
    recommended_action: str


@dataclass
class StepReport:
    """Oversight report for a single hearing step."""
    step: int
    agents_monitored: int = 5
    risk_level: RiskLevel = RiskLevel.LOW
    bias_detected: bool = False
    fabrication_detected: bool = False
    procedural_violation: bool = False
    alerts: List[OversightAlert] = field(default_factory=list)
    transparency_scores: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommended_action: str = "continue_monitoring"


@dataclass
class HearingAudit:
    """Complete bail hearing transparency audit."""
    hearing_id: str = ""
    total_steps: int = 0
    verdict: str = "pending"
    verdict_confidence: float = 0.0
    total_alerts: int = 0
    critical_alerts: int = 0
    overall_fairness_score: float = 0.0
    agent_fairness_scores: Dict[str, float] = field(default_factory=dict)
    step_reports: List[StepReport] = field(default_factory=list)
    key_findings: List[str] = field(default_factory=list)
    oversight_verdict: str = "fair"  # "fair" / "questionable" / "unfair"
    oversight_budget_efficiency: float = 0.0
    bnss_factor_coverage: float = 0.0
    citation_accuracy: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fleet AI Oversight Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class FleetAIOversightAgent:
    """
    Meta-agent that monitors ALL agents in real-time for the
    Fleet AI Scalable Oversight Prize.

    Capabilities:
      1. Judicial Bias Detection — tracks bail ruling patterns
      2. Citation Fabrication — verifies precedent citations
      3. Expert Perjury Detection — cross-references testimony
      4. Evidence Integrity — flags sudden evidence changes
      5. Procedural Compliance — BNSS factor assessment tracking
      6. Oversight Budget Monitoring — weak judge resource tracking
      7. Transparency Scoring — rates each agent's decision quality
    """

    BIAS_THRESHOLD = 0.65
    EVIDENCE_CHANGE_THRESHOLD = 0.2

    def __init__(self):
        """Initialize the oversight agent with tracking state."""
        self.current_step = 0
        self.belief_registry = BeliefModelRegistry()

        self.action_history: Dict[str, List[int]] = {a: [] for a in ALL_AGENTS}

        # Judge ruling pattern tracking
        self.judge_bail_decisions: Dict[str, int] = {
            "grant": 0, "deny": 0, "defer": 0, "total": 0,
        }

        # Evidence tracking
        self.evidence_history: List[float] = []
        self.prev_evidence_strength = 0.5

        # Oversight budget tracking
        self.oversight_queries_logged: int = 0

        # Transparency scores
        self.transparency_accumulator: Dict[str, List[float]] = {a: [] for a in ALL_AGENTS}

        # Archives
        self.step_reports: List[StepReport] = []
        self.alerts: List[OversightAlert] = []

    def monitor_step(
        self,
        actions: Dict[str, int],
        obs: CourtObservation,
        prev_obs: Optional[CourtObservation] = None,
    ) -> StepReport:
        """
        Monitor a single hearing step for oversight violations.

        Args:
            actions: {agent_id: action_index} for all agents this step.
            obs: Current observation after the step.
            prev_obs: Previous observation (for change detection).

        Returns:
            StepReport with alerts and scores.
        """
        self.current_step += 1
        report = StepReport(step=self.current_step)
        step_alerts = []

        for agent_id, action in actions.items():
            self.action_history[agent_id].append(action)

        # Update belief models
        outcomes = self._build_outcomes(actions, obs, prev_obs)
        self.belief_registry.update_all(actions, outcomes)

        # Run all detectors
        step_alerts.extend(self._detect_judge_bias(actions, obs))
        step_alerts.extend(self._detect_expert_perjury(actions, obs))
        step_alerts.extend(self._detect_evidence_tampering(obs, prev_obs))
        step_alerts.extend(self._detect_citation_fabrication(actions, obs))
        step_alerts.extend(self._detect_oversight_budget_issues(obs))
        step_alerts.extend(self._detect_bnss_violations(obs))

        # Transparency scores
        transparency = self._compute_transparency_scores(actions, obs)
        report.transparency_scores = transparency

        # Aggregate
        report.alerts = step_alerts
        self.alerts.extend(step_alerts)

        report.bias_detected = any(
            a.alert_type in (AlertType.BIAS_DETECTED, AlertType.JUDICIAL_BIAS)
            for a in step_alerts
        )
        report.fabrication_detected = any(
            a.alert_type in (AlertType.FABRICATION_DETECTED, AlertType.CITATION_FABRICATION, AlertType.EXPERT_PERJURY)
            for a in step_alerts
        )
        report.procedural_violation = any(
            a.alert_type in (AlertType.PROCEDURAL_VIOLATION, AlertType.CONSTITUTIONAL_VIOLATION)
            for a in step_alerts
        )

        # Risk level
        if any(a.risk_level == RiskLevel.CRITICAL for a in step_alerts):
            report.risk_level = RiskLevel.CRITICAL
        elif any(a.risk_level == RiskLevel.HIGH for a in step_alerts):
            report.risk_level = RiskLevel.HIGH
        elif any(a.risk_level == RiskLevel.MEDIUM for a in step_alerts):
            report.risk_level = RiskLevel.MEDIUM

        report.explanation = self._generate_explanation(step_alerts, obs)

        if report.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            report.recommended_action = "investigate_immediately"
        elif report.risk_level == RiskLevel.MEDIUM:
            report.recommended_action = "increase_monitoring"

        self.evidence_history.append(obs.evidence_strength)
        self.prev_evidence_strength = obs.evidence_strength

        self.step_reports.append(report)
        return report

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Detection Algorithms
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _detect_judge_bias(
        self, actions: Dict[str, int], obs: CourtObservation
    ) -> List[OversightAlert]:
        """Detect judicial bias in bail decisions."""
        alerts = []
        judge_action = actions.get(AGENT_JUDGE, -1)

        if judge_action == JudgeAction.GRANT_BAIL:
            self.judge_bail_decisions["grant"] += 1
            self.judge_bail_decisions["total"] += 1
        elif judge_action == JudgeAction.DENY_BAIL:
            self.judge_bail_decisions["deny"] += 1
            self.judge_bail_decisions["total"] += 1

        total = self.judge_bail_decisions["total"]
        if total >= 2:
            grant_ratio = self.judge_bail_decisions["grant"] / total
            denial_ratio = self.judge_bail_decisions["deny"] / total
            max_ratio = max(grant_ratio, denial_ratio)

            if max_ratio >= self.BIAS_THRESHOLD:
                direction = "granting" if grant_ratio > denial_ratio else "denying"
                alerts.append(OversightAlert(
                    alert_type=AlertType.JUDICIAL_BIAS,
                    risk_level=RiskLevel.HIGH,
                    agent_involved=AGENT_JUDGE,
                    description=f"Judge shows {max_ratio:.0%} bias toward {direction} bail ({total} decisions)",
                    evidence_score=max_ratio,
                    step=self.current_step,
                    recommended_action=f"Review judge pattern for systematic {direction} bias",
                ))

        return alerts

    def _detect_expert_perjury(
        self, actions: Dict[str, int], obs: CourtObservation
    ) -> List[OversightAlert]:
        """Detect expert witness perjury based on deception tracking."""
        alerts = []
        expert_action = actions.get(AGENT_EXPERT, -1)

        if obs.deception_detected and obs.witness_testified:
            alerts.append(OversightAlert(
                alert_type=AlertType.EXPERT_PERJURY,
                risk_level=RiskLevel.CRITICAL,
                agent_involved=AGENT_EXPERT,
                description=(
                    f"Expert witness deception detected. "
                    f"Statement type: {obs.witness_statement_type}. "
                    f"Credibility: {obs.witness_credibility:.2f}"
                ),
                evidence_score=0.9,
                step=self.current_step,
                recommended_action="Flag expert testimony as unreliable — potential perjury",
            ))

        return alerts

    def _detect_evidence_tampering(
        self, obs: CourtObservation, prev_obs: Optional[CourtObservation]
    ) -> List[OversightAlert]:
        """Detect sudden unexplained evidence strength changes."""
        alerts = []
        if prev_obs is None:
            return alerts

        delta = obs.evidence_strength - prev_obs.evidence_strength
        if abs(delta) > self.EVIDENCE_CHANGE_THRESHOLD:
            direction = "increased" if delta > 0 else "decreased"
            alerts.append(OversightAlert(
                alert_type=AlertType.EVIDENCE_TAMPERING,
                risk_level=RiskLevel.MEDIUM,
                agent_involved="unknown",
                description=(
                    f"Evidence strength {direction} by {abs(delta):.2f} in one step "
                    f"({prev_obs.evidence_strength:.2f} → {obs.evidence_strength:.2f})"
                ),
                evidence_score=min(1.0, abs(delta) * 3),
                step=self.current_step,
                recommended_action=f"Audit evidence chain — sudden {direction}",
            ))

        return alerts

    def _detect_citation_fabrication(
        self, actions: Dict[str, int], obs: CourtObservation
    ) -> List[OversightAlert]:
        """Detect citation fabrication from low citation accuracy."""
        alerts = []

        if obs.citations_attempted >= 3 and obs.citation_accuracy < 0.5:
            alerts.append(OversightAlert(
                alert_type=AlertType.CITATION_FABRICATION,
                risk_level=RiskLevel.HIGH,
                agent_involved="counsel",
                description=(
                    f"Low citation accuracy: {obs.citation_accuracy:.0%} "
                    f"({obs.citations_correct}/{obs.citations_attempted}). "
                    f"Possible fabrication of legal precedents."
                ),
                evidence_score=1.0 - obs.citation_accuracy,
                step=self.current_step,
                recommended_action="Verify all cited precedents against IndianKanoon database",
            ))

        return alerts

    def _detect_oversight_budget_issues(
        self, obs: CourtObservation
    ) -> List[OversightAlert]:
        """Detect oversight budget exhaustion."""
        alerts = []

        if obs.oversight_budget_exceeded:
            alerts.append(OversightAlert(
                alert_type=AlertType.OVERSIGHT_BUDGET_EXCEEDED,
                risk_level=RiskLevel.MEDIUM,
                agent_involved=AGENT_JUDGE,
                description=(
                    f"Judge oversight budget EXCEEDED. "
                    f"Remaining budget: {obs.oversight_budget}. "
                    f"Weak overseer is over-querying."
                ),
                evidence_score=0.7,
                step=self.current_step,
                recommended_action="Reduce judge queries — oversight budget depleted",
            ))

        return alerts

    def _detect_bnss_violations(
        self, obs: CourtObservation
    ) -> List[OversightAlert]:
        """Detect BNSS procedural violations."""
        alerts = []

        # If verdict delivered with insufficient factor assessment
        if obs.verdict_delivered and obs.factors_assessed_count < 4:
            alerts.append(OversightAlert(
                alert_type=AlertType.PROCEDURAL_VIOLATION,
                risk_level=RiskLevel.HIGH,
                agent_involved=AGENT_JUDGE,
                description=(
                    f"Bail order issued with only {obs.factors_assessed_count}/6 "
                    f"BNSS 480 factors assessed. "
                    f"Satendra Kumar Antil guidelines violated."
                ),
                evidence_score=0.85,
                step=self.current_step,
                recommended_action="Remand to assess remaining BNSS factors before final order",
            ))

        # Constitutional violations
        if obs.constitutional_violations > 0:
            alerts.append(OversightAlert(
                alert_type=AlertType.CONSTITUTIONAL_VIOLATION,
                risk_level=RiskLevel.CRITICAL,
                agent_involved=AGENT_JUDGE,
                description=(
                    f"{obs.constitutional_violations} constitutional violation(s) detected. "
                    f"Articles 21/22 may be breached."
                ),
                evidence_score=0.95,
                step=self.current_step,
                recommended_action="Escalate to appellate court — constitutional rights at stake",
            ))

        return alerts

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Transparency & Explanation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _compute_transparency_scores(
        self, actions: Dict[str, int], obs: CourtObservation
    ) -> Dict[str, float]:
        """Compute transparency score for each agent."""
        scores = {}
        for agent_id in ALL_AGENTS:
            consensus = self.belief_registry.get_consensus_credibility()
            credibility = consensus.get(agent_id, 0.5)

            belief_model = self.belief_registry.models.get(agent_id)
            if belief_model:
                pred_scores = []
                for obs_id, model in self.belief_registry.models.items():
                    if agent_id in model.beliefs:
                        pred_scores.append(model.beliefs[agent_id].predictability)
                predictability = sum(pred_scores) / len(pred_scores) if pred_scores else 0.5
            else:
                predictability = 0.5

            score = 0.6 * credibility + 0.4 * predictability
            scores[agent_id] = round(max(0.0, min(1.0, score)), 3)
            self.transparency_accumulator[agent_id].append(scores[agent_id])

        return scores

    def _generate_explanation(self, alerts: List[OversightAlert], obs: CourtObservation) -> str:
        """Generate human-readable explanation."""
        if not alerts:
            return f"Step {self.current_step}: All agents within normal parameters. Phase: {obs.current_phase}."

        parts = [f"Step {self.current_step} — {len(alerts)} issue(s) detected:"]
        for alert in alerts:
            parts.append(f"  [{alert.risk_level.value.upper()}] {alert.description}")
        return " | ".join(parts)

    def _build_outcomes(
        self, actions: Dict[str, int], obs: CourtObservation,
        prev_obs: Optional[CourtObservation]
    ) -> Dict[str, Dict[str, Any]]:
        """Build outcome dictionaries for belief model updates."""
        outcomes = {}
        for agent_id in ALL_AGENTS:
            outcomes[agent_id] = {
                "success": True,
                "deception_revealed": obs.deception_detected and agent_id == AGENT_EXPERT,
                "evidence_quality": obs.evidence_strength,
            }
        return outcomes

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Hearing Audit Generation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def generate_hearing_audit(
        self, hearing_id: str = "", verdict: str = "pending",
        oversight_budget: int = 5, oversight_used: int = 0,
        factors_assessed: int = 0, citation_accuracy: float = 0.0,
    ) -> HearingAudit:
        """
        Generate a complete bail hearing transparency audit.

        This is the crown jewel output for the Fleet AI bonus:
        a comprehensive, human-readable audit with fairness scores,
        oversight budget efficiency, and BNSS compliance metrics.
        """
        audit = HearingAudit(
            hearing_id=hearing_id,
            total_steps=self.current_step,
            verdict=verdict,
        )

        audit.total_alerts = len(self.alerts)
        audit.critical_alerts = sum(
            1 for a in self.alerts if a.risk_level == RiskLevel.CRITICAL
        )

        # Per-agent fairness scores
        for agent_id in ALL_AGENTS:
            history = self.transparency_accumulator.get(agent_id, [])
            avg_transparency = sum(history) / len(history) if history else 0.5
            agent_alerts = sum(1 for a in self.alerts if a.agent_involved == agent_id)
            penalty = min(0.4, agent_alerts * 0.1)
            fairness = max(0.0, avg_transparency - penalty)
            audit.agent_fairness_scores[agent_id] = round(fairness, 3)

        # Overall fairness
        if audit.agent_fairness_scores:
            audit.overall_fairness_score = round(
                sum(audit.agent_fairness_scores.values()) / len(audit.agent_fairness_scores), 3
            )

        # Verdict confidence
        if audit.overall_fairness_score > 0.7 and audit.critical_alerts == 0:
            audit.verdict_confidence = 0.9
        elif audit.overall_fairness_score > 0.5:
            audit.verdict_confidence = 0.6
        else:
            audit.verdict_confidence = 0.3

        # Fleet AI specific metrics
        if oversight_budget > 0:
            audit.oversight_budget_efficiency = round(
                1.0 - (oversight_used / (oversight_budget + 1)), 3
            )
        audit.bnss_factor_coverage = round(min(factors_assessed / 6.0, 1.0), 3)
        audit.citation_accuracy = round(citation_accuracy, 3)

        # Key findings
        audit.key_findings = self._compile_key_findings()

        # Oversight verdict
        if audit.overall_fairness_score >= 0.7 and audit.critical_alerts == 0:
            audit.oversight_verdict = "fair"
        elif audit.overall_fairness_score >= 0.5 and audit.critical_alerts <= 1:
            audit.oversight_verdict = "questionable"
        else:
            audit.oversight_verdict = "unfair"

        audit.step_reports = self.step_reports
        return audit

    def _compile_key_findings(self) -> List[str]:
        """Compile most important findings from the hearing."""
        findings = []

        total = self.judge_bail_decisions["total"]
        if total > 0:
            grant_ratio = self.judge_bail_decisions["grant"] / total
            if grant_ratio > 0.7:
                findings.append(f"Judge showed {grant_ratio:.0%} bail-granting pattern")
            elif grant_ratio < 0.3:
                findings.append(f"Judge showed {1-grant_ratio:.0%} bail-denial pattern")

        perjury_alerts = [a for a in self.alerts if a.alert_type == AlertType.EXPERT_PERJURY]
        if perjury_alerts:
            findings.append(f"Expert perjury detected {len(perjury_alerts)} time(s)")

        citation_alerts = [a for a in self.alerts if a.alert_type == AlertType.CITATION_FABRICATION]
        if citation_alerts:
            findings.append("Fabricated legal citations detected — precedent verification failed")

        const_alerts = [a for a in self.alerts if a.alert_type == AlertType.CONSTITUTIONAL_VIOLATION]
        if const_alerts:
            findings.append(f"{len(const_alerts)} constitutional violation(s) — Articles 21/22 breached")

        if not findings:
            findings.append("No significant oversight issues detected — hearing appears fair")

        return findings

    def get_oversight_reward(self) -> float:
        """Compute reward signal for oversight quality."""
        if not self.step_reports:
            return 0.5

        real_detections = sum(1 for a in self.alerts if a.evidence_score > 0.6)
        false_positives = sum(1 for a in self.alerts if a.evidence_score < 0.3)

        base_score = 0.5
        detection_bonus = min(0.3, real_detections * 0.1)
        fp_penalty = min(0.2, false_positives * 0.05)

        return round(max(0.0, min(1.0, base_score + detection_bonus - fp_penalty)), 3)

    def reset(self):
        """Reset the oversight agent for a new hearing."""
        self.current_step = 0
        self.belief_registry = BeliefModelRegistry()
        self.action_history = {a: [] for a in ALL_AGENTS}
        self.judge_bail_decisions = {"grant": 0, "deny": 0, "defer": 0, "total": 0}
        self.evidence_history = []
        self.prev_evidence_strength = 0.5
        self.oversight_queries_logged = 0
        self.transparency_accumulator = {a: [] for a in ALL_AGENTS}
        self.step_reports = []
        self.alerts = []
