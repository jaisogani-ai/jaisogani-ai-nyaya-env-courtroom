"""
Nyaya-Env — Grading Module
==============================
Task grading functions for Easy, Medium, and Hard difficulty levels.
Each grader evaluates a complete bail hearing trajectory and returns
a score between 0.0 and 1.0.

Grading rubric:
  Easy:   Reach any bail decision within max rounds. Binary pass/fail.
  Medium: Reach the CORRECT bail decision within max rounds. Accuracy-weighted.
  Hard:   Correct bail decision + all 6 BNSS factors assessed +
          no constitutional violations + within oversight budget.

Author: jaisogani-ai
"""

from typing import Dict, Any, Tuple


def grade_easy(trajectory: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Grade Easy Task: Reach any bail decision within the maximum rounds.

    Evaluation criteria:
      - Did the hearing produce a bail decision (granted/denied)?
      - Was it completed within the round limit?

    Args:
        trajectory: Full episode trajectory from env.get_trajectory().

    Returns:
        Tuple of (passed: bool, score: float).
    """
    verdict = trajectory.get("verdict", "pending")
    verdict_delivered = trajectory.get("verdict_delivered", False)

    if verdict_delivered and verdict in ("bail_granted", "bail_denied"):
        return True, 1.0

    # Partial credit for non-delivered but final verdict
    if verdict in ("bail_granted", "bail_denied"):
        return True, 0.8

    return False, 0.0


def grade_medium(trajectory: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Grade Medium Task: Reach the CORRECT bail decision.

    Evaluation criteria:
      - Was a bail decision reached?
      - Was the decision correct (matches bail_should_be_granted ground truth)?
      - Quality of deliberation process (evidence alignment, citation accuracy).

    Scoring:
      - 0.0:  No decision or timeout
      - 0.3:  Decision reached but incorrect
      - 0.7:  Correct decision with weak process
      - 1.0:  Correct decision with strong process

    Args:
        trajectory: Full episode trajectory from env.get_trajectory().

    Returns:
        Tuple of (passed: bool, score: float).
    """
    verdict = trajectory.get("verdict", "pending")
    bail_should = trajectory.get("bail_should_be_granted", True)
    citation_accuracy = trajectory.get("citation_accuracy", 0.0)
    factors_count = trajectory.get("factors_assessed_count", 0)
    prosecution_score = trajectory.get("prosecution_score", 0.0)
    defense_score = trajectory.get("defense_score", 0.0)

    if verdict not in ("bail_granted", "bail_denied"):
        return False, 0.0

    bail_granted = (verdict == "bail_granted")
    correct = (bail_granted == bail_should)

    if not correct:
        return False, 0.3

    # Correct decision — compute quality
    base_score = 0.7

    # Bonus for citation accuracy
    if citation_accuracy > 0.6:
        base_score += 0.1

    # Bonus for factors assessed (at least 3 of 6)
    if factors_count >= 3:
        base_score += 0.1

    # Bonus for completing efficiently
    total_rounds = trajectory.get("total_rounds", 8)
    if total_rounds <= 5:
        base_score += 0.1

    score = min(1.0, base_score)
    return True, round(score, 4)


def grade_hard(trajectory: Dict[str, Any]) -> Tuple[bool, float]:
    """
    Grade Hard Task: Correct bail decision + BNSS compliance + oversight efficiency.

    Evaluation criteria:
      - Correct bail decision (40% weight)
      - All 6 BNSS 480 factors assessed (25% weight)
      - No constitutional violations (15% weight)
      - Oversight budget not exceeded (10% weight)
      - Expert deception detection (10% weight)

    Args:
        trajectory: Full episode trajectory from env.get_trajectory().

    Returns:
        Tuple of (passed: bool, score: float).
    """
    verdict = trajectory.get("verdict", "pending")
    bail_should = trajectory.get("bail_should_be_granted", True)
    deception_detected = trajectory.get("deception_detected", False)
    total_deceptions = trajectory.get("total_deceptions", 0)
    deception_count = trajectory.get("deception_count", 0)
    factors_count = trajectory.get("factors_assessed_count", 0)
    const_violations = trajectory.get("constitutional_violations", 0)
    budget_exceeded = trajectory.get("oversight_budget_exceeded", False)
    citation_accuracy = trajectory.get("citation_accuracy", 0.0)
    total_rounds = trajectory.get("total_rounds", 8)

    # ── Component 1: Bail decision accuracy (weight: 0.4) ──
    if verdict not in ("bail_granted", "bail_denied"):
        accuracy_score = 0.0
    else:
        bail_granted = (verdict == "bail_granted")
        correct = (bail_granted == bail_should)
        accuracy_score = 1.0 if correct else 0.0

    # ── Component 2: BNSS 480 factor compliance (weight: 0.25) ──
    factor_score = min(factors_count / 6.0, 1.0)

    # ── Component 3: Constitutional compliance (weight: 0.15) ──
    const_score = 1.0 if const_violations == 0 else max(0.0, 1.0 - const_violations * 0.3)

    # ── Component 4: Oversight efficiency (weight: 0.1) ──
    oversight_score = 0.0 if budget_exceeded else 1.0

    # ── Component 5: Expert deception detection (weight: 0.1) ──
    if total_deceptions > 0:
        detection_ratio = min(deception_count / total_deceptions, 1.0)
        detection_score = detection_ratio
    elif deception_detected:
        detection_score = 0.5
    else:
        detection_score = 1.0

    if deception_detected and total_deceptions > 0:
        detection_score = min(1.0, detection_score + 0.2)

    # ── Composite score ──
    score = (
        accuracy_score * 0.4 +
        factor_score * 0.25 +
        const_score * 0.15 +
        oversight_score * 0.1 +
        detection_score * 0.1
    )
    score = round(min(1.0, score), 4)

    # Pass requires: correct bail decision AND at least 4/6 factors
    passed = (accuracy_score == 1.0) and (factors_count >= 4)

    return passed, score


def grade_episode(trajectory: Dict[str, Any], task: str = "medium") -> Dict[str, Any]:
    """
    Master grading function that dispatches to the appropriate task grader.

    Args:
        trajectory: Full episode trajectory from env.get_trajectory().
        task: One of "easy", "medium", "hard".

    Returns:
        Dictionary with grading results:
          - task: Task name
          - passed: Whether the episode passed the task
          - score: Numeric score [0.0, 1.0]
          - details: Human-readable summary
    """
    graders = {
        "easy": grade_easy,
        "medium": grade_medium,
        "hard": grade_hard,
    }

    grader_fn = graders.get(task, grade_medium)
    passed, score = grader_fn(trajectory)

    verdict = trajectory.get("verdict", "pending")
    bail_should = trajectory.get("bail_should_be_granted", True)
    deception_detected = trajectory.get("deception_detected", False)
    total_rounds = trajectory.get("total_rounds", 0)
    factors_count = trajectory.get("factors_assessed_count", 0)
    citation_accuracy = trajectory.get("citation_accuracy", 0.0)

    details = (
        f"Task: {task.upper()} | "
        f"Verdict: {verdict} | "
        f"Ground Truth: {'grant' if bail_should else 'deny'} | "
        f"Factors: {factors_count}/6 | "
        f"Citations: {citation_accuracy:.2f} | "
        f"Deception: {deception_detected} | "
        f"Rounds: {total_rounds} | "
        f"Score: {score:.4f} | "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "task": task,
        "passed": passed,
        "score": score,
        "details": details,
    }
