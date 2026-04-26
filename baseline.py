"""
Nyaya-Env — Baseline Agents
==============================
Heuristic-based baseline agents for benchmarking the bail hearing
environment. Each agent follows a simple rule-based strategy.

Author: jaisogani-ai
"""

import random
import sys
import time
from typing import Dict, Any, List, Tuple

from environment import (
    CourtRoomEnv, CourtAction, CourtObservation,
    ALL_AGENTS, AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE,
    AGENT_CLERK, AGENT_EXPERT,
    JudgeAction, ProsecutorAction, DefenseAction, ExpertAction,
)
from grader import grade_episode


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline Agent Strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class BaselineJudge:
    """
    Heuristic judge that systematically assesses BNSS factors
    before making a bail decision based on prosecution vs defense scores.
    """
    def act(self, obs: CourtObservation) -> int:
        # First rounds: assess factors
        if obs.factors_assessed_count < 2 and obs.hearing_round <= 2:
            return JudgeAction.ASSESS_FLIGHT_RISK
        if obs.factors_assessed_count < 4 and obs.hearing_round <= 4:
            return JudgeAction.ASSESS_GRAVITY
        if obs.oversight_budget > 0 and obs.hearing_round <= 5:
            return JudgeAction.ASK_CLARIFICATION

        # Later rounds: impose conditions or decide
        if obs.hearing_round >= 5:
            if obs.prosecution_score > obs.defense_score + 0.1:
                return JudgeAction.DENY_BAIL
            else:
                return JudgeAction.GRANT_BAIL

        return JudgeAction.IMPOSE_CONDITION


class BaselineProsecutor:
    """
    Heuristic prosecutor that presents evidence and argues flight risk
    when the evidence supports it.
    """
    def act(self, obs: CourtObservation) -> int:
        if obs.hearing_round <= 1:
            return ProsecutorAction.PRESENT_EVIDENCE
        if obs.hearing_round == 2:
            return ProsecutorAction.CITE_BNS_SECTION
        if obs.flight_risk_score > 0.5 and obs.hearing_round <= 4:
            return ProsecutorAction.ARGUE_FLIGHT_RISK
        if obs.case_type == "pmla_bail":
            return ProsecutorAction.INVOKE_PMLA_TWIN_TEST
        if obs.hearing_round >= 4:
            return ProsecutorAction.CROSS_EXAMINE_EXPERT
        return ProsecutorAction.CITE_SC_PRECEDENT


class BaselineDefense:
    """
    Heuristic defense that argues Article 21 when delay is excessive,
    challenges evidence, and proposes bail conditions.
    """
    def act(self, obs: CourtObservation) -> int:
        if obs.article21_threshold_breached:
            return DefenseAction.INVOKE_ARTICLE_21
        if not obs.charge_sheet_filed and obs.days_since_arrest > 90:
            return DefenseAction.ARGUE_90_DAY_DEFAULT_BAIL
        if obs.case_type == "uapa_43d_bail" and obs.delay_duration_months > 24:
            return DefenseAction.CITE_NAJEEB_DELAY
        if obs.hearing_round <= 2:
            return DefenseAction.INVOKE_ARTICLE_21
        if obs.hearing_round <= 3:
            return DefenseAction.CITE_ANTIL_GUIDELINES
        if obs.case_type == "pmla_bail":
            return DefenseAction.CHALLENGE_PMLA_TWIN_TEST
        return DefenseAction.PROPOSE_BAIL_CONDITIONS


class BaselineExpert:
    """
    Heuristic expert witness that mostly testifies truthfully
    with occasional key fact reveals.
    """
    def act(self, obs: CourtObservation) -> int:
        if obs.hearing_round <= 3:
            return ExpertAction.TESTIFY_TRUTHFUL
        if obs.hearing_round == 4:
            return ExpertAction.REVEAL_KEY_FACT
        return ExpertAction.TESTIFY_TRUTHFUL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Baseline Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_baseline(
    num_episodes: int = 100,
    task: str = "medium",
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run baseline agents for benchmarking.

    Args:
        num_episodes: Number of episodes to run.
        task: Task difficulty.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        Dictionary with baseline results.
    """
    env = CourtRoomEnv(seed=seed)
    judge = BaselineJudge()
    prosecutor = BaselineProsecutor()
    defense = BaselineDefense()
    expert = BaselineExpert()

    scores = []
    passed_count = 0

    if verbose:
        print(f"Running baseline ({num_episodes} episodes, task={task})...")

    for ep in range(num_episodes):
        obs = env.reset(task=task)

        while not obs.done:
            action = CourtAction(
                judge=judge.act(obs),
                prosecutor=prosecutor.act(obs),
                defense=defense.act(obs),
                clerk=0,
                expert_witness=expert.act(obs),
            )
            obs = env.step(action)

        trajectory = env.get_trajectory()
        result = grade_episode(trajectory, task)
        scores.append(result["score"])
        if result["passed"]:
            passed_count += 1

    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_rate = passed_count / num_episodes if num_episodes > 0 else 0.0

    if verbose:
        print(f"  Avg Score: {avg_score:.4f}")
        print(f"  Pass Rate: {pass_rate:.1%} ({passed_count}/{num_episodes})")

    return {
        "task": task,
        "num_episodes": num_episodes,
        "avg_score": round(avg_score, 4),
        "pass_rate": round(pass_rate, 4),
        "passed": passed_count,
    }


def main():
    """Run baseline across all task difficulties."""
    print("=" * 55)
    print("  NYAYA-ENV — BASELINE BENCHMARK")
    print("=" * 55)
    print()

    for task in ["easy", "medium", "hard"]:
        result = run_baseline(num_episodes=100, task=task)
        print()

    print("=" * 55)


if __name__ == "__main__":
    main()
