"""
Nyaya-Env — Training Pipeline
================================
Multi-agent tabular Q-learning training loop with:
  - 4 verifiable metric curves (20% judging rubric)
  - Fleet AI oversight tracking
  - Snorkel expert truthfulness tracking
  - Adaptive curriculum integration
  - 4-plot visualization saved to training_results.png

Metrics tracked:
  1. citation_accuracy     — correct vs fabricated citations
  2. statute_f1            — BNSS factor assessment coverage
  3. expert_truthfulness   — delta between testimony and ground truth
  4. oversight_efficiency  — judge decisions per oversight budget turn

Author: jaisogani-ai
"""

import random
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional

from environment import (
    CourtRoomEnv, CourtAction, CourtObservation,
    ALL_AGENTS, LEARNING_AGENTS,
    AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE,
    AGENT_CLERK, AGENT_EXPERT,
    JudgeAction, ProsecutorAction, DefenseAction, ExpertAction,
)
from grader import grade_episode
from realistic_cases import CaseGenerator, CaseType


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Tabular Q-Learning Agent
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TabularAgent:
    """
    Multi-agent tabular Q-learning with epsilon-greedy exploration.
    Each agent has its own Q-table indexed by discretized state keys.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        lr: float = 0.01,
        gamma: float = 0.99,
        seed: int = 42,
    ):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.gamma = gamma
        self._rng = random.Random(seed)

        # Action space sizes for each learning agent
        self.action_sizes = {
            AGENT_JUDGE: 7,       # 0-6
            AGENT_PROSECUTOR: 7,  # 0-6 (v3.0: separate BNS/BNSS/SC cite)
            AGENT_DEFENSE: 7,     # 0-6
            AGENT_EXPERT: 4,      # 0-3
        }

        # Q-tables: {agent: {state_key: {action: value}}}
        self.q_tables: Dict[str, Dict[str, Dict[int, float]]] = {
            a: {} for a in LEARNING_AGENTS
        }

    def _state_key(self, obs: CourtObservation, agent: str) -> str:
        """Discretize observation into a hashable state key."""
        ev = min(int(obs.evidence_strength * 5), 4)
        fr = min(int(obs.flight_risk_score * 5), 4)
        ps = min(int(obs.prosecution_score * 5), 4)
        ds = min(int(obs.defense_score * 5), 4)
        ph = self._phase_idx(obs.current_phase)
        budget = min(obs.oversight_budget, 7)
        factors = min(obs.factors_assessed_count, 6)

        # Agent-specific state features
        if agent == AGENT_JUDGE:
            return f"J_{ev}_{fr}_{ps}_{ds}_{ph}_{budget}_{factors}"
        elif agent == AGENT_PROSECUTOR:
            return f"P_{ev}_{fr}_{ps}_{ds}_{ph}"
        elif agent == AGENT_DEFENSE:
            art21 = 1 if obs.article21_threshold_breached else 0
            return f"D_{ev}_{fr}_{ps}_{ds}_{ph}_{art21}"
        else:  # expert
            return f"E_{ev}_{ph}"

    @staticmethod
    def _phase_idx(phase: str) -> int:
        phases = {
            "filing": 0, "prosecution_args": 1, "expert_examination": 2,
            "defense_args": 3, "cross_examination": 4, "conditions": 5,
            "final_arguments": 6, "bail_order": 7,
        }
        return phases.get(phase, 0)

    def act(self, obs: CourtObservation, agent: str) -> int:
        """Select action using epsilon-greedy policy."""
        if agent not in self.action_sizes:
            return 0  # Clerk always 0

        if self._rng.random() < self.epsilon:
            return self._rng.randint(0, self.action_sizes[agent] - 1)

        key = self._state_key(obs, agent)
        if key not in self.q_tables[agent]:
            self.q_tables[agent][key] = {
                a: 0.0 for a in range(self.action_sizes[agent])
            }
        return max(self.q_tables[agent][key], key=self.q_tables[agent][key].get)

    def update(
        self, agent: str, obs: CourtObservation,
        action: int, reward: float,
        next_obs: CourtObservation, done: bool
    ):
        """Tabular Q-learning update."""
        if agent not in self.action_sizes:
            return

        key = self._state_key(obs, agent)
        if key not in self.q_tables[agent]:
            self.q_tables[agent][key] = {
                a: 0.0 for a in range(self.action_sizes[agent])
            }

        next_key = self._state_key(next_obs, agent)
        if next_key not in self.q_tables[agent]:
            self.q_tables[agent][next_key] = {
                a: 0.0 for a in range(self.action_sizes[agent])
            }

        best_next = max(self.q_tables[agent][next_key].values()) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_tables[agent][key].get(action, 0.0)
        self.q_tables[agent][key][action] = (
            self.q_tables[agent][key].get(action, 0.0) + self.lr * td_error
        )

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Training Loop
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(
    num_episodes: int = 1000,
    seed: int = 42,
    save_plots: bool = True,
    save_json: bool = True,
    print_every: int = 50,
) -> Dict[str, Any]:
    """
    Train multi-agent bail hearing system.

    Tracks and visualizes 4 verifiable metrics:
      1. citation_accuracy
      2. statute_f1 (BNSS factor coverage)
      3. expert_truthfulness
      4. oversight_efficiency

    Args:
        num_episodes: Number of training episodes.
        seed: Random seed for reproducibility.
        save_plots: Whether to save training_results.png.
        save_json: Whether to save training_results.json.
        print_every: Print progress every N episodes.

    Returns:
        Dictionary with all training results.
    """
    env = CourtRoomEnv(seed=seed)
    agent = TabularAgent(seed=seed)

    # ── Tracking arrays ──
    rewards_per_episode = []
    citation_accuracy_history = []
    statute_f1_history = []
    expert_truthfulness_history = []
    oversight_efficiency_history = []
    bail_decision_accuracy = []

    print("=" * 65)
    print("  🏛️  NYAYA-ENV — TRAINING PIPELINE")
    print("  Scalable-Oversight Gym for India's 54M Case Backlog")
    print("=" * 65)
    print(f"  Episodes: {num_episodes}")
    print(f"  Agents: Judge (weak overseer), Prosecutor, Defense, Clerk (deterministic), Expert (SME)")
    print(f"  Case types: PMLA, BNS 318, UAPA 43D, BNS 111")
    print()

    start_time = time.time()

    for episode in range(num_episodes):
        # ── Reset environment ──
        task = random.choice(["easy", "medium", "hard"])
        obs = env.reset(task=task)
        ep_reward = 0.0
        steps = 0

        while not obs.done and steps < 20:
            steps += 1

            # ── Select actions ──
            actions = CourtAction(
                judge=agent.act(obs, AGENT_JUDGE),
                prosecutor=agent.act(obs, AGENT_PROSECUTOR),
                defense=agent.act(obs, AGENT_DEFENSE),
                clerk=0,  # Always deterministic
                expert_witness=agent.act(obs, AGENT_EXPERT),
            )

            next_obs = env.step(actions)

            # ── Update Q-tables for learning agents ──
            action_map = {
                AGENT_JUDGE: actions.judge,
                AGENT_PROSECUTOR: actions.prosecutor,
                AGENT_DEFENSE: actions.defense,
                AGENT_EXPERT: actions.expert_witness,
            }

            for a_name, a_val in action_map.items():
                r = next_obs.rewards.get(a_name, 0.0)
                agent.update(a_name, obs, a_val, r, next_obs, next_obs.done)
                ep_reward += r

            obs = next_obs

        agent.decay_epsilon()

        # ── Extract trajectory metrics ──
        trajectory = env.get_trajectory()

        # Metric 1: Citation accuracy
        cite_acc = trajectory.get("citation_accuracy", 0.0)
        citation_accuracy_history.append(cite_acc)

        # Metric 2: Statute F1 (BNSS factor coverage as ratio)
        factors = trajectory.get("factors_assessed_count", 0)
        stat_f1 = min(factors / 6.0, 1.0)
        statute_f1_history.append(stat_f1)

        # Metric 3: Expert truthfulness (1 - delta)
        delta = trajectory.get("expert_truthfulness_delta", 0.0)
        expert_truth = max(0.0, 1.0 - delta)
        expert_truthfulness_history.append(expert_truth)

        # Metric 4: Oversight efficiency
        budget = trajectory.get("oversight_budget", 5)
        used = trajectory.get("oversight_queries_used", 0)
        exceeded = trajectory.get("oversight_budget_exceeded", False)
        if budget > 0:
            efficiency = 1.0 - (used / (budget + 1)) if not exceeded else 0.0
        else:
            efficiency = 0.0
        oversight_efficiency_history.append(efficiency)

        # Bail decision accuracy
        verdict = trajectory.get("verdict", "pending")
        bail_should = trajectory.get("bail_should_be_granted", True)
        if verdict in ("bail_granted", "bail_denied"):
            correct = (verdict == "bail_granted") == bail_should
            bail_decision_accuracy.append(1.0 if correct else 0.0)
        else:
            bail_decision_accuracy.append(0.0)

        rewards_per_episode.append(ep_reward)

        # ── Print progress ──
        if (episode + 1) % print_every == 0:
            w = min(print_every, episode + 1)
            acc = sum(bail_decision_accuracy[-w:]) / w
            cite = sum(citation_accuracy_history[-w:]) / w
            stat = sum(statute_f1_history[-w:]) / w
            exp_t = sum(expert_truthfulness_history[-w:]) / w
            ov_eff = sum(oversight_efficiency_history[-w:]) / w
            rew = sum(rewards_per_episode[-w:]) / w

            print(
                f"  Ep {episode+1:5d}/{num_episodes} | "
                f"BailAcc: {acc:.1%} | "
                f"Cite: {cite:.2f} | "
                f"BNSS: {stat:.2f} | "
                f"Expert: {exp_t:.2f} | "
                f"Oversight: {ov_eff:.2f} | "
                f"Rew: {rew:+.1f} | "
                f"ε: {agent.epsilon:.3f}"
            )

    elapsed = time.time() - start_time

    # ── Final summary ──
    last_100 = min(100, num_episodes)
    final_acc = sum(bail_decision_accuracy[-last_100:]) / last_100
    final_cite = sum(citation_accuracy_history[-last_100:]) / last_100
    final_stat = sum(statute_f1_history[-last_100:]) / last_100
    final_exp = sum(expert_truthfulness_history[-last_100:]) / last_100
    final_ov = sum(oversight_efficiency_history[-last_100:]) / last_100

    print()
    print("=" * 65)
    print("  🏛️  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Episodes:           {num_episodes}")
    print(f"  Time:               {elapsed:.1f}s")
    print(f"  Final Bail Accuracy: {final_acc:.1%}")
    print(f"  Final Citation Acc:  {final_cite:.3f}")
    print(f"  Final BNSS F1:       {final_stat:.3f}")
    print(f"  Final Expert Truth:  {final_exp:.3f}")
    print(f"  Final Oversight Eff: {final_ov:.3f}")
    print("=" * 65)

    results = {
        "num_episodes": num_episodes,
        "elapsed_seconds": round(elapsed, 2),
        "final_bail_accuracy": round(final_acc, 4),
        "final_citation_accuracy": round(final_cite, 4),
        "final_statute_f1": round(final_stat, 4),
        "final_expert_truthfulness": round(final_exp, 4),
        "final_oversight_efficiency": round(final_ov, 4),
        "rewards_per_episode": [round(r, 4) for r in rewards_per_episode],
        "citation_accuracy_history": [round(c, 4) for c in citation_accuracy_history],
        "statute_f1_history": [round(s, 4) for s in statute_f1_history],
        "expert_truthfulness_history": [round(e, 4) for e in expert_truthfulness_history],
        "oversight_efficiency_history": [round(o, 4) for o in oversight_efficiency_history],
        "bail_decision_accuracy": [round(a, 4) for a in bail_decision_accuracy],
    }

    # ── Save JSON ──
    if save_json:
        with open("training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("  Saved: training_results.json")

    # ── Save plots ──
    if save_plots:
        _plot_training_results(
            rewards_per_episode,
            citation_accuracy_history,
            statute_f1_history,
            expert_truthfulness_history,
            oversight_efficiency_history,
            bail_decision_accuracy,
        )
        print("  Saved: training_results.png")

    return results


def _plot_training_results(
    rewards: List[float],
    citation_acc: List[float],
    statute_f1: List[float],
    expert_truth: List[float],
    oversight_eff: List[float],
    bail_acc: List[float],
):
    """Generate 4 training metric plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ⚠ matplotlib not available — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Nyaya-Env — Training Results\n"
        "Scalable-Oversight Gym for Indian Bail Jurisprudence",
        fontsize=14, fontweight="bold"
    )

    window = 50

    def moving_avg(data, w):
        return [np.mean(data[max(0, i - w):i]) for i in range(1, len(data) + 1)]

    # ── Plot 1: Citation Accuracy ──
    axes[0, 0].plot(citation_acc, alpha=0.2, color="#3498db", linewidth=0.5)
    if len(citation_acc) >= window:
        ma = moving_avg(citation_acc, window)
        axes[0, 0].plot(ma, color="#2c3e50", linewidth=2, label=f"{window}-ep MA")
    axes[0, 0].set_title("1. Citation Accuracy", fontweight="bold")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_ylim(-0.05, 1.05)
    axes[0, 0].axhline(y=0.7, color="#27ae60", linestyle="--", alpha=0.5, label="Target")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # ── Plot 2: BNSS Statute F1 ──
    axes[0, 1].plot(statute_f1, alpha=0.2, color="#e67e22", linewidth=0.5)
    if len(statute_f1) >= window:
        ma = moving_avg(statute_f1, window)
        axes[0, 1].plot(ma, color="#d35400", linewidth=2, label=f"{window}-ep MA")
        axes[0, 1].fill_between(range(len(ma)), ma, alpha=0.1, color="#e67e22")
    axes[0, 1].set_title("2. BNSS 480 Factor Coverage (Statute F1)", fontweight="bold")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("F1 Score")
    axes[0, 1].set_ylim(-0.05, 1.05)
    axes[0, 1].axhline(y=1.0, color="#27ae60", linestyle="--", alpha=0.5, label="All 6 factors")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # ── Plot 3: Expert Truthfulness (Snorkel) ──
    axes[1, 0].plot(expert_truth, alpha=0.2, color="#9b59b6", linewidth=0.5)
    if len(expert_truth) >= window:
        ma = moving_avg(expert_truth, window)
        axes[1, 0].plot(ma, color="#8e44ad", linewidth=2, label=f"{window}-ep MA")
        axes[1, 0].fill_between(range(len(ma)), ma, alpha=0.1, color="#9b59b6")
    axes[1, 0].set_title("3. Expert Truthfulness (Snorkel SME)", fontweight="bold")
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Truthfulness (1 - delta)")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # ── Plot 4: Oversight Efficiency (Fleet AI) ──
    axes[1, 1].plot(oversight_eff, alpha=0.2, color="#27ae60", linewidth=0.5)
    if len(oversight_eff) >= window:
        ma = moving_avg(oversight_eff, window)
        axes[1, 1].plot(ma, color="#1e8449", linewidth=2, label=f"{window}-ep MA")
        axes[1, 1].fill_between(range(len(ma)), ma, alpha=0.1, color="#27ae60")
    axes[1, 1].set_title("4. Oversight Efficiency (Fleet AI)", fontweight="bold")
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("Efficiency")
    axes[1, 1].set_ylim(-0.05, 1.05)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_results.png", dpi=150, bbox_inches="tight")
    plt.close()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    num_eps = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    results = train(num_episodes=num_eps)
