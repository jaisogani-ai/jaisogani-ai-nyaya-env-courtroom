"""
Nyaya-Env — LLM-Powered Inference + Bhashini Mock
====================================================
Runs all 3 tasks (easy, medium, hard) using an LLM via the OpenAI Client.
Falls back to rule-based agents if the API is unavailable.

Bhashini Integration: Mock stub for Hindi translation demo.
Mark as "Bhashini-ready" — real API key can be added post-hackathon.

Logging format (MANDATORY for OpenEnv):
  [START] {"task": "...", "model": "..."}
  [STEP]  {"step": 1, "action": "...", "reward": 0.0, "done": false}
  [END]   {"task": "...", "score": 0.0, "success": true}

Environment variables:
  API_BASE_URL  — Base URL for the LLM API
  MODEL_NAME    — Model identifier to use
  HF_TOKEN      — HuggingFace API token for authentication

Author: jaisogani-ai
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, Any, Optional, List

from environment import (
    CourtRoomEnv, CourtAction, CourtObservation,
    ALL_AGENTS, AGENT_JUDGE, AGENT_PROSECUTOR, AGENT_DEFENSE,
    AGENT_CLERK, AGENT_EXPERT,
    JudgeAction, ProsecutorAction, DefenseAction, ExpertAction,
)
from grader import grade_episode

# ── OpenAI client (used for ALL LLM calls) ──
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MAX_TIME_PER_TASK = 360  # 6 minutes per task


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Bhashini Mock — Hindi Translation (Bhashini-ready)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# MOCK STUB: Replace with real Bhashini API integration
# Real API endpoint: https://bhashini.gov.in/ulca/apis
# This generates placeholder Hindi equivalents for demo purposes.

BHASHINI_HINDI_MAP = {
    "bail_granted": "ज़मानत मंज़ूर (Bail Granted)",
    "bail_denied": "ज़मानत नामंज़ूर (Bail Denied)",
    "Article 21": "अनुच्छेद 21 — जीवन और स्वतंत्रता का अधिकार",
    "flight_risk": "फ़रार होने का ख़तरा",
    "evidence_strength": "साक्ष्य की मज़बूती",
    "judge": "न्यायाधीश",
    "prosecutor": "अभियोजक",
    "defense": "बचाव पक्ष",
    "expert_witness": "विशेषज्ञ गवाह",
    "clerk": "न्यायालय लिपिक",
    "PMLA": "धन शोधन निवारण अधिनियम (PMLA)",
    "UAPA": "विधि विरुद्ध क्रिया-कलाप (निवारण) अधिनियम (UAPA)",
    "BNS 318": "भारतीय न्याय संहिता धारा 318 (धोखाधड़ी)",
    "BNSS 480": "भारतीय नागरिक सुरक्षा संहिता धारा 480",
    "bail_is_rule": "ज़मानत नियम है, जेल अपवाद है",
}


def bhashini_translate(text: str) -> str:
    """
    Mock Bhashini API: translate key legal terms to Hindi.

    NOTE: This is a MOCK STUB for demo purposes.
    Production integration requires:
      - Bhashini ULCA API key
      - NMT model pipeline configuration
      - Real-time translation service

    Args:
        text: English text to translate.

    Returns:
        Hindi translation (mock) or original text.
    """
    # Try exact match first
    if text in BHASHINI_HINDI_MAP:
        return BHASHINI_HINDI_MAP[text]

    # Try partial match
    for eng, hindi in BHASHINI_HINDI_MAP.items():
        if eng.lower() in text.lower():
            text = text.replace(eng, hindi)

    return f"[भाषिणी] {text}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Logging Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log_start(task: str, model: str):
    """Print [START] log in the exact required format."""
    print(f'[START] {json.dumps({"task": task, "model": model})}')
    sys.stdout.flush()


def log_step(step: int, action: str, reward: float, done: bool):
    """Print [STEP] log in the exact required format."""
    print(f'[STEP] {json.dumps({"step": step, "action": action, "reward": round(reward, 4), "done": done})}')
    sys.stdout.flush()


def log_end(task: str, score: float, success: bool):
    """Print [END] log in the exact required format."""
    print(f'[END] {json.dumps({"task": task, "score": round(score, 4), "success": success})}')
    sys.stdout.flush()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Agent — Uses OpenAI Client with 5-Agent Bail Hearing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LLMAgent:
    """
    LLM-powered agent that uses the OpenAI Client to decide actions
    for all 5 bail hearing agents simultaneously.

    Falls back to rule-based decisions if the API call fails.
    """

    def __init__(self, api_base_url: str, model_name: str, hf_token: str):
        self.model_name = model_name
        self.client = None
        self.api_available = False

        if OPENAI_AVAILABLE and hf_token:
            try:
                self.client = OpenAI(
                    base_url=api_base_url,
                    api_key=hf_token,
                )
                self.api_available = True
            except Exception as e:
                print(f"[WARN] OpenAI client init failed: {e}", file=sys.stderr)

    def decide(self, obs: CourtObservation, env: CourtRoomEnv) -> CourtAction:
        """Decide actions for all 5 agents given current observation."""
        if self.api_available and self.client:
            try:
                return self._llm_decide(obs, env)
            except Exception as e:
                print(f"[WARN] LLM call failed, using rule-based fallback: {e}", file=sys.stderr)
        return self._rule_based_decide(obs)

    def _llm_decide(self, obs: CourtObservation, env: CourtRoomEnv) -> CourtAction:
        """Use the LLM to decide actions for all 5 agents."""
        system_prompt = (
            "You are controlling 5 agents in an Indian bail hearing simulation (Nyaya-Env v3.0). "
            "Based on the current state, decide the best action for EACH agent. "
            "Your goal is to reach the CORRECT bail decision through adversarial legal reasoning.\n\n"
            "LEGAL FRAMEWORK (NEWEST LAWS — effective July 2024):\n"
            "  BNS 2023 replaces IPC.  BNSS 2023 replaces CrPC.\n"
            "  Article 21: Right to life and liberty\n"
            "  Core principle: Bail is rule, jail is exception (Gudikanti 1977)\n\n"
            "AGENT ACTION SPACES:\n"
            "Judge: 0=assess_flight_risk, 1=assess_gravity, 2=ask_clarification (costs budget), "
            "3=impose_condition, 4=grant_bail, 5=deny_bail, 6=video_remand_order\n"
            "Prosecutor: 0=present_evidence, 1=cite_BNS_section (+0.2), 2=cite_BNSS_section (+0.2), "
            "3=cite_SC_precedent (+0.3), 4=argue_flight_risk, 5=invoke_PMLA_twin_test, 6=cross_examine_expert\n"
            "Defense: 0=invoke_Article_21, 1=cite_Antil_guidelines, 2=argue_90_day_default_bail (BNSS 187), "
            "3=challenge_PMLA_twin_test, 4=propose_bail_conditions, 5=cite_Najeeb_delay, 6=examine_expert\n"
            "Expert: 0=testify_truthful, 1=testify_partial, 2=testify_fabricated, 3=reveal_key_fact\n\n"
            "Clerk is ALWAYS 0 (deterministic BNSS rule engine).\n\n"
            "Respond with ONLY a JSON object: "
            '{"judge": <int>, "prosecutor": <int>, "defense": <int>, "expert_witness": <int>}'
        )

        user_prompt = (
            f"CURRENT BAIL HEARING STATE (Round {obs.hearing_round}/{obs.max_rounds}):\n"
            f"- Case Type: {obs.case_type}\n"
            f"- Phase: {obs.current_phase}\n"
            f"- Evidence Strength: {obs.evidence_strength:.2f}\n"
            f"- Flight Risk: {obs.flight_risk_score:.2f}\n"
            f"- Case Gravity: {obs.case_gravity:.2f}\n"
            f"- Prosecution Score: {obs.prosecution_score:.2f}\n"
            f"- Defense Score: {obs.defense_score:.2f}\n"
            f"- Oversight Budget Remaining: {obs.oversight_budget}\n"
            f"- BNSS Factors Assessed: {obs.factors_assessed_count}/6\n"
            f"- Citation Accuracy: {obs.citation_accuracy:.2f}\n"
            f"- Expert Type: {obs.expert_type}\n"
            f"- Delay: {obs.delay_duration_months} months\n"
            f"- Article 21 Breach: {obs.article21_threshold_breached}\n"
            f"- Video Remand (BNSS 530): {obs.video_remand}\n"
            f"- Days Since Arrest: {obs.days_since_arrest}\n"
            f"- Charge Sheet Filed: {obs.charge_sheet_filed}\n"
            f"- Deception Detected: {obs.deception_detected}\n"
            f"- Verdict Status: {obs.verdict}\n"
            f"- INJECTED EVIDENCE (MUST ADAPT TO THIS): {obs.injected_events if hasattr(obs, 'injected_events') else []}\n\n"
            "Decide the best action for each agent. Use action numbers only."
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=200,
            timeout=30,
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON from response
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        start_idx = content.find("{")
        end_idx = content.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = content[start_idx:end_idx]
            parsed = json.loads(json_str)

            return CourtAction(
                judge=int(parsed.get("judge", 0)) % 7,
                prosecutor=int(parsed.get("prosecutor", 0)) % 6,
                defense=int(parsed.get("defense", 0)) % 7,
                clerk=0,  # Always deterministic
                expert_witness=int(parsed.get("expert_witness", 0)) % 4,
            )

        raise ValueError(f"Could not parse LLM response: {content}")

    def _rule_based_decide(self, obs: CourtObservation) -> CourtAction:
        """
        Sophisticated rule-based fallback for 5-agent bail hearing.

        Strategy:
          - Judge: Systematically assess BNSS factors, then decide
          - Prosecutor: Present evidence, cite statutes, argue flight risk
          - Defense: Argue liberty, invoke Article 21, challenge evidence
          - Expert: Testify truthfully, reveal key facts
          - Clerk: Always 0 (deterministic)
        """
        phase = obs.current_phase
        rnd = obs.hearing_round
        ev = obs.evidence_strength
        pros = obs.prosecution_score
        defs = obs.defense_score

        # ── Judge logic — weak overseer ──
        if rnd <= 2:
            judge = JudgeAction.ASSESS_FLIGHT_RISK
        elif rnd <= 4:
            if obs.factors_assessed_count < 4:
                judge = JudgeAction.ASSESS_GRAVITY
            elif obs.oversight_budget > 0:
                judge = JudgeAction.ASK_CLARIFICATION
            else:
                judge = JudgeAction.IMPOSE_CONDITION
        elif rnd >= 5:
            if pros > defs + 0.1:
                judge = JudgeAction.DENY_BAIL
            else:
                judge = JudgeAction.GRANT_BAIL
        else:
            judge = JudgeAction.VIDEO_REMAND_ORDER

        # ── Prosecutor logic ──
        if rnd <= 1:
            prosecutor = ProsecutorAction.PRESENT_EVIDENCE
        elif rnd == 2:
            prosecutor = ProsecutorAction.CITE_BNS_SECTION
        elif obs.case_type == "pmla_bail" and rnd <= 4:
            prosecutor = ProsecutorAction.INVOKE_PMLA_TWIN_TEST
        elif obs.flight_risk_score > 0.5:
            prosecutor = ProsecutorAction.ARGUE_FLIGHT_RISK
        elif rnd >= 4:
            prosecutor = ProsecutorAction.CROSS_EXAMINE_EXPERT
        else:
            prosecutor = ProsecutorAction.CITE_SC_PRECEDENT

        # ── Defense logic ──
        if obs.article21_threshold_breached:
            defense = DefenseAction.INVOKE_ARTICLE_21
        elif not obs.charge_sheet_filed and obs.days_since_arrest > 90:
            defense = DefenseAction.ARGUE_90_DAY_DEFAULT_BAIL
        elif obs.case_type == "uapa_43d_bail" and obs.delay_duration_months > 24:
            defense = DefenseAction.CITE_NAJEEB_DELAY
        elif rnd <= 2:
            defense = DefenseAction.INVOKE_ARTICLE_21
        elif rnd <= 4:
            defense = DefenseAction.CITE_ANTIL_GUIDELINES
        elif obs.case_type == "pmla_bail":
            defense = DefenseAction.CHALLENGE_PMLA_TWIN_TEST
        else:
            defense = DefenseAction.PROPOSE_BAIL_CONDITIONS

        # ── Expert logic ──
        if rnd <= 3:
            expert = ExpertAction.TESTIFY_TRUTHFUL
        elif rnd == 4:
            expert = ExpertAction.REVEAL_KEY_FACT
        else:
            expert = ExpertAction.TESTIFY_TRUTHFUL

        return CourtAction(
            judge=int(judge),
            prosecutor=int(prosecutor),
            defense=int(defense),
            clerk=0,
            expert_witness=int(expert),
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Task Runner
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_task(task: str, agent: LLMAgent, env: CourtRoomEnv) -> Dict[str, Any]:
    """Run a single task to completion with mandatory logging."""
    start_time = time.time()
    model = agent.model_name if agent.api_available else "rule-based-fallback"

    log_start(task, model)

    try:
        obs = env.reset(task=task)
        step_num = 0

        while not obs.done:
            step_num += 1

            elapsed = time.time() - start_time
            if elapsed > MAX_TIME_PER_TASK:
                print(f"[WARN] Task {task} exceeded time budget ({elapsed:.1f}s)", file=sys.stderr)
                break

            action = agent.decide(obs, env)
            obs = env.step(action)

            total_reward = sum(obs.rewards.values()) if obs.rewards else 0.0

            action_summary = (
                f"J:{action.judge} P:{action.prosecutor} D:{action.defense} "
                f"C:{action.clerk} E:{action.expert_witness}"
            )

            log_step(step_num, action_summary, total_reward, obs.done)

            if step_num > 20:
                print(f"[WARN] Task {task}: forced stop after {step_num} steps", file=sys.stderr)
                break

        trajectory = env.get_trajectory()
        result = grade_episode(trajectory, task)

        # ── Bhashini Hindi demo ──
        verdict_hindi = bhashini_translate(trajectory.get("verdict", "pending"))
        print(f"  [भाषिणी] Verdict: {verdict_hindi}", file=sys.stderr)

        log_end(task, result["score"], result["passed"])
        return result

    except Exception as e:
        print(f"[ERROR] Task {task} failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        log_end(task, 0.0, False)
        return {"task": task, "passed": False, "score": 0.0, "details": f"Error: {e}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main Entry Point
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    """Run all 3 tasks: easy → medium → hard."""
    print("=" * 65)
    print("  🏛️  NYAYA-ENV — LLM Inference Pipeline")
    print("  Scalable-Oversight Gym for Indian Bail Jurisprudence")
    print("=" * 65)
    print(f"  API Base URL: {API_BASE_URL}")
    print(f"  Model:        {MODEL_NAME}")
    print(f"  Token:        {'***' + HF_TOKEN[-4:] if HF_TOKEN else 'NOT SET'}")
    print(f"  Bhashini:     MOCK STUB (Bhashini-ready)")
    print("=" * 65)

    agent = LLMAgent(API_BASE_URL, MODEL_NAME, HF_TOKEN)
    if agent.api_available:
        print("[INFO] LLM API connected successfully.")
    else:
        print("[INFO] LLM API unavailable — using rule-based fallback.")

    env = CourtRoomEnv(seed=42)

    tasks = ["easy", "medium", "hard"]
    results = []
    total_start = time.time()

    for task in tasks:
        print(f"\n{'─' * 55}")
        print(f"  Running Task: {task.upper()}")
        print(f"{'─' * 55}")

        result = run_task(task, agent, env)
        results.append(result)
        print(f"  Result: {result['details']}")

    total_time = time.time() - total_start
    print(f"\n{'═' * 65}")
    print("  🏛️  INFERENCE COMPLETE — SUMMARY")
    print(f"{'═' * 65}")

    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  [{status}] {r['task'].upper():8s} — Score: {r['score']:.4f}")

    avg_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average Score:   {avg_score:.4f}")
    print(f"  Total Time:      {total_time:.1f}s")
    print(f"  Time Budget:     {'OK' if total_time < 1200 else 'EXCEEDED'} (limit: 20min)")
    print(f"  Bhashini Status: MOCK STUB (ready for Bhashini API key)")
    print(f"{'═' * 65}")


if __name__ == "__main__":
    main()
