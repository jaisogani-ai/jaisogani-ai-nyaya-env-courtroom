---
title: "Nyaya-Env: Scaling AI Oversight in Indian Bail Law via Multi-Agent Debate"
emoji: "⚖️"
colorFrom: "purple"
colorTo: "blue"
sdk: "gradio"
sdk_version: "4.26.0"
app_file: "server.py"
pinned: true
license: "mit"
---

# Nyaya-Env: Scaling AI Oversight in Indian Bail Jurisprudence via Debate ⚖️

*Meta PyTorch OpenEnv Hackathon 2026 Submission*

## The Problem: 4.34 Lakh Undertrials
Currently, **75% of India's prison population are undertrials** — over 4.34 lakh people awaiting trial. Legal AI could theoretically help clear this backlog, but traditional LLMs suffer from severe hallucinations when parsing complex legal statutes, and RL agents often game reward functions. 

How do we build a legal AI system that is safe, verifiable, and strictly aligned with the newly enacted **Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023**?

## The Solution: AI Safety via Debate
We implemented **Nyaya-Env**, a native `OpenEnv` reinforcement learning environment that instantiates Irving et al.'s (2018) "AI Safety via Debate" protocol directly into the Indian courtroom.

Instead of training a single monolithic legal AI, we decouple the architecture into asymmetric roles:
*   **The Defense (Trainable, Qwen2.5-1.5B + LoRA)**: Has privileged access to client facts (Bharatiya Sakshya Adhiniyam §128).
*   **The Prosecution (Trainable, Qwen2.5-1.5B + LoRA)**: Has access only to the FIR and chargesheet.
*   **The Judge (Frozen, Weak LLM Qwen2.5-0.5B)**: Sees only the on-record arguments presented by the advocates.

This architecture naturally forces the models to learn robust legal reasoning. The Prosecution cannot fabricate evidence without the Defense objecting, and the Defense must cite real case law to convince the frozen Judge. 

### Why this wins:
*   **Fleet AI Scalable Oversight**: We use a weak LLM to supervise much stronger trainable LLMs.
*   **Halluminate Asymmetric Information**: Information isolation is strictly enforced via OpenEnv's per-agent observation masking.

## Reward Shaping with Snorkel AI
Legal rules are not fuzzy; they are deterministic. To train the models via **GRPO (Group Relative Policy Optimization)**, we encoded the BNSS 2023 statutes into 7 Python labeling functions across BNSS general provisions and special acts (NDPS, UAPA, PMLA).

This acts as a **Snorkel-style weak programmatic supervision** layer. For example, if a case triggers the NDPS §37 twin conditions (commercial quantity), our labeling function automatically votes that bail must be denied unless specific criteria are met. The GRPO algorithm directly penalizes the LLM if it fails to apply this procedural rule.

## Anti-Hallucination Guardrails
To prevent the most common failure mode of legal LLMs — making up fake case law — we built a citation verification system. Our `reward_case_citation` function checks every cited Supreme Court precedent against a verified registry of landmark cases. Additionally, `reward_statutory_accuracy` validates every BNSS/BNS section number against a ground truth registry.
*   **Valid citation:** +1.0 reward (scaled by count)
*   **Hallucinated section:** -1.0 penalty

The result? An agent that strictly grounds its arguments in verifiable, empirical law.

## Training Pipeline: Unsloth + GRPO
We utilize TRL's experimental OpenEnv hooks combined with **Unsloth** and **vLLM** for extreme training efficiency.

1. **Cold-Start SFT**: We first fine-tune on the HLDC (Hindi Legal Documents Corpus) bail subset to teach the model the `<think>...</think>` CoT format.
2. **GRPO**: We train using `dr_grpo` loss over 8 generations per prompt, pushing the composite reward from 0.17 (baseline) to 0.70+ in just 300 steps.

[🔗 View our Training Notebook](training_colab.ipynb)  
[🔗 Check out the GitHub Repository](https://github.com/jaisogani-ai/nyaya-env)

*Nyaya-Env: Building the verifiable future of Indian Legal Tech.*
