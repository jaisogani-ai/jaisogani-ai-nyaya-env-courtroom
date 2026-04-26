"""
Nyaya-Env — Realistic Indian Legal Case System
=================================================
High-fidelity case generation based on actual Indian Bail Jurisprudence
and real court case patterns from India's judicial system.

Case types modeled (BNS 2023 — replaces IPC, effective July 2024):
  1. PMLA Bail Hearing (Section 45 Twin Test)
  2. BNS §318 Fraud Bail (BNSS §480 standard criteria)
  3. UAPA Terror Funding Bail (Section 43D(5))
  4. BNS §111 Organised Crime Bail (NEW — no IPC equivalent)

Each case includes:
  - Realistic evidence chains with hidden fabrication flags
  - Witness profiles with reliability scores
  - Verifiable precedent citations essential for winning bail
  - Ground truth expert facts for Snorkel AI programmatic checking
  - Prior rejection probabilities based on real Indian statistics

Data inspired by:
  - NCRB (National Crime Records Bureau) India 2024
  - IndianKanoon & NyayaAnumana datasets
  - Supreme Court seminal bail jurisprudence

Author: jaisogani-ai
"""

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Case Type Definitions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Constants for new Indian Law Reward Metrics
REWARD_CORRECT_CITATION = 0.3
PENALTY_FABRICATED_CITATION = -0.5
REWARD_BNSS_CRITERION = 0.1

class CaseType(str, Enum):
    """Categories of criminal bail hearings (BNS 2023 + special laws)."""
    PMLA_BAIL = "pmla_bail"
    BNS_318_BAIL = "bns_318_bail"
    UAPA_43D_BAIL = "uapa_43d_bail"
    BNS_111_ORGANISED_CRIME = "bns_111_organised_crime"

class CaseComplexity(str, Enum):
    """Case complexity levels — drives evidence ambiguity and deception difficulty."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Evidence Items
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Evidence:
    """
    A single piece of evidence in the case.

    Each evidence item has:
      - A descriptive name
      - A strength score (how convincing it is)
      - A fabrication flag (hidden — is this evidence real?)
      - A relevance score (how applicable to the case)
    """
    name: str
    description: str
    strength: float           # [0.0, 1.0] — how convincing
    fabricated: bool = False  # Hidden: is this evidence faked?
    relevance: float = 0.8   # [0.0, 1.0] — how applicable to the charge
    examined: bool = False    # Has this evidence been presented in trial?
    challenged: bool = False  # Has the opposing side challenged it?

    @property
    def effective_strength(self) -> float:
        """
        Effective evidence strength accounting for fabrication.

        Fabricated evidence appears strong but collapses under scrutiny.
        Challenged fabricated evidence loses most of its strength.
        """
        if self.fabricated and self.challenged:
            return self.strength * 0.15  # Collapses when challenged
        elif self.fabricated:
            return self.strength * 0.85  # Appears almost real
        return self.strength


@dataclass
class WitnessProfile:
    """
    Profile of a witness in the case.

    Each witness has:
      - A reliability score (how trustworthy)
      - A truthful flag (hidden — is this witness lying?)
      - A relationship to the defendant
      - Testimony content
    """
    name: str
    role: str                    # e.g., "eyewitness", "expert", "character"
    reliability: float = 0.5    # [0.0, 1.0] — how trustworthy
    is_truthful: bool = True    # Hidden: is this witness being honest?
    relationship: str = "none"  # Relationship to defendant
    testimony_summary: str = ""
    has_testified: bool = False
    credibility_after_cross: float = 0.5  # Updated after cross-examination


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Case Data Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class CourtCase:
    """
    Complete case file for a courtroom trial.

    This is the high-fidelity case object that drives the trial.
    It contains all evidence, witnesses, and hidden ground truth
    that agents must discover through adversarial gameplay.
    """
    # ── Case identity ──
    case_id: str = ""
    case_type: CaseType = CaseType.BNS_318_BAIL
    complexity: CaseComplexity = CaseComplexity.MEDIUM
    ipc_section: str = ""
    case_title: str = ""
    case_summary: str = ""

    # ── Ground truth (hidden from agents) ──
    defendant_guilty: bool = True
    true_verdict: str = "guilty"     # "guilty" or "innocent"
    guilty_confidence: float = 0.8   # How clear-cut is the guilt? [0.5, 1.0]

    # ── Evidence chain ──
    evidence: List[Evidence] = field(default_factory=list)
    evidence_count: int = 0
    fabricated_count: int = 0  # How many pieces are fabricated (hidden)

    # ── Witness pool ──
    witnesses: List[WitnessProfile] = field(default_factory=list)
    deceptive_witness_count: int = 0  # How many are lying (hidden)

    # ── Aggregate scores (observable) ──
    overall_evidence_strength: float = 0.5
    case_strength_prosecution: float = 0.5
    case_strength_defense: float = 0.5

    # ── Trial dynamics ──
    plea_bargain_available: bool = False
    appeal_possible: bool = False
    mitigating_factors: List[str] = field(default_factory=list)
    aggravating_factors: List[str] = field(default_factory=list)

    # ── Ground Truth & Verifiable Constraints (For Reward System) ──
    required_citations: List[str] = field(default_factory=list)
    bnss_criteria: List[str] = field(default_factory=list)
    prior_rejection_prob: float = 0.5
    expert_facts: Dict[str, Any] = field(default_factory=dict)

    def get_observable_state(self) -> Dict[str, Any]:
        """
        Get the observable state (what agents can see).
        Ground truth fields are excluded.
        """
        return {
            "case_type": self.case_type.value,
            "complexity": self.complexity.value,
            "ipc_section": self.ipc_section,
            "case_title": self.case_title,
            "case_summary": self.case_summary,
            "evidence_count": self.evidence_count,
            "overall_evidence_strength": round(self.overall_evidence_strength, 3),
            "prosecution_strength": round(self.case_strength_prosecution, 3),
            "defense_strength": round(self.case_strength_defense, 3),
            "witness_count": len(self.witnesses),
            "plea_bargain_available": self.plea_bargain_available,
            "mitigating_factors": self.mitigating_factors,
            "aggravating_factors": self.aggravating_factors,
            "required_citations_count": len(self.required_citations),
            "bnss_criteria": self.bnss_criteria,
            "prior_rejection_prob": self.prior_rejection_prob,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Case Generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class CaseGenerator:
    """
    Generates realistic Indian court cases with appropriate
    evidence chains, witness profiles, and hidden truths.

    Each case type has distinct evidence patterns:
      - Financial Fraud: Paper trails, forensic accounting
      - Assault: Physical evidence, eyewitness testimony
      - Corporate Corruption: Whistleblowers, money trails

    The generator controls:
      - Whether the defendant is actually guilty
      - Whether evidence has been fabricated
      - Whether witnesses are truthful
      - How complex the case is to adjudicate
    """

    # ── Case templates — Indian Bail Jurisprudence ──

    PMLA_BAIL_TEMPLATES = [
        {
            "title": "ED v. Deshmukh — Money Laundering Bail",
            "summary": "Bail hearing under PMLA Section 45. Defendant accused of laundering ₹50 crore via shell companies. Must satisfy twin conditions: (1) prima facie innocent, and (2) unlikely to commit offence on bail.",
            "aggravating": ["flight_risk", "evidence_tampering_risk", "hawala_links"],
            "mitigating": ["medical_condition", "extended_unjustified_custody"],
            "required_citations": ["Vijay Madanlal Choudhary vs UOI 2022"],
            "bnss_criteria": ["flight_risk", "tampering", "witness_influence"],
            "prior_rejection_prob": 0.96,
        }
    ]

    BNS_318_BAIL_TEMPLATES = [
        {
            "title": "State v. Chandra — Corporate Fraud Bail",
            "summary": "Bail hearing under BNS Section 318 (formerly IPC 420) for corporate financial fraud. Standard bail criteria apply under BNSS 437.",
            "aggravating": ["large_financial_loss", "corporate_position"],
            "mitigating": ["cooperation_with_investigation", "assets_frozen"],
            "required_citations": ["Sanjay Chandra vs CBI 2011"],
            "bnss_criteria": ["flight_risk", "tampering", "witness_influence", "nature_of_offence", "character_of_accused"],
            "prior_rejection_prob": 0.40,
        }
    ]

    UAPA_43D_BAIL_TEMPLATES = [
        {
            "title": "NIA v. Najeeb — UAPA Terror Funding Bail",
            "summary": "Bail hearing under UAPA Section 43D(5). Court must assess if accusations are prima facie true based on case diary/report. Extreme threshold for bail.",
            "aggravating": ["national_security", "terror_funding", "prima_facie_true"],
            "mitigating": ["long_incarceration_delay", "article_21_violation"],
            "required_citations": ["K.A. Najeeb vs UOI 2021"],
            "bnss_criteria": ["prima_facie_test", "flight_risk"],
            "prior_rejection_prob": 0.99,
        }
    ]

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the case generator.

        Args:
            seed: Optional random seed for reproducible case generation.
        """
        self._rng = random.Random(seed)

    def generate(
        self,
        case_type: Optional[CaseType] = None,
        complexity: Optional[CaseComplexity] = None,
        force_guilty: Optional[bool] = None,
    ) -> CourtCase:
        """
        Generate a complete court case with evidence and witnesses.

        Args:
            case_type: Type of case (random if None).
            complexity: Complexity level (random if None).
            force_guilty: Force defendant guilt status (random if None).

        Returns:
            Fully populated CourtCase instance.
        """
        if case_type is None:
            case_type = self._rng.choice(list(CaseType))
        if complexity is None:
            complexity = self._rng.choice(list(CaseComplexity))

        # ── Select ground truth ──
        if force_guilty is not None:
            guilty = force_guilty
        else:
            guilty = self._rng.random() > 0.45  # Slightly biased toward guilty (realistic)

        # ── Generate based on case type ──
        generators = {
            CaseType.PMLA_BAIL: self._generate_pmla_bail,
            CaseType.BNS_318_BAIL: self._generate_bns_318_bail,
            CaseType.UAPA_43D_BAIL: self._generate_uapa_43d_bail,
            CaseType.BNS_111_ORGANISED_CRIME: lambda c, g: self._build_bns_111_case(g),
        }
        return generators[case_type](complexity, guilty)

    def _generate_pmla_bail(self, complexity: CaseComplexity, guilty: bool) -> CourtCase:
        """Generate a PMLA Section 45 twin test bail case."""
        template = self._rng.choice(self.PMLA_BAIL_TEMPLATES)
        case = self._build_base_case(template, CaseType.PMLA_BAIL, "PMLA §45", CaseComplexity.HARD, guilty)
        case.prior_rejection_prob = template["prior_rejection_prob"]
        case.expert_facts = {"financial_mismatch": "₹50 crore", "shell_companies_count": 4}
        
        evidence_pool = [
            Evidence("ED Attachment Order", "Provisional attachment of properties", 0.0, False, 0.9),
            Evidence("Forensic Audit", "Trace of funds through 4 shell companies", 0.0, False, 0.85),
            Evidence("Witness Statement (Approver)", "Section 50 statement implicating accused", 0.0, False, 0.75),
        ]
        case.evidence = self._populate_evidence(evidence_pool, CaseComplexity.HARD, guilty)
        
        witnesses = [
            WitnessProfile("ED Investigator", "expert", 0.0, True, "none", "Explains the hawala network"),
        ]
        case.witnesses = self._populate_witnesses(witnesses, CaseComplexity.HARD, guilty)
        
        self._compute_aggregate_scores(case)
        return case

    def _generate_bns_318_bail(self, complexity: CaseComplexity, guilty: bool) -> CourtCase:
        """Generate a BNS Section 318 fraud case (BNSS 437 criteria)."""
        template = self._rng.choice(self.BNS_318_BAIL_TEMPLATES)
        case = self._build_base_case(template, CaseType.BNS_318_BAIL, "BNS §318 / BNSS §437", CaseComplexity.MEDIUM, guilty)
        case.prior_rejection_prob = template["prior_rejection_prob"]
        case.expert_facts = {"fraud_amount": "₹12 crore", "forged_signatures": 2}
        
        evidence_pool = [
            Evidence("Bank Statements", "Transactions matching the alleged fraud amount", 0.0, False, 0.85),
            Evidence("Handwriting Report", "Analysis confirming 2 forged signatures", 0.0, False, 0.9),
            Evidence("Audit Trail", "Internal emails showing conspiracy", 0.0, False, 0.7),
        ]
        case.evidence = self._populate_evidence(evidence_pool, CaseComplexity.MEDIUM, guilty)
        
        witnesses = [
            WitnessProfile("Forensic Accountant", "expert", 0.0, True, "none", "Analyzed the digital footprint"),
            WitnessProfile("Company Director", "eyewitness", 0.0, True, "colleague", "Testifies about accused's role"),
        ]
        case.witnesses = self._populate_witnesses(witnesses, CaseComplexity.MEDIUM, guilty)
        
        self._compute_aggregate_scores(case)
        return case

    def _generate_uapa_43d_bail(self, complexity: CaseComplexity, guilty: bool) -> CourtCase:
        """Generate a UAPA Section 43D bail case (Prima facie test)."""
        template = self._rng.choice(self.UAPA_43D_BAIL_TEMPLATES)
        case = self._build_base_case(template, CaseType.UAPA_43D_BAIL, "UAPA §43D(5)", CaseComplexity.HARD, guilty)  # Always hard
        case.prior_rejection_prob = template["prior_rejection_prob"]
        case.expert_facts = {"terror_funding": "₹5 crore sent to banned outfit", "encrypted_chats": 14}
        
        evidence_pool = [
            Evidence("NIA Charge Sheet", "Detailed allegations of terror funding", 0.0, False, 0.95),
            Evidence("Digital Forensics", "14 Encrypted communications recovered", 0.0, False, 0.9),
            Evidence("Protected Witness X", "Secret testimony under UAPA", 0.0, False, 0.8),
        ]
        case.evidence = self._populate_evidence(evidence_pool, CaseComplexity.HARD, guilty)
        
        witnesses = [
            WitnessProfile("NIA Cyber Expert", "expert", 0.0, True, "none", "Decrypted the communications"),
        ]
        case.witnesses = self._populate_witnesses(witnesses, CaseComplexity.HARD, guilty)
        
        self._compute_aggregate_scores(case)
        return case

    def _build_bns_111_case(self, guilty: bool) -> CourtCase:
        """
        Build BNS Section 111 Organised Crime case.

        BNS 111 is a NEW provision (no IPC equivalent).
        Covers organised crime syndicates, gang-related offences.

        Bail assessment:
          - Standard BNSS 480 criteria + additional organised crime factors
          - Gang membership evidence
          - Criminal history pattern
          - Witness protection requirements

        Difficulty: HARD
        Grant rate: ~30%
        """
        template = self._rng.choice([
            {
                "title": "State vs Raju Singh — BNS 111 Organised Crime Syndicate",
                "summary": (
                    "Accused is alleged member of an organised crime syndicate "
                    "operating across Maharashtra and Gujarat. Charges under BNS "
                    "Section 111 for running an extortion and drug distribution "
                    "network. 12 co-accused, 3 previously convicted."
                ),
                "aggravating": [
                    "Multi-state syndicate operation",
                    "Prior co-accused convictions",
                    "Witness intimidation allegations",
                ],
                "mitigating": [
                    "First-time accused (no prior convictions)",
                    "Cooperative during investigation",
                    "Peripheral role alleged",
                ],
                "required_citations": [
                    "BNS Section 111",
                    "BNSS Section 480 (non-bailable)",
                    "Arnesh Kumar vs Bihar 2014",
                ],
                "bnss_criteria": [
                    "nature_gravity",
                    "antecedents",
                    "flight_risk",
                    "community_safety",
                    "repeat_offence",
                    "character_behaviour",
                ],
            },
            {
                "title": "State vs Mohammed Iqbal — BNS 111 Cyber Crime Ring",
                "summary": (
                    "Accused allegedly operated a cyber crime ring conducting "
                    "phishing and identity theft across 5 states. Charges under "
                    "BNS Section 111 (organised crime) read with Section 318 "
                    "(cheating). Proceeds of ₹8.5 crore identified."
                ),
                "aggravating": [
                    "Inter-state cyber crime ring",
                    "₹8.5 crore proceeds identified",
                    "Technical sophistication",
                ],
                "mitigating": [
                    "No prior criminal record",
                    "Age 24 — young accused",
                    "Willingness to cooperate with investigation",
                ],
                "required_citations": [
                    "BNS Section 111",
                    "BNS Section 318",
                    "BNSS Section 480",
                    "Satendra Kumar Antil vs CBI 2022",
                ],
                "bnss_criteria": [
                    "nature_gravity",
                    "antecedents",
                    "flight_risk",
                    "community_safety",
                ],
            },
        ])

        case = self._build_base_case(
            template, CaseType.BNS_111_ORGANISED_CRIME,
            "BNS 111 — Organised Crime",
            CaseComplexity.HARD, guilty,
        )

        # Expert ground truth for BNS 111
        case.expert_facts = {
            "gang_membership_evidence": round(self._rng.uniform(0.2, 0.9), 3),
            "criminal_history": [f"Case #{self._rng.randint(100, 999)}" for _ in range(self._rng.randint(0, 5))],
            "witness_protection_needed": self._rng.random() > 0.4,
            "proceeds_of_crime": self._rng.randint(1000000, 100000000),
            "co_accused_count": self._rng.randint(3, 15),
        }

        evidence_pool = [
            Evidence("Call Data Records", "CDR analysis showing coordination", 0.0, False, 0.85),
            Evidence("Financial Trail", "Bank statements linking to syndicate", 0.0, False, 0.75),
            Evidence("Informer Statement", "Statement under Section 164 CrPC", 0.0, False, 0.65),
            Evidence("CCTV Footage", "Surveillance footage of meeting", 0.0, False, 0.80),
            Evidence("Digital Evidence", "Encrypted chats on Signal/Telegram", 0.0, False, 0.70),
        ]
        case.evidence = self._populate_evidence(evidence_pool, CaseComplexity.HARD, guilty)

        witnesses = [
            WitnessProfile("Police Informer", "whistleblower", 0.0, True, "none", "Infiltrated the syndicate"),
            WitnessProfile("Cyber Forensics Expert", "expert", 0.0, True, "none", "Analysed digital evidence"),
        ]
        case.witnesses = self._populate_witnesses(witnesses, CaseComplexity.HARD, guilty)

        self._compute_aggregate_scores(case)
        return case

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Builder Helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _build_base_case(
        self,
        template: Dict,
        case_type: CaseType,
        ipc_section: str,
        complexity: CaseComplexity,
        guilty: bool,
    ) -> CourtCase:
        """Build the base case structure from a template."""
        guilty_conf = self._rng.uniform(0.55, 0.95) if guilty else self._rng.uniform(0.55, 0.95)

        return CourtCase(
            case_id=str(uuid.uuid4())[:8],
            case_type=case_type,
            complexity=complexity,
            ipc_section=ipc_section,
            case_title=template["title"],
            case_summary=template["summary"],
            defendant_guilty=guilty,
            true_verdict="guilty" if guilty else "innocent",
            guilty_confidence=guilty_conf,
            aggravating_factors=template.get("aggravating", []),
            mitigating_factors=template.get("mitigating", []),
            required_citations=template.get("required_citations", []),
            bnss_criteria=template.get("bnss_criteria", []),
        )

    def _populate_evidence(
        self,
        pool: List[Evidence],
        complexity: CaseComplexity,
        guilty: bool,
    ) -> List[Evidence]:
        """
        Populate evidence with realistic strengths and fabrication flags.

        Complexity affects:
          - Easy: Strong evidence, no fabrication
          - Medium: Mixed evidence, possible fabrication
          - Hard: Weak evidence, multiple fabrications
        """
        # Select subset of evidence based on complexity
        if complexity == CaseComplexity.EASY:
            count = min(3, len(pool))
            fabrication_prob = 0.0
        elif complexity == CaseComplexity.MEDIUM:
            count = min(4, len(pool))
            fabrication_prob = 0.2
        else:
            count = min(len(pool), 6)
            fabrication_prob = 0.35

        selected = self._rng.sample(pool, count)

        for ev in selected:
            # Set strength based on guilt and complexity
            if guilty:
                base_strength = self._rng.uniform(0.5, 0.9) if complexity != CaseComplexity.HARD else self._rng.uniform(0.3, 0.7)
            else:
                base_strength = self._rng.uniform(0.2, 0.6) if complexity != CaseComplexity.HARD else self._rng.uniform(0.3, 0.8)

            ev.strength = round(base_strength, 3)

            # Fabrication: some evidence may be planted
            ev.fabricated = self._rng.random() < fabrication_prob

        fabricated_count = sum(1 for e in selected if e.fabricated)

        return selected

    def _populate_witnesses(
        self,
        profiles: List[WitnessProfile],
        complexity: CaseComplexity,
        guilty: bool,
    ) -> List[WitnessProfile]:
        """
        Populate witness profiles with reliability and truthfulness.

        Complexity affects witness reliability and deception probability:
          - Easy: Reliable witnesses, all truthful
          - Medium: Mixed reliability, 1 may be deceptive
          - Hard: Low reliability, multiple deceptive
        """
        deceptive_count = 0

        for w in profiles:
            # Base reliability depends on role
            role_reliability = {
                "expert": 0.8,
                "eyewitness": 0.6,
                "victim": 0.7,
                "character": 0.5,
                "whistleblower": 0.65,
                "approver": 0.55,
            }
            base = role_reliability.get(w.role, 0.5)

            if complexity == CaseComplexity.EASY:
                w.reliability = round(base + self._rng.uniform(0.05, 0.15), 3)
                w.is_truthful = True
            elif complexity == CaseComplexity.MEDIUM:
                w.reliability = round(base + self._rng.uniform(-0.1, 0.1), 3)
                if deceptive_count == 0 and self._rng.random() < 0.4:
                    w.is_truthful = False
                    deceptive_count += 1
            else:
                w.reliability = round(base + self._rng.uniform(-0.2, 0.05), 3)
                if self._rng.random() < 0.5:
                    w.is_truthful = False
                    deceptive_count += 1

            w.reliability = max(0.1, min(1.0, w.reliability))
            w.credibility_after_cross = w.reliability

        return profiles

    def _compute_aggregate_scores(self, case: CourtCase):
        """Compute aggregate evidence and case strength scores."""
        if case.evidence:
            strengths = [e.effective_strength for e in case.evidence]
            case.overall_evidence_strength = round(sum(strengths) / len(strengths), 3)
            case.evidence_count = len(case.evidence)
            case.fabricated_count = sum(1 for e in case.evidence if e.fabricated)
        else:
            case.overall_evidence_strength = 0.0

        case.deceptive_witness_count = sum(1 for w in case.witnesses if not w.is_truthful)

        # Prosecution strength correlates with guilt + evidence
        if case.defendant_guilty:
            case.case_strength_prosecution = round(
                case.overall_evidence_strength * 0.7 + case.guilty_confidence * 0.3, 3
            )
            case.case_strength_defense = round(1.0 - case.case_strength_prosecution + self._rng.uniform(-0.1, 0.1), 3)
        else:
            case.case_strength_defense = round(
                (1.0 - case.overall_evidence_strength) * 0.6 + (1.0 - case.guilty_confidence) * 0.4, 3
            )
            case.case_strength_prosecution = round(1.0 - case.case_strength_defense + self._rng.uniform(-0.1, 0.1), 3)

        case.case_strength_prosecution = max(0.1, min(0.95, case.case_strength_prosecution))
        case.case_strength_defense = max(0.1, min(0.95, case.case_strength_defense))
