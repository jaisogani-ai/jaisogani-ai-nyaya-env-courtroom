# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Nyaya-Env Curriculum Learning Scheduler
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# As per Hackathon Guide: "Keep the task simple at first... 
# easy tasks with short horizons, medium tasks... harder tasks
# only after the model starts getting non-zero reward."
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

from realistic_cases import CaseType

class EpisodeCurriculum:
    """
    Deterministic episode-based curriculum scheduler for TRL.
    Forces the requested progression:
    - Ep 1-100:   BNS 318 (simple fraud)
    - Ep 101-300: PMLA (medium)
    - Ep 301-500: UAPA (hard)
    """
    
    def __init__(self, max_episodes: int = 500):
        self.max_episodes = max_episodes
        self.current_episode = 1
        
    def step(self) -> None:
        """Advance the curriculum by one episode."""
        self.current_episode += 1
        
    def get_current_case_type(self) -> str:
        """Returns the specific CaseType enum value to use for the current episode."""
        if self.current_episode <= 100:
            return CaseType.BNS_318_BAIL.value
        elif self.current_episode <= 300:
            return CaseType.PMLA_BAIL.value
        else:
            return CaseType.UAPA_43D_BAIL.value

    def get_prompt_context(self) -> str:
        """Returns a system prompt context dynamically adjusting to the complexity."""
        case = self.get_current_case_type()
        
        if case == CaseType.BNS_318_BAIL.value:
            return "You are an Indian bail court judge. This is a standard BNS 318 fraud case. Apply BNSS 480 criteria (flight risk, tampering) to decide."
        elif case == CaseType.PMLA_BAIL.value:
            return "You are an Indian bail court judge. This is a complex PMLA case. You MUST apply the Section 45 Twin Test and cite the 'Vijay Madanlal' precedent."
        elif case == CaseType.UAPA_43D_BAIL.value:
            return "You are an Indian bail court judge. This is a highly severe UAPA terror funding case. The threshold for bail is extreme under Section 43D(5). Cite 'K.A. Najeeb'."
        
        return "You are an Indian bail court judge. Apply the law impartially."
