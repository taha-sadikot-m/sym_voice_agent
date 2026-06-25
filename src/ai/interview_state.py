"""
LangGraph State Definition for Interview Agent — v2
"""
from typing import TypedDict, List, Optional, Dict, Any


class InterviewState(TypedDict):
    """State for the interview agent workflow"""

    # Session configuration
    session_id: str
    company: str
    role: str
    difficulty_tier: str
    mock_mode: str
    job_description: str
    resume_text: str
    resume_parsed: Dict[str, Any]

    # Derived tier/mode context
    tier_label: str
    tier_target_user: str
    tier_question_style: str
    tier_comfort_goal: str
    mock_mode_label: str
    mock_mode_focus: str

    # Persona
    ai_persona_name: Optional[str]
    ai_persona_title: Optional[str]
    ai_introduction: Optional[str]

    # Full conversation thread
    messages: List[Dict[str, Any]]
    questions_asked: List[Dict[str, Any]]

    # Interviewer working memory
    current_topic: Optional[str]
    pending_follow_ups: List[str]
    interviewer_hypothesis: Optional[str]
    candidate_verbosity: Optional[str]
    resume_anchors: List[str]
    last_acknowledgment: Optional[str]
    topics_covered: List[str]

    # Evaluation tracking
    competencies_evaluated: Dict[str, float]
    identified_strengths: List[str]
    identified_weaknesses: List[str]
    red_flags: List[str]
    positive_signals: List[str]

    # Flow control
    question_count: int
    interview_duration: int
    should_continue: bool
    termination_reason: Optional[str]
    current_question: Optional[str]
    awaiting_response: bool
    sufficient_signal: bool
    needs_follow_up: bool
    follow_up_context: Optional[str]

    # Output
    final_analysis: Optional[Dict[str, Any]]
