"""
Next-Generation AI Interviewer
Provides step-by-step interview methods called directly from views.
"""

# PROMPT ENGINEERING STANDARDS FOR THIS CODEBASE
# ===============================================
# 1. Always inject the persona name into AI-role prompts.
# 2. Always request structured JSON for multi-field responses.
# 3. Always include role context, resume anchors, history, interviewer state, then task.
# 4. Use different creativity levels by task type.
# 5. Never drop conversation memory entirely; compress older turns instead.
# 6. Ground questions in resume anchors whenever available.
# 7. Never use placeholders in user-facing text.

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from decouple import config
from langchain_google_genai import ChatGoogleGenerativeAI

from .interview_context import analyze_job_description, get_mode_context, get_tier_context
from .interview_state import InterviewState

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 5
MAX_QUESTIONS = 15
EMA_ALPHA = 0.6


class InterviewAgent:
    """AI-powered interview agent with adaptive, stateful interviewing."""

    def __init__(self):
        api_key = config('GEMINI_API_KEY', default='')
        if not api_key:
            logger.error('GEMINI_API_KEY not found in environment')
            raise ValueError('GEMINI_API_KEY is required for InterviewAgent')

        # thinking_budget=0 disables Gemini 2.5-flash's extended thinking mode.
        # Thinking adds 5-30s per call — we don't need chain-of-thought for structured JSON tasks.
        self.llm = ChatGoogleGenerativeAI(
            model='models/gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.7,
            thinking_budget=0,
        )
        self.creative_llm = ChatGoogleGenerativeAI(
            model='models/gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.8,
            thinking_budget=0,
        )
        self.eval_llm = ChatGoogleGenerativeAI(
            model='models/gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.3,
            thinking_budget=0,
        )
        self.analysis_llm = ChatGoogleGenerativeAI(
            model='models/gemini-2.5-flash',
            google_api_key=api_key,
            temperature=0.5,
            thinking_budget=0,
        )

        logger.info('InterviewAgent initialized successfully')

    def initialize_state(self, session_data: Dict[str, Any]) -> InterviewState:
        """Initialize interview state from session data."""
        tier = session_data.get('difficulty_tier', 'EASY')
        mode = session_data.get('mock_mode', 'RECRUITER_WARMUP')
        tier_ctx = get_tier_context(tier)
        mode_ctx = get_mode_context(mode)
        resume_text = session_data.get('resume_text', '')
        # Anchor extraction is deferred to the view (run in parallel with introduce()).
        # Any pre-extracted anchors passed in session_data are used directly.
        resume_anchors = session_data.get('resume_anchors', [])

        return {
            'session_id': session_data.get('session_id'),
            'company': session_data.get('company', 'the company'),
            'role': session_data.get('role', 'this position'),
            'difficulty_tier': tier,
            'mock_mode': mode,
            'job_description': session_data.get('job_description', ''),
            'resume_text': resume_text,
            'resume_parsed': session_data.get('resume_parsed', {}),
            'tier_label': tier_ctx['label'],
            'tier_target_user': tier_ctx['target_user'],
            'tier_question_style': tier_ctx['question_style'],
            'tier_comfort_goal': tier_ctx['comfort_goal'],
            'mock_mode_label': mode_ctx['label'],
            'mock_mode_focus': mode_ctx['focus'],
            'ai_persona_name': None,
            'ai_persona_title': None,
            'ai_introduction': None,
            'messages': [],
            'questions_asked': [],
            'current_topic': None,
            'pending_follow_ups': [],
            'interviewer_hypothesis': 'neutral',
            'candidate_verbosity': None,
            'resume_anchors': resume_anchors,
            'last_acknowledgment': None,
            'topics_covered': [],
            'competencies_evaluated': {},
            'identified_strengths': [],
            'identified_weaknesses': [],
            'red_flags': [],
            'positive_signals': [],
            'question_count': 0,
            'interview_duration': 0,
            'should_continue': True,
            'termination_reason': None,
            'current_question': None,
            'awaiting_response': False,
            'sufficient_signal': False,
            'needs_follow_up': False,
            'follow_up_context': None,
            'final_analysis': None,
        }

    def introduce(self, state: InterviewState) -> InterviewState:
        """Generate a professional interviewer persona and introduction."""
        state = self.ensure_state_defaults(state)
        logger.info(f"Generating introduction for {state['company']} - {state['role']}")

        prompt = f"""You are creating the interviewer persona for a mock interview.

Company: {state['company']}
Role: {state['role']}
Difficulty Tier: {state['tier_label']} ({state['tier_target_user']})
Tone Goal: {state['tier_comfort_goal']}
Mock Mode: {state['mock_mode_label']}
Mode Focus: {state['mock_mode_focus']}

Return your response as a JSON object with this exact structure:
{{
  "name": "<realistic first + last name for the company culture>",
  "title": "<your job title at the company>",
  "introduction": "<the full introduction text, under 120 words, ending by asking if they're ready>"
}}

The introduction text must NOT use placeholders like [Name]. Address the candidate directly.
Respond ONLY with valid JSON, nothing else."""

        try:
            response_data = self._invoke_json(prompt, purpose='creative', default={})
            persona_name = response_data.get('name', 'Hiring Manager')
            persona_title = response_data.get('title', 'Hiring Manager')
            introduction = response_data.get(
                'introduction',
                f"Hello, I'm the hiring manager for the {state['role']} role at {state['company']}. We'll keep this conversational and practical so I can understand how you think and work. Are you ready to begin?",
            )

            state['ai_persona_name'] = persona_name
            state['ai_persona_title'] = persona_title
            state['ai_introduction'] = introduction
            state['messages'].append({
                'role': 'ai',
                'content': introduction,
                'type': 'introduction',
                'timestamp': datetime.now().isoformat(),
            })
            logger.info(f"Introduction generated successfully. Persona: {persona_name} ({persona_title})")
        except Exception as e:
            logger.error(f'Error generating introduction: {str(e)}')
            fallback = (
                f"Hello, I'm your interviewer for the {state['role']} role at {state['company']}. "
                f"We'll keep this practical and conversational so you can explain your real experience. Are you ready to begin?"
            )
            state['ai_persona_name'] = 'Hiring Manager'
            state['ai_persona_title'] = 'Hiring Manager'
            state['ai_introduction'] = fallback
            state['messages'].append({
                'role': 'ai',
                'content': fallback,
                'type': 'introduction',
                'timestamp': datetime.now().isoformat(),
            })

        return state

    def generate_question(self, state: InterviewState) -> InterviewState:
        """Generate the next main interview question."""
        state = self.ensure_state_defaults(state)
        logger.info(f"Generating question #{state['question_count'] + 1} for session {state['session_id']}")

        history_summary = self._summarize_history(state)
        competency_gaps = self._identify_gaps(state)
        jd_analysis = analyze_job_description(state.get('job_description', ''))
        anchors = state.get('resume_anchors', [])
        anchor_block = '\n'.join(f'- {anchor}' for anchor in anchors) if anchors else '- (No specific anchors available)'
        strengths = ', '.join(state.get('identified_strengths', [])[-4:]) or 'None yet'
        weaknesses = ', '.join(state.get('identified_weaknesses', [])[-3:]) or 'None yet'

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
You are interviewing the candidate for the {state['role']} role.

ROLE AND COMPANY CONTEXT:
- Company: {state['company']}
- Role: {state['role']}
- Difficulty Tier: {state['tier_label']} ({state['tier_target_user']})
- Question Style: {state['tier_question_style']}
- Tone Goal: {state['tier_comfort_goal']}
- Mock Mode: {state['mock_mode_label']}
- Mode Focus: {state['mock_mode_focus']}

JOB DESCRIPTION:
{state['job_description'][:1500] if state['job_description'] else 'Not provided'}
{jd_analysis.get('key_skills', '')}
{jd_analysis.get('key_requirements', '')}

CANDIDATE BACKGROUND — RESUME ANCHORS:
Your question MUST reference at least one of the following if it is relevant to the competency you're exploring:
{anchor_block}

CONVERSATION HISTORY:
{history_summary}

CURRENT INTERVIEWER STATE:
- Topics already covered: {', '.join(state['topics_covered']) if state['topics_covered'] else 'None yet'}
- Competency gaps to explore: {', '.join(competency_gaps) if competency_gaps else 'cover remaining evidence gaps'}
- Interviewer hypothesis: {state.get('interviewer_hypothesis', 'neutral')}
- Candidate verbosity: {state.get('candidate_verbosity', 'unknown')}
- Strengths seen: {strengths}
- Weaknesses seen: {weaknesses}

RULES:
- Do NOT revisit previously covered topics.
- Ask one clear, natural question only.
- If leaning_yes: ask a slightly tougher verifying question.
- If leaning_no: give the candidate a fair chance in an uncovered area.
- If the candidate is terse, end with a specific prompt like "Walk me through exactly what you did."
- If the candidate is verbose, keep the question tightly scoped.
- Keep the question under 35 words when possible.

Return ONLY a JSON object:
{{
  "question": "<the interview question text>",
  "topic": "<2-4 word topic tag>",
  "target_competency": "<which competency this question assesses>"
}}"""

        try:
            question_data = self._invoke_json(prompt, purpose='question', default={})
            question = (question_data.get('question') or '').strip()
            topic = (question_data.get('topic') or 'general fit').strip()
            target_competency = (question_data.get('target_competency') or 'communication').strip()
            if not question:
                raise ValueError('Empty question returned by LLM')
        except Exception as e:
            logger.error(f'Error generating question: {str(e)}')
            question, topic, target_competency = self._build_fallback_question(state, competency_gaps)

        state['current_question'] = question
        state['question_count'] = state.get('question_count', 0) + 1
        state['current_topic'] = topic
        if topic and topic not in state['topics_covered']:
            state['topics_covered'].append(topic)

        timestamp = datetime.now().isoformat()
        state['messages'].append({
            'role': 'ai',
            'content': question,
            'type': 'question',
            'question_number': state['question_count'],
            'topic': topic,
            'timestamp': timestamp,
        })
        state['questions_asked'].append({
            'question': question,
            'question_number': state['question_count'],
            'topic': topic,
            'target_competency': target_competency,
            'timestamp': timestamp,
        })
        state['awaiting_response'] = True

        logger.info(f"Question #{state['question_count']} generated successfully on topic: {topic}")
        return state

    def generate_acknowledgment(self, state: InterviewState) -> InterviewState:
        """Generate a short, specific reaction to the latest answer."""
        state = self.ensure_state_defaults(state)
        if not state['questions_asked']:
            return state

        last_qa = state['questions_asked'][-1]
        answer = last_qa.get('answer', '')
        score = last_qa.get('score', 5)
        last_ack = state.get('last_acknowledgment', '')
        if not answer:
            return state

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
A candidate just answered your interview question. Write a brief, genuine 1-2 sentence reaction.

YOUR QUESTION WAS: {last_qa.get('question', '')}
THEIR ANSWER: {answer[:400]}
ANSWER QUALITY (internal only): {score}/10

RULES:
- Reference one specific thing they said.
- Do NOT use generic phrases like "Great answer" or "Thanks for sharing".
- Do NOT ask the next question yet.
- If score < 5, stay neutral and professional.
- If score >= 7, sound genuinely interested.
- Keep it conversational and different from this previous acknowledgment: {last_ack}

Respond with ONLY the acknowledgment text."""

        try:
            response = self._get_llm('creative').invoke(prompt)
            acknowledgment = str(response.content).strip()
            if acknowledgment and acknowledgment != last_ack:
                state['last_acknowledgment'] = acknowledgment
                state['messages'].append({
                    'role': 'ai',
                    'content': acknowledgment,
                    'type': 'acknowledgment',
                    'timestamp': datetime.now().isoformat(),
                })
        except Exception as e:
            logger.error(f'Error generating acknowledgment: {e}')

        return state

    def generate_followup_question(self, state: InterviewState) -> InterviewState:
        """Generate a targeted follow-up question when the last answer lacked depth."""
        state = self.ensure_state_defaults(state)
        last_qa = state['questions_asked'][-1] if state['questions_asked'] else {}
        context = state.get('follow_up_context') or (state.get('pending_follow_ups') or ['The answer lacked specifics.'])[0]

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
You just asked a question and the answer needs more depth. Ask ONE targeted follow-up.

YOUR ORIGINAL QUESTION: {last_qa.get('question', '')}
CANDIDATE'S ANSWER: {last_qa.get('answer', '')[:500]}
WHY A FOLLOW-UP IS NEEDED: {context}

FOLLOW-UP RULES:
- Do NOT repeat the original question.
- Reference something specific the candidate said.
- Push for the missing outcome, metric, decision, or ownership.
- Keep it under 30 words.
- Sound natural and curious.

Respond with ONLY the follow-up question text."""

        try:
            response = self._get_llm('question').invoke(prompt)
            question = str(response.content).strip().strip('"').strip("'")
            if not question:
                raise ValueError('Empty follow-up question')
        except Exception as e:
            logger.error(f'Error generating follow-up: {e}')
            question = 'Can you walk me through your exact role, the decision you made, and the result?'

        state['current_question'] = question
        state['question_count'] = state.get('question_count', 0) + 1
        state['needs_follow_up'] = False
        state['follow_up_context'] = None
        if state.get('pending_follow_ups'):
            state['pending_follow_ups'] = state['pending_follow_ups'][1:]

        timestamp = datetime.now().isoformat()
        state['messages'].append({
            'role': 'ai',
            'content': question,
            'type': 'followup_question',
            'question_number': state['question_count'],
            'timestamp': timestamp,
        })
        state['questions_asked'].append({
            'question': question,
            'question_number': state['question_count'],
            'is_followup': True,
            'topic': state.get('current_topic'),
            'timestamp': timestamp,
        })
        state['awaiting_response'] = True
        return state

    def evaluate_answer(self, state: InterviewState) -> InterviewState:
        """Evaluate the latest answer and update the working interview state."""
        state = self.ensure_state_defaults(state)
        user_messages = [message for message in state.get('messages', []) if message.get('role') == 'user']
        if not user_messages:
            logger.warning('No user answer found to evaluate')
            return state

        user_answer = user_messages[-1].get('content', '')
        question = state.get('current_question') or (state['questions_asked'][-1].get('question', '') if state['questions_asked'] else '')
        logger.info(f"Evaluating answer for question #{state['question_count']} in session {state['session_id']}")

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
You are evaluating one answer in an interview for the {state['role']} role.

ROLE AND COMPANY CONTEXT:
- Company: {state['company']}
- Role: {state['role']}
- Difficulty Tier: {state['tier_label']} ({state['tier_target_user']})
- Mock Mode: {state['mock_mode_label']}
- Current topic: {state.get('current_topic', 'unknown')}

CANDIDATE BACKGROUND — RESUME ANCHORS:
{chr(10).join(f'- {anchor}' for anchor in state.get('resume_anchors', [])) if state.get('resume_anchors') else '- No resume anchors available'}

CONVERSATION HISTORY:
{self._summarize_history(state)}

CURRENT INTERVIEWER STATE:
- Hypothesis: {state.get('interviewer_hypothesis', 'neutral')}
- Candidate verbosity so far: {state.get('candidate_verbosity', 'unknown')}
- Topics covered: {', '.join(state.get('topics_covered', [])) if state.get('topics_covered') else 'None'}
- Question count: {state.get('question_count', 0)}

QUESTION: {question}
ANSWER: {user_answer}

Return ONLY valid JSON with this exact schema:
{{
  "score": <0-10>,
  "feedback": "<2-3 sentences>",
  "positive_signals": ["..."],
  "negative_signals": ["..."],
  "competencies": {{
    "technical_skills": <0.0-1.0>,
    "communication": <0.0-1.0>,
    "problem_solving": <0.0-1.0>,
    "leadership": <0.0-1.0>,
    "domain_knowledge": <0.0-1.0>,
    "cultural_fit": <0.0-1.0>
  }},
  "red_flags": ["..."],
  "needs_follow_up": <true/false>,
  "follow_up_context": "<exactly what was vague or missing>",
  "sufficient_signal": <true/false>,
  "interviewer_hypothesis": "<strong_yes | leaning_yes | neutral | leaning_no | strong_no>",
  "candidate_verbosity": "<terse | moderate | verbose>"
}}"""

        try:
            eval_data = self._invoke_json(prompt, purpose='evaluation', default={})
            eval_data = self._normalize_evaluation(eval_data, state, user_answer)
        except Exception as e:
            logger.error(f'Error evaluating answer: {str(e)}')
            eval_data = self._fallback_evaluation(state, user_answer)

        self._extend_unique(state['identified_strengths'], eval_data.get('positive_signals', []))
        self._extend_unique(state['identified_weaknesses'], eval_data.get('negative_signals', []))
        self._extend_unique(state['red_flags'], eval_data.get('red_flags', []))
        self._extend_unique(state['positive_signals'], eval_data.get('positive_signals', []))

        for comp, score in eval_data.get('competencies', {}).items():
            if score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue

            current = state['competencies_evaluated'].get(comp)
            if current is not None:
                state['competencies_evaluated'][comp] = round((EMA_ALPHA * score) + ((1 - EMA_ALPHA) * float(current)), 4)
            else:
                state['competencies_evaluated'][comp] = round(score, 4)

        last_question = state['questions_asked'][-1] if state['questions_asked'] else None
        needs_follow_up = bool(eval_data.get('needs_follow_up', False))
        if last_question and last_question.get('is_followup'):
            needs_follow_up = False

        state['needs_follow_up'] = needs_follow_up
        state['follow_up_context'] = eval_data.get('follow_up_context') if needs_follow_up else None
        state['sufficient_signal'] = bool(eval_data.get('sufficient_signal', False)) and state['question_count'] >= MIN_QUESTIONS
        state['interviewer_hypothesis'] = eval_data.get('interviewer_hypothesis', state.get('interviewer_hypothesis', 'neutral'))

        if state['question_count'] >= 2 and (not state.get('candidate_verbosity') or state.get('candidate_verbosity') == 'unknown'):
            state['candidate_verbosity'] = eval_data.get('candidate_verbosity', 'moderate')
        elif not state.get('candidate_verbosity'):
            state['candidate_verbosity'] = eval_data.get('candidate_verbosity', 'moderate')

        if state['needs_follow_up'] and state['follow_up_context']:
            state['pending_follow_ups'].append(state['follow_up_context'])

        if last_question is not None:
            last_question['evaluation'] = eval_data
            last_question['answer'] = user_answer
            last_question['score'] = eval_data.get('score', 5)

        state['awaiting_response'] = False
        logger.info(f"Answer evaluated. Score: {eval_data.get('score', 'N/A')}/10")
        return state

    def should_continue_interview(self, state: InterviewState) -> InterviewState:
        """Decide whether to continue without a separate LLM call."""
        state = self.ensure_state_defaults(state)
        question_count = state.get('question_count', 0)
        logger.info(f"Evaluating continuation for session {state['session_id']}. Questions asked: {question_count}")

        if state.get('termination_reason') == 'user_ended':
            state['should_continue'] = False
            return state

        if question_count >= MAX_QUESTIONS:
            state['should_continue'] = False
            state['termination_reason'] = 'max_questions'
            return state

        if question_count < MIN_QUESTIONS:
            state['should_continue'] = True
            return state

        if state.get('sufficient_signal', False):
            state['should_continue'] = False
            state['termination_reason'] = 'ai_decided'
            return state

        state['should_continue'] = True
        return state

    def evaluate_and_next(self, state: InterviewState) -> InterviewState:
        """Single LLM call: evaluate answer + acknowledgment + next question.

        Replaces three sequential calls:
          evaluate_answer() + generate_acknowledgment() + generate_question/generate_followup_question()

        Reduces per-answer Gemini round-trips from 3 → 1, cutting latency by ~4-12 s.
        Falls back to the original three separate calls on any error.
        """
        state = self.ensure_state_defaults(state)
        user_messages = [m for m in state.get('messages', []) if m.get('role') == 'user']
        if not user_messages:
            logger.warning('No user answer found in evaluate_and_next')
            return state

        user_answer = user_messages[-1].get('content', '')
        question = state.get('current_question') or (
            state['questions_asked'][-1].get('question', '') if state['questions_asked'] else ''
        )
        last_qa = state['questions_asked'][-1] if state['questions_asked'] else None
        is_last_followup = bool(last_qa and last_qa.get('is_followup'))
        last_ack = state.get('last_acknowledgment', '')

        history_summary = self._summarize_history(state)
        competency_gaps = self._identify_gaps(state)
        anchors = state.get('resume_anchors', [])
        anchor_block = '\n'.join(f'- {a}' for a in anchors) if anchors else '- (No specific anchors available)'
        strengths = ', '.join(state.get('identified_strengths', [])[-4:]) or 'None yet'
        weaknesses = ', '.join(state.get('identified_weaknesses', [])[-3:]) or 'None yet'
        topic_block = ', '.join(state.get('topics_covered', [])) or 'None yet'

        logger.info(f"evaluate_and_next: single-call for session {state['session_id']}, question #{state['question_count']}")

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
You are conducting a mock interview for the {state['role']} role. Complete three tasks in one response.

=== TASK 1: EVALUATE THE ANSWER ===
ROLE CONTEXT:
- Company: {state['company']}
- Role: {state['role']}
- Difficulty Tier: {state['tier_label']} ({state['tier_target_user']})
- Mock Mode: {state['mock_mode_label']}
- Current topic: {state.get('current_topic', 'unknown')}
- Questions asked so far: {state['question_count']}

CANDIDATE BACKGROUND:
{anchor_block}

CONVERSATION HISTORY (summarized):
{history_summary}

QUESTION ASKED: {question}
CANDIDATE'S ANSWER: {user_answer[:500]}

CURRENT INTERVIEWER STATE:
- Hypothesis: {state.get('interviewer_hypothesis', 'neutral')}
- Candidate verbosity: {state.get('candidate_verbosity', 'unknown')}
- Topics covered: {topic_block}
- Competency gaps: {', '.join(competency_gaps) if competency_gaps else 'none'}
- Strengths seen: {strengths}
- Weaknesses seen: {weaknesses}

=== TASK 2: BRIEF ACKNOWLEDGMENT ===
Write 1-2 sentences reacting naturally to the answer. Reference something specific they said.
Rules:
- No generic phrases ("Great answer", "Thanks for sharing").
- MUST NOT end with a question mark. MUST NOT contain any question at all.
- Statements only — observations or reactions, never interrogative.
- Must be different from this previous acknowledgment: "{last_ack}"

=== TASK 3: GENERATE NEXT QUESTION ===
{"This was already a follow-up question — generate a fresh question on a new topic (next_question_type: fresh)." if is_last_followup else ""}
Topics already covered: {topic_block}
Competency gaps to explore: {', '.join(competency_gaps) if competency_gaps else 'cover remaining evidence gaps'}
Resume anchors (reference where relevant): {anchor_block}

Rules for next_question:
- If needs_follow_up is true: targeted follow-up on same topic (next_question_type: "follow_up")
- If needs_follow_up is false: fresh question on an uncovered competency (next_question_type: "fresh")
- Keep the question under 35 words
- Reference a resume anchor if relevant
- Do NOT repeat any previously covered topic

Return ONLY valid JSON:
{{
  "score": <0-10>,
  "feedback": "<2-3 sentences evaluating the answer>",
  "positive_signals": ["<concrete strength observed>"],
  "negative_signals": ["<gap or weakness observed>"],
  "red_flags": ["<any concern>"],
  "competencies": {{
    "technical_skills": <0.0-1.0>,
    "communication": <0.0-1.0>,
    "problem_solving": <0.0-1.0>,
    "leadership": <0.0-1.0>,
    "domain_knowledge": <0.0-1.0>,
    "cultural_fit": <0.0-1.0>
  }},
  "needs_follow_up": <true/false>,
  "follow_up_context": "<what was vague or missing>",
  "sufficient_signal": <true/false>,
  "interviewer_hypothesis": "<strong_yes|leaning_yes|neutral|leaning_no|strong_no>",
  "candidate_verbosity": "<terse|moderate|verbose>",
  "acknowledgment": "<1-2 sentence natural reaction>",
  "next_question": "<the next question text>",
  "next_question_type": "<follow_up|fresh>",
  "next_topic": "<2-4 word topic tag>",
  "next_competency": "<which competency this explores>"
}}"""

        try:
            data = self._invoke_json(prompt, purpose='evaluation', default={})
            data = self._normalize_combined(data, state, user_answer)
        except Exception as e:
            logger.error(f'evaluate_and_next LLM call failed: {e}. Falling back to separate calls.')
            state = self.evaluate_answer(state)
            state = self.generate_acknowledgment(state)
            follow_up_needed = state.get('needs_follow_up', False) and not is_last_followup
            if follow_up_needed:
                state = self.generate_followup_question(state)
            else:
                state = self.generate_question(state)
            return state

        # === Apply evaluation results (mirrors evaluate_answer) ===
        self._extend_unique(state['identified_strengths'], data.get('positive_signals', []))
        self._extend_unique(state['identified_weaknesses'], data.get('negative_signals', []))
        self._extend_unique(state['red_flags'], data.get('red_flags', []))
        self._extend_unique(state['positive_signals'], data.get('positive_signals', []))

        for comp, score in data.get('competencies', {}).items():
            if score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            current = state['competencies_evaluated'].get(comp)
            if current is not None:
                state['competencies_evaluated'][comp] = round(
                    (EMA_ALPHA * score) + ((1 - EMA_ALPHA) * float(current)), 4
                )
            else:
                state['competencies_evaluated'][comp] = round(score, 4)

        needs_follow_up = bool(data.get('needs_follow_up', False))
        if is_last_followup:
            needs_follow_up = False  # never double-follow-up

        state['needs_follow_up'] = needs_follow_up
        state['follow_up_context'] = data.get('follow_up_context') if needs_follow_up else None
        state['sufficient_signal'] = bool(data.get('sufficient_signal', False)) and state['question_count'] >= MIN_QUESTIONS
        state['interviewer_hypothesis'] = data.get('interviewer_hypothesis', state.get('interviewer_hypothesis', 'neutral'))

        if state['question_count'] >= 2 and (not state.get('candidate_verbosity') or state.get('candidate_verbosity') == 'unknown'):
            state['candidate_verbosity'] = data.get('candidate_verbosity', 'moderate')
        elif not state.get('candidate_verbosity'):
            state['candidate_verbosity'] = data.get('candidate_verbosity', 'moderate')

        if needs_follow_up and state['follow_up_context']:
            state['pending_follow_ups'].append(state['follow_up_context'])

        if last_qa is not None:
            last_qa['evaluation'] = data
            last_qa['answer'] = user_answer
            last_qa['score'] = data.get('score', 5)

        state['awaiting_response'] = False
        logger.info(f"Answer evaluated. Score: {data.get('score', 'N/A')}/10")

        # === Apply acknowledgment (mirrors generate_acknowledgment) ===
        acknowledgment = (data.get('acknowledgment') or '').strip()
        if acknowledgment and acknowledgment != last_ack:
            state['last_acknowledgment'] = acknowledgment
            state['messages'].append({
                'role': 'ai',
                'content': acknowledgment,
                'type': 'acknowledgment',
                'timestamp': datetime.now().isoformat(),
            })

        # === Apply next question (mirrors generate_question / generate_followup_question) ===
        next_question = (data.get('next_question') or '').strip()
        next_question_type = data.get('next_question_type', 'fresh')
        next_topic = (data.get('next_topic') or state.get('current_topic') or 'general fit').strip()
        next_competency = (data.get('next_competency') or 'communication').strip()
        is_followup_question = (next_question_type == 'follow_up') and needs_follow_up

        if not next_question:
            next_question, next_topic, next_competency = self._build_fallback_question(state, self._identify_gaps(state))
            is_followup_question = False

        state['current_question'] = next_question
        state['question_count'] = state.get('question_count', 0) + 1

        if not is_followup_question:
            state['current_topic'] = next_topic
            if next_topic and next_topic not in state['topics_covered']:
                state['topics_covered'].append(next_topic)

        if is_followup_question:
            state['needs_follow_up'] = False
            state['follow_up_context'] = None
            if state.get('pending_follow_ups'):
                state['pending_follow_ups'] = state['pending_follow_ups'][1:]

        timestamp = datetime.now().isoformat()
        msg_type = 'followup_question' if is_followup_question else 'question'
        state['messages'].append({
            'role': 'ai',
            'content': next_question,
            'type': msg_type,
            'question_number': state['question_count'],
            'topic': next_topic,
            'timestamp': timestamp,
        })
        state['questions_asked'].append({
            'question': next_question,
            'question_number': state['question_count'],
            'topic': next_topic,
            'target_competency': next_competency,
            'is_followup': is_followup_question,
            'timestamp': timestamp,
        })
        state['awaiting_response'] = True

        logger.info(
            f"Next {'follow-up ' if is_followup_question else ''}question "
            f"#{state['question_count']} generated on topic: {next_topic}"
        )
        return state

    def generate_final_analysis(self, state: InterviewState) -> InterviewState:
        """Generate a comprehensive final analysis for the completed interview."""
        state = self.ensure_state_defaults(state)
        logger.info(f"Generating final analysis for session {state['session_id']}")
        qa_summary = self._build_qa_summary(state)

        prompt = f"""You are {state.get('ai_persona_name') or 'the interviewer'}, {state.get('ai_persona_title') or 'Hiring Manager'} at {state['company']}.
Provide a final interview assessment for the {state['role']} role.

INTERVIEW SUMMARY:
- Questions asked: {state['question_count']}
- Termination reason: {state.get('termination_reason', 'Interview completed')}
- Difficulty Tier: {state['tier_label']}
- Mock Mode: {state['mock_mode_label']}
- Interviewer final hypothesis: {state.get('interviewer_hypothesis', 'neutral')}
- Candidate communication style: {state.get('candidate_verbosity', 'unknown')}

CANDIDATE BACKGROUND — RESUME ANCHORS:
{chr(10).join(f'- {anchor}' for anchor in state.get('resume_anchors', [])) if state.get('resume_anchors') else '- No resume anchors available'}

QUESTIONS AND ANSWERS:
{qa_summary}

COMPETENCIES EVALUATED:
{json.dumps(state['competencies_evaluated'], indent=2)}

OBSERVATIONS:
Strengths: {', '.join(state.get('identified_strengths', [])[:10]) or 'None'}
Weaknesses: {', '.join(state.get('identified_weaknesses', [])[:10]) or 'None'}
Red Flags: {', '.join(state.get('red_flags', [])) or 'None'}

Return ONLY valid JSON:
{{
  "overall_score": <0-100 integer>,
  "verdict": "<STRONG_YES | YES | MAYBE | NO | STRONG_NO>",
  "summary": "<2-3 sentence overall assessment>",
  "strengths": ["..."],
  "weaknesses": ["..."],
  "competency_breakdown": {{
    "technical_skills": {{"score": <0-10>, "notes": "<brief comment>"}},
    "communication": {{"score": <0-10>, "notes": "<brief comment>"}},
    "problem_solving": {{"score": <0-10>, "notes": "<brief comment>"}},
    "cultural_fit": {{"score": <0-10>, "notes": "<brief comment>"}}
  }},
  "improvement_suggestions": ["..."],
  "detailed_feedback": "<paragraph of detailed feedback>",
  "communication_style_notes": "<specific observation about how they communicated>",
  "standout_moment": "<the single most impressive or concerning moment, with quote>",
  "interview_completeness": "<COMPLETE | PARTIAL | INCOMPLETE>",
  "completeness_note": "<why complete or partial>"
}}"""

        try:
            analysis = self._invoke_json(prompt, purpose='analysis', default={})
            state['final_analysis'] = analysis
            logger.info(
                f"Final analysis generated. Verdict: {analysis.get('verdict', 'N/A')}, "
                f"Score: {analysis.get('overall_score', 'N/A')}/100"
            )
        except Exception as e:
            logger.error(f'Error generating final analysis: {str(e)}')
            state['final_analysis'] = self._fallback_final_analysis(state)

        return state

    def ensure_state_defaults(self, state: Dict[str, Any]) -> InterviewState:
        """Backfill missing keys for older saved interview states."""
        tier = state.get('difficulty_tier', 'EASY')
        mode = state.get('mock_mode', 'RECRUITER_WARMUP')
        tier_ctx = get_tier_context(tier)
        mode_ctx = get_mode_context(mode)

        state.setdefault('session_id', '')
        state.setdefault('company', 'the company')
        state.setdefault('role', 'this position')
        state.setdefault('difficulty_tier', tier)
        state.setdefault('mock_mode', mode)
        state.setdefault('job_description', '')
        state.setdefault('resume_text', '')
        state.setdefault('resume_parsed', {})
        state.setdefault('tier_label', tier_ctx['label'])
        state.setdefault('tier_target_user', tier_ctx['target_user'])
        state.setdefault('tier_question_style', tier_ctx['question_style'])
        state.setdefault('tier_comfort_goal', tier_ctx['comfort_goal'])
        state.setdefault('mock_mode_label', mode_ctx['label'])
        state.setdefault('mock_mode_focus', mode_ctx['focus'])
        state.setdefault('ai_persona_name', None)
        state.setdefault('ai_persona_title', None)
        state.setdefault('ai_introduction', None)
        state.setdefault('messages', [])
        state.setdefault('questions_asked', [])
        state.setdefault('current_topic', None)
        state.setdefault('pending_follow_ups', [])
        state.setdefault('interviewer_hypothesis', 'neutral')
        state.setdefault('candidate_verbosity', None)
        state.setdefault('resume_anchors', self._extract_resume_anchors(state.get('resume_text', '')))
        state.setdefault('last_acknowledgment', None)
        state.setdefault('topics_covered', [])
        state.setdefault('competencies_evaluated', {})
        state.setdefault('identified_strengths', [])
        state.setdefault('identified_weaknesses', [])
        state.setdefault('red_flags', [])
        state.setdefault('positive_signals', [])
        state.setdefault('question_count', 0)
        state.setdefault('interview_duration', 0)
        state.setdefault('should_continue', True)
        state.setdefault('termination_reason', None)
        state.setdefault('current_question', None)
        state.setdefault('awaiting_response', False)
        state.setdefault('sufficient_signal', False)
        state.setdefault('needs_follow_up', False)
        state.setdefault('follow_up_context', None)
        state.setdefault('final_analysis', None)
        return state

    def _extract_resume_anchors(self, resume_text: str) -> List[str]:
        """Extract concrete facts from the resume for grounded questioning."""
        if not resume_text or len(resume_text.strip()) < 100:
            return []

        prompt = f"""Read this resume and extract 5-8 specific, concrete facts that an interviewer could reference in a question.
Focus on named projects, measurable outcomes, specific technologies, team sizes, company names, and notable achievements.

RESUME:
{resume_text[:2000]}

Return ONLY a JSON array of strings. Each string must be a complete factual statement."""

        try:
            data = self._invoke_json(prompt, purpose='evaluation', default=[])
            if isinstance(data, list):
                return [str(item).strip() for item in data if str(item).strip()][:8]
        except Exception as e:
            logger.warning(f'Could not extract resume anchors via LLM: {e}')

        lines = [line.strip(' -•\t') for line in resume_text.splitlines() if line.strip()]
        ranked = []
        for line in lines:
            if len(line) < 30:
                continue
            score = 0
            if re.search(r'\b\d+[%+xkKmM]?\b', line):
                score += 2
            if re.search(r'python|java|django|react|aws|docker|kubernetes|sql|api|fastapi|node', line, re.I):
                score += 1
            if re.search(r'led|built|designed|implemented|managed|launched|improved|scaled', line, re.I):
                score += 1
            if score > 0:
                ranked.append((score, line))

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [line for _, line in ranked[:6]]

    def _summarize_history(self, state: InterviewState) -> str:
        """Build a full conversation summary for prompt context."""
        if not state['questions_asked']:
            return 'No questions asked yet.'

        all_qa = state['questions_asked']
        total = len(all_qa)
        summary_parts = []

        for index, qa in enumerate(all_qa):
            q_num = qa.get('question_number', index + 1)
            question = qa.get('question', '')
            answer = qa.get('answer', 'No answer recorded')
            score = qa.get('score', 'N/A')
            is_followup = qa.get('is_followup', False)
            label = 'FOLLOW-UP' if is_followup else f'Q{q_num}'

            if index < total - 2:
                summary_parts.append(
                    f"{label}: {question}\n"
                    f"Answer (summary): {answer[:150]}{'...' if len(answer) > 150 else ''}\n"
                    f"Score: {score}/10"
                )
            else:
                summary_parts.append(
                    f"{label}: {question}\n"
                    f"Answer: {answer[:400]}{'...' if len(answer) > 400 else ''}\n"
                    f"Score: {score}/10"
                )

        return '\n\n---\n\n'.join(summary_parts)

    def _identify_gaps(self, state: InterviewState) -> List[str]:
        """Identify the key competencies still needing better evidence."""
        all_competencies = [
            'technical_skills',
            'communication',
            'problem_solving',
            'leadership',
            'domain_knowledge',
            'cultural_fit',
        ]
        gaps = []
        for competency in all_competencies:
            score = state['competencies_evaluated'].get(competency)
            if score is None:
                gaps.append(competency)
            elif score < 0.55:
                gaps.append(f'{competency} needs more evidence')
        return gaps[:3]

    def _build_qa_summary(self, state: InterviewState) -> str:
        if not state['questions_asked']:
            return 'No questions were asked.'

        parts = []
        for qa in state['questions_asked']:
            parts.append(
                f"Q{qa.get('question_number', '?')}: {qa.get('question', '')}\n"
                f"A: {qa.get('answer', 'Not answered')[:350]}{'...' if len(qa.get('answer', '')) > 350 else ''}\n"
                f"Score: {qa.get('score', 'N/A')}/10"
            )
        return '\n---\n'.join(parts)

    def _get_llm(self, purpose: str = 'default'):
        mapping = {
            'creative': getattr(self, 'creative_llm', None),
            'question': getattr(self, 'llm', None),
            'evaluation': getattr(self, 'eval_llm', None),
            'analysis': getattr(self, 'analysis_llm', None),
        }
        return mapping.get(purpose) or getattr(self, 'llm', None)

    def _strip_code_fences(self, content: str) -> str:
        if '```json' in content:
            return content.split('```json', 1)[1].split('```', 1)[0].strip()
        if '```' in content:
            return content.split('```', 1)[1].split('```', 1)[0].strip()
        return content.strip()

    def _invoke_json(self, prompt: str, purpose: str = 'default', default: Optional[Any] = None) -> Any:
        llm = self._get_llm(purpose)
        if llm is None:
            if default is not None:
                return default
            raise ValueError('No LLM configured for InterviewAgent')

        response = llm.invoke(prompt)
        content = self._strip_code_fences(str(response.content).strip())
        if not content:
            if default is not None:
                return default
            raise ValueError('Empty response from model')

        # Direct parse first (fast path — works when Gemini respects the format)
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Fallback: extract the first {...} block. Handles Gemini adding text
        # before/after the JSON object (e.g. "Here is the JSON:\n{...}\n")
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f'_invoke_json: could not parse JSON for purpose={purpose!r}, snippet={content[:120]!r}')
        if default is not None:
            return default
        raise ValueError(f'Could not parse JSON from model response')

    def _normalize_evaluation(self, eval_data: Dict[str, Any], state: InterviewState, user_answer: str) -> Dict[str, Any]:
        word_count = len(user_answer.split())
        eval_data['score'] = max(0, min(10, int(float(eval_data.get('score', 5)))))
        eval_data['feedback'] = eval_data.get('feedback') or 'The answer provided some signal, but more evidence would help.'
        eval_data['positive_signals'] = eval_data.get('positive_signals') or []
        eval_data['negative_signals'] = eval_data.get('negative_signals') or []
        eval_data['red_flags'] = eval_data.get('red_flags') or []
        eval_data['competencies'] = eval_data.get('competencies') or {}
        eval_data['follow_up_context'] = eval_data.get('follow_up_context') or 'The answer needs more specificity around actions and outcomes.'
        eval_data['interviewer_hypothesis'] = eval_data.get('interviewer_hypothesis') or 'neutral'
        eval_data['candidate_verbosity'] = eval_data.get('candidate_verbosity') or ('verbose' if word_count > 140 else 'terse' if word_count < 35 else 'moderate')
        return eval_data

    def _normalize_combined(self, data: Dict[str, Any], state: InterviewState, user_answer: str) -> Dict[str, Any]:
        """Normalize evaluate_and_next response fields."""
        word_count = len(user_answer.split())
        data['score'] = max(0, min(10, int(float(data.get('score', 5)))))
        data['feedback'] = data.get('feedback') or 'The answer provided some signal.'
        data['positive_signals'] = data.get('positive_signals') or []
        data['negative_signals'] = data.get('negative_signals') or []
        data['red_flags'] = data.get('red_flags') or []
        data['competencies'] = data.get('competencies') or {}
        data['follow_up_context'] = data.get('follow_up_context') or 'The answer needs more specificity.'
        data['interviewer_hypothesis'] = data.get('interviewer_hypothesis') or 'neutral'
        data['candidate_verbosity'] = data.get('candidate_verbosity') or (
            'verbose' if word_count > 140 else 'terse' if word_count < 35 else 'moderate'
        )
        # Hard guard: strip any sentence ending in '?' from the acknowledgment.
        # Gemini sometimes puts the next question inside the acknowledgment field.
        ack_raw = (data.get('acknowledgment') or '').strip()
        if ack_raw:
            sentences = re.split(r'(?<=[.!?])\s+', ack_raw)
            clean_sentences = [s for s in sentences if not s.rstrip().endswith('?')]
            ack_raw = ' '.join(clean_sentences).strip()
            # If stripping removed everything, keep just the first sentence regardless
            if not ack_raw and sentences:
                ack_raw = re.split(r'(?<=[.!?])\s+', (data.get('acknowledgment') or '').strip())[0]
        data['acknowledgment'] = ack_raw
        data['next_question'] = (data.get('next_question') or '').strip()
        data['next_question_type'] = data.get('next_question_type') or 'fresh'
        data['next_topic'] = (data.get('next_topic') or 'general fit').strip()
        data['next_competency'] = (data.get('next_competency') or 'communication').strip()
        return data

    def _fallback_evaluation(self, state: InterviewState, user_answer: str) -> Dict[str, Any]:
        word_count = len(user_answer.split())
        verbose_label = 'verbose' if word_count > 140 else 'terse' if word_count < 35 else 'moderate'
        score = 4 if word_count < 20 else 6 if word_count < 60 else 7
        needs_follow_up = word_count < 45 or not re.search(r'\b\d+\b', user_answer)

        return {
            'score': score,
            'feedback': 'The answer showed some relevant context, but stronger specifics and outcomes would improve the assessment.' if needs_follow_up else 'The answer was reasonably structured and gave usable signal for evaluation.',
            'positive_signals': ['Answered the question directly'] if user_answer else [],
            'negative_signals': ['Needs more detail and clearer ownership'] if needs_follow_up else [],
            'competencies': {
                'communication': 0.45 if needs_follow_up else 0.7,
                'problem_solving': 0.4 if needs_follow_up else 0.65,
            },
            'red_flags': [],
            'needs_follow_up': needs_follow_up,
            'follow_up_context': 'The answer did not clearly explain the candidate’s specific actions, decisions, or measurable outcomes.',
            'sufficient_signal': False,
            'interviewer_hypothesis': 'leaning_no' if needs_follow_up else 'neutral',
            'candidate_verbosity': verbose_label,
        }

    def _fallback_final_analysis(self, state: InterviewState) -> Dict[str, Any]:
        values = [float(value) for value in state['competencies_evaluated'].values() if value is not None]
        avg_score = sum(values) / len(values) if values else 0.55
        overall_score = int(avg_score * 100)
        if overall_score >= 80:
            verdict = 'YES'
        elif overall_score >= 65:
            verdict = 'MAYBE'
        else:
            verdict = 'NO'

        return {
            'overall_score': overall_score,
            'verdict': verdict,
            'summary': f"The interview produced {state['question_count']} questions worth of signal for the {state['role']} role.",
            'strengths': state.get('identified_strengths', [])[:5] or ['Stayed engaged throughout the interview'],
            'weaknesses': state.get('identified_weaknesses', [])[:4] or ['More evidence is needed in key competency areas'],
            'competency_breakdown': {
                'technical_skills': {'score': round(state['competencies_evaluated'].get('technical_skills', avg_score) * 10, 1), 'notes': 'Based on interview evidence gathered.'},
                'communication': {'score': round(state['competencies_evaluated'].get('communication', avg_score) * 10, 1), 'notes': 'Based on clarity and structure of answers.'},
                'problem_solving': {'score': round(state['competencies_evaluated'].get('problem_solving', avg_score) * 10, 1), 'notes': 'Based on examples and reasoning quality.'},
                'cultural_fit': {'score': round(state['competencies_evaluated'].get('cultural_fit', avg_score) * 10, 1), 'notes': 'Estimated from collaboration and ownership cues.'},
            },
            'improvement_suggestions': ['Use clearer STAR examples', 'Add measurable outcomes to each answer', 'Be explicit about your personal ownership'],
            'detailed_feedback': 'The interview generated enough signal to identify both strengths and areas that would benefit from more specificity.',
            'communication_style_notes': f"The candidate communicated in a {state.get('candidate_verbosity', 'moderate')} style.",
            'standout_moment': 'No single standout moment could be isolated confidently from the fallback path.',
            'interview_completeness': 'PARTIAL' if state.get('question_count', 0) < MIN_QUESTIONS else 'COMPLETE',
            'completeness_note': f"The interview ended after {state.get('question_count', 0)} questions.",
        }

    def _build_fallback_question(self, state: InterviewState, competency_gaps: List[str]):
        gap = competency_gaps[0] if competency_gaps else 'communication'
        question = f"Tell me about a project where you had to demonstrate strong {gap.replace('_', ' ')} in a real situation."
        if state.get('candidate_verbosity') == 'terse':
            question += ' Walk me through exactly what you did.'
        return question, gap.replace('_', ' '), gap

    def _extend_unique(self, target: List[str], values: List[str]) -> None:
        for value in values:
            cleaned = str(value).strip()
            if cleaned and cleaned not in target:
                target.append(cleaned)
