"""
Shared utilities for interview agent and AI service.
"""

from typing import Dict


def get_tier_context(tier: str) -> Dict[str, str]:
    mapping = {
        'EASY': {
            'label': 'Easy',
            'target_user': 'Undergraduate (UG) students',
            'question_style': 'Standard recruiter questions focused on comfort and confidence',
            'comfort_goal': 'Keep tone warm, supportive, and confidence-building',
        },
        'MEDIUM': {
            'label': 'Medium',
            'target_user': 'Postgraduate (PG) students',
            'question_style': 'Management trainee-level questions that are structured and analytical',
            'comfort_goal': 'Balance encouragement with structured and analytical probing',
        },
        'HARD': {
            'label': 'Hard',
            'target_user': 'Experienced professionals',
            'question_style': 'Industry-relevant, knowledge-based questions without in-depth theory',
            'comfort_goal': 'Stay rigorous, practical, and interview-real without becoming theoretical',
        },
    }
    return mapping.get(tier, mapping['EASY'])


def get_mode_context(mode: str) -> Dict[str, str]:
    mapping = {
        'RECRUITER_WARMUP': {
            'label': 'Recruiter Warm-up',
            'focus': 'Build rapport, communication comfort, and baseline readiness',
        },
        'BEHAVIORAL_STAR': {
            'label': 'Behavioral STAR',
            'focus': 'Test behavioral depth using STAR-style evidence and outcomes',
        },
        'SITUATIONAL_PROBLEM_SOLVING': {
            'label': 'Situational Problem-Solving',
            'focus': 'Assess structured thinking and decision-making in practical scenarios',
        },
        'ROLE_AND_INDUSTRY_APPLIED': {
            'label': 'Role and Industry Applied',
            'focus': 'Evaluate role-fit using industry-relevant, practical knowledge',
        },
        'EXECUTIVE_PRESENCE': {
            'label': 'Executive Presence',
            'focus': 'Assess clarity, stakeholder communication, ownership, and leadership presence',
        },
    }
    return mapping.get(mode, mapping['RECRUITER_WARMUP'])


def analyze_job_description(job_description: str) -> Dict[str, str]:
    """
    Extract lightweight interview context from a job description.
    """
    if not job_description or len(job_description.strip()) < 50:
        return {}

    jd_lower = job_description.lower()
    tech_keywords = [
        'python', 'java', 'javascript', 'react', 'node', 'aws', 'docker',
        'kubernetes', 'sql', 'nosql', 'mongodb', 'postgresql', 'redis',
        'microservices', 'api', 'rest', 'graphql', 'machine learning',
        'ai', 'cloud', 'devops', 'ci/cd', 'git', 'agile', 'scrum',
    ]

    found_skills = [skill for skill in tech_keywords if skill in jd_lower]

    if any(token in jd_lower for token in ['senior', 'lead', 'principal']):
        experience_level = 'Senior/Lead level'
    elif any(token in jd_lower for token in ['junior', 'entry', 'fresher']):
        experience_level = 'Junior/Entry level'
    else:
        experience_level = 'Mid-level'

    result = {}
    if found_skills:
        result['key_skills'] = f"Key Technologies: {', '.join(found_skills[:6])}"
    result['key_requirements'] = f"Experience Level: {experience_level}"
    return result
