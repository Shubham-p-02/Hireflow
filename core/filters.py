"""
Candidate filtering functions for skills, location, and experience.
Provides post-search filtering to refine results based on job requirements.
"""

import re
import sys
sys.path.append(".")
from typing import List, Dict, Any
from utils.utils import get_logger

logger = get_logger(__name__)


def _normalize_skill(skill: str) -> str:
    """Lowercase and strip non-alphanumeric characters for loose comparison.

    LLM-parsed skill strings vary in spacing/casing/punctuation (e.g. "Gen AI"
    vs "GenAI"), so exact string matching against them is too brittle.
    """
    return re.sub(r'[^a-z0-9]', '', skill.lower())


def _get(candidate: Dict[str, Any], key: str, default=None):
    """Look up a field at top-level first, then inside 'metadata'.

    The hybrid indexer returns flat dicts (key at top level), while the
    vector store returns nested dicts (key inside 'metadata').  This helper
    handles both shapes so filters work regardless of source.
    """
    if key in candidate:
        return candidate[key]
    return candidate.get('metadata', {}).get(key, default)


def filter_by_skills(candidates: List[Dict[str, Any]], required_skills: List[str]) -> List[Dict[str, Any]]:
    """Filter candidates who have at least one required skill.

    Matching is partial and normalized (case/space/punctuation-insensitive) since
    LLM-parsed skill strings rarely match the recruiter's exact wording
    (e.g. "Gen AI" vs "GenAI"). Candidates need only overlap on one skill;
    the re-ranker handles finer-grained fit scoring afterward.
    """
    if not required_skills:
        return candidates

    normalized_required = [_normalize_skill(s) for s in required_skills if _normalize_skill(s)]
    if not normalized_required:
        return candidates

    filtered_candidates = []

    for candidate in candidates:
        candidate_skills = _get(candidate, 'skills', [])
        normalized_candidate_skills = [_normalize_skill(cs) for cs in candidate_skills if _normalize_skill(cs)]

        has_match = any(
            req in cand or cand in req
            for req in normalized_required
            for cand in normalized_candidate_skills
        )

        if has_match:
            filtered_candidates.append(candidate)

    return filtered_candidates


def filter_by_location(candidates: List[Dict[str, Any]], target_locations: List[str]) -> List[Dict[str, Any]]:
    """Filter candidates by target locations (case-insensitive partial match)"""
    if not target_locations:
        return candidates

    filtered_candidates = []

    for candidate in candidates:
        candidate_location = (_get(candidate, 'location', '') or '').lower()
        location_match = any(
            loc.lower() in candidate_location for loc in target_locations
        )

        if location_match:
            filtered_candidates.append(candidate)

    return filtered_candidates


def filter_by_experience(candidates: List[Dict[str, Any]], min_experience: float) -> List[Dict[str, Any]]:
    """Filter candidates who meet minimum experience requirement"""
    if min_experience is None:
        return candidates

    filtered_candidates = []

    for candidate in candidates:
        candidate_experience = _get(candidate, 'experience')

        if candidate_experience is not None and candidate_experience >= min_experience:
            filtered_candidates.append(candidate)

    return filtered_candidates


def apply_filters(
    candidates: List[Dict[str, Any]],
    required_skills: List[str] = None,
    target_locations: List[str] = None,
    min_experience: float = None
) -> List[Dict[str, Any]]:
    """Apply all filters (skills, location, experience) to candidate list"""
    filtered_candidates = candidates
    before = len(filtered_candidates)

    if required_skills:
        filtered_candidates = filter_by_skills(filtered_candidates, required_skills)
        after = len(filtered_candidates)
        if after < before:
            logger.info(f"Skills filter: {before} → {after} candidates (required: {required_skills})")
        before = after

    if target_locations:
        filtered_candidates = filter_by_location(filtered_candidates, target_locations)
        after = len(filtered_candidates)
        if after < before:
            logger.info(f"Location filter: {before} → {after} candidates (target: {target_locations})")
        before = after

    if min_experience is not None:
        filtered_candidates = filter_by_experience(filtered_candidates, min_experience)
        after = len(filtered_candidates)
        if after < before:
            logger.info(f"Experience filter: {before} → {after} candidates (min: {min_experience})")

    return filtered_candidates

if __name__ == "__main__":
    candidates = [
        {"name": "Alice",  "skills": ["Python", "SQL", "AWS"],          "location": "New York",      "experience": 5},
        {"name": "Bob",    "skills": ["Java", "Spring"],                 "location": "San Francisco", "experience": 2},
        {"name": "Carol",  "skills": ["Python", "Django", "SQL"],        "location": "Austin",        "experience": 7},
        {"name": "Dave",   "skills": ["Python", "AWS"],                  "location": "New York",      "experience": 1},
    ]

    print("=== filter_by_skills ===")
    result = filter_by_skills(candidates, required_skills=["Python", "SQL"])
    print(f"Candidates with Python AND SQL: {[c['name'] for c in result]}")

    print("\n=== filter_by_location ===")
    result = filter_by_location(candidates, target_locations=["New York"])
    print(f"Candidates in New York: {[c['name'] for c in result]}")

    print("\n=== filter_by_experience ===")
    result = filter_by_experience(candidates, min_experience=3)
    print(f"Candidates with >= 3 years experience: {[c['name'] for c in result]}")

    print("\n=== apply_filters (Python + New York + 3 yrs) ===")
    result = apply_filters(
        candidates,
        required_skills=["Python"],
        target_locations=["New York"],
        min_experience=3,
    )
    print(f"After all filters: {[c['name'] for c in result]}")