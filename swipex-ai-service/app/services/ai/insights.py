"""M4 Career Intelligence & Insights Service (Skill Gap, Resume Performance, Hiring Trends)."""

import statistics
from collections import Counter

from app.schemas.insights import (
    HiringTrendsRequest,
    HiringTrendsResponse,
    ResumePerformanceRequest,
    ResumePerformanceResponse,
    SkillGapItem,
    SkillGapRequest,
    SkillGapResponse,
)


def analyze_skill_gap(req: SkillGapRequest) -> SkillGapResponse:
    """Identify critical and optional skill gaps compared against target role/benchmark jobs."""
    if not req.jobs:
        return SkillGapResponse(
            missing_critical_skills=[],
            missing_optional_skills=[],
            match_rate=100.0 if req.user_skills else 0.0,
        )

    user_skills_set = {s.lower().strip() for s in req.user_skills if s.strip()}
    
    req_counts: Counter[str] = Counter()
    opt_counts: Counter[str] = Counter()
    total_required_mentions = 0
    total_required_matched = 0

    total_jobs = len(req.jobs)

    for job in req.jobs:
        req_set = {s.lower().strip() for s in job.required_skills if s.strip()}
        opt_set = {s.lower().strip() for s in job.skills if s.strip()} - req_set

        total_required_mentions += len(req_set)

        for s in req_set:
            if s in user_skills_set:
                total_required_matched += 1
            else:
                req_counts[s] += 1

        for s in opt_set:
            if s not in user_skills_set:
                opt_counts[s] += 1

    missing_critical: list[SkillGapItem] = [
        SkillGapItem(
            skill=skill,
            market_frequency=count,
            importance_score=round((count / total_jobs) * 100.0, 2),
            category="required",
        )
        for skill, count in req_counts.most_common()
    ]

    missing_optional: list[SkillGapItem] = [
        SkillGapItem(
            skill=skill,
            market_frequency=count,
            importance_score=round((count / total_jobs) * 100.0, 2),
            category="nice_to_have",
        )
        for skill, count in opt_counts.most_common()
    ]

    if total_required_mentions > 0:
        match_rate = round((total_required_matched / total_required_mentions) * 100.0, 2)
    else:
        match_rate = 100.0

    return SkillGapResponse(
        missing_critical_skills=missing_critical,
        missing_optional_skills=missing_optional,
        match_rate=match_rate,
    )


def compute_resume_performance(req: ResumePerformanceRequest) -> ResumePerformanceResponse:
    """Compute summary statistics and score distribution for resume ATS performance."""
    scores = req.job_scores
    if not scores:
        return ResumePerformanceResponse(
            average_score=0.0,
            median_score=0.0,
            min_score=0.0,
            max_score=0.0,
            score_distribution={"0-40": 0, "41-60": 0, "61-80": 0, "81-100": 0},
        )

    avg_val = round(statistics.mean(scores), 2)
    med_val = round(statistics.median(scores), 2)
    min_val = round(min(scores), 2)
    max_val = round(max(scores), 2)

    dist = {"0-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for s in scores:
        if s <= 40.0:
            dist["0-40"] += 1
        elif s <= 60.0:
            dist["41-60"] += 1
        elif s <= 80.0:
            dist["61-80"] += 1
        else:
            dist["81-100"] += 1

    return ResumePerformanceResponse(
        average_score=avg_val,
        median_score=med_val,
        min_score=min_val,
        max_score=max_val,
        score_distribution=dist,
    )


def compute_hiring_trends(req: HiringTrendsRequest) -> HiringTrendsResponse:
    """Aggregate market demand statistics across job postings dataset."""
    jobs = req.jobs
    if not jobs:
        return HiringTrendsResponse(
            top_demanded_skills=[],
            average_years_required=0.0,
            work_mode_distribution={},
            experience_level_distribution={},
        )

    skill_counts: Counter[str] = Counter()
    work_modes: Counter[str] = Counter()
    exp_levels: Counter[str] = Counter()
    exp_years: list[float] = []

    for job in jobs:
        all_job_skills = {s.lower().strip() for s in (job.required_skills + job.skills) if s.strip()}
        for s in all_job_skills:
            skill_counts[s] += 1

        if job.work_mode:
            work_modes[job.work_mode.lower()] += 1
        if job.experience_level:
            exp_levels[job.experience_level.lower()] += 1
        if job.min_experience_years is not None:
            exp_years.append(job.min_experience_years)

    top_skills = [
        {"skill": skill, "count": count, "demand_percentage": round((count / len(jobs)) * 100.0, 1)}
        for skill, count in skill_counts.most_common(15)
    ]

    avg_years = round(statistics.mean(exp_years), 1) if exp_years else 0.0

    return HiringTrendsResponse(
        top_demanded_skills=top_skills,
        average_years_required=avg_years,
        work_mode_distribution=dict(work_modes),
        experience_level_distribution=dict(exp_levels),
    )
