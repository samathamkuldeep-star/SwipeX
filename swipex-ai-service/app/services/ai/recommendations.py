"""Semantic Job Recommendations and Ranking Service."""

import logging
from typing import Any

from app.schemas.recommendations import (
    FeedJobsRequest,
    FeedJobsResponse,
    RankedJob,
    RankJobsRequest,
    RankJobsResponse,
)
from app.services.ai.embeddings import get_embedding_service
from app.services.ai.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def rank_candidate_jobs(req: RankJobsRequest) -> RankJobsResponse:
    """Rank a set of candidate jobs using semantic, skill, and swipe signal weights."""
    if not req.candidate_jobs:
        return RankJobsResponse(ranked_jobs=[])

    user_skills_set = {s.lower().strip() for s in req.resume_skills if s.strip()}

    # Pre-embed candidate job descriptions if resume_text is available
    job_embeddings: list[list[float] | None] = [None] * len(req.candidate_jobs)
    resume_embedding: list[float] | None = None

    if req.resume_text and req.resume_text.strip():
        try:
            emb_svc = get_embedding_service()
            resume_embedding = emb_svc.embed_text(req.resume_text[:2000])
            job_texts = [f"{j.title} {j.description[:1000]}" for j in req.candidate_jobs]
            embedded_jobs = emb_svc.embed_batch(job_texts)
            if len(embedded_jobs) == len(req.candidate_jobs):
                job_embeddings = embedded_jobs
        except Exception as err:
            logger.warning("Embedding candidate jobs failed in recommendation ranker: %s", err)

    # Compute swipe preference target vector/skills if swiped_job_ids provided
    swiped_ids_set = set(req.swiped_job_ids)

    ranked_items: list[RankedJob] = []

    for idx, job in enumerate(req.candidate_jobs):
        # 1. Skill overlap score
        job_skills = {s.lower().strip() for s in (job.required_skills + job.skills + job.keywords) if s.strip()}
        if job_skills and user_skills_set:
            matched = user_skills_set & job_skills
            missing = job_skills - user_skills_set
            skill_score = (len(matched) / len(job_skills)) * 100.0
            matched_list = sorted([s for s in (job.required_skills + job.skills) if s.lower().strip() in matched])
            missing_list = sorted([s for s in job.required_skills if s.lower().strip() in missing])
        elif not job_skills:
            skill_score = 100.0  # Perfect on skills if job specifies none
            matched_list = []
            missing_list = []
        else:
            skill_score = 0.0
            matched_list = []
            missing_list = list(job.required_skills)

        # 2. Semantic score
        semantic_score: float | None = None
        if resume_embedding is not None and job_embeddings[idx] is not None:
            raw_sim = get_embedding_service().cosine_similarity(resume_embedding, job_embeddings[idx])
            # Scale 0.20-0.75 -> 0-100
            if raw_sim <= 0.20:
                semantic_score = 0.0
            elif raw_sim >= 0.75:
                semantic_score = 100.0
            else:
                semantic_score = (raw_sim - 0.20) / (0.75 - 0.20) * 100.0

        # 3. Swipe bias score
        swipe_bias_score: float | None = None
        if swiped_ids_set:
            if job.job_id in swiped_ids_set:
                swipe_bias_score = 100.0
            else:
                swipe_bias_score = 0.0
        else:
            swipe_bias_score = None

        # 4. Blended Score calculation
        sb_val = swipe_bias_score if swipe_bias_score is not None else 0.0

        if semantic_score is not None:
            if swipe_bias_score is not None:
                blended = 0.55 * semantic_score + 0.35 * skill_score + 0.10 * sb_val
            else:
                blended = 0.60 * semantic_score + 0.40 * skill_score
        else:
            if swipe_bias_score is not None:
                blended = 0.85 * skill_score + 0.15 * sb_val
            else:
                blended = skill_score

        blended_clamped = max(0.0, min(100.0, round(blended, 2)))

        ranked_items.append(
            RankedJob(
                job_id=job.job_id,
                score=blended_clamped,
                semantic_score=round(semantic_score, 2) if semantic_score is not None else None,
                skill_score=round(skill_score, 2),
                swipe_bias_score=round(swipe_bias_score, 2) if swipe_bias_score is not None else None,
                matched_skills=matched_list,
                missing_skills=missing_list,
            )
        )

    # Sort descending by blended score
    ranked_items.sort(key=lambda x: x.score, reverse=True)

    return RankJobsResponse(ranked_jobs=ranked_items[: req.top_k])


def feed_jobs_from_index(req: FeedJobsRequest) -> FeedJobsResponse:
    """Retrieve and rank recommendations directly from ChromaDB vector store index."""
    vector_store = get_vector_store()
    
    # Query vector store if query embedding is available
    query_text = req.resume_text or " ".join(req.resume_skills) or "Software Engineer"
    
    metadata_filter: dict[str, Any] = {}
    if req.filters:
        if req.filters.work_mode:
            metadata_filter["work_mode"] = req.filters.work_mode
        if req.filters.experience_level:
            metadata_filter["experience_level"] = req.filters.experience_level
        if req.filters.company_type:
            metadata_filter["company_type"] = req.filters.company_type

    try:
        query_embedding = get_embedding_service().embed_text(query_text)
        results = vector_store.query_similar_jobs(
            query_embedding=query_embedding,
            top_k=req.top_k,
            metadata_filter=metadata_filter if metadata_filter else None,
        )

        ranked_jobs: list[RankedJob] = []
        user_skills_set = {s.lower().strip() for s in req.resume_skills if s.strip()}

        for res in results:
            raw_sim = res.get("distance", 0.0)  # distance or similarity
            # Convert cosine distance/similarity if needed
            sem_score = max(0.0, min(100.0, (1.0 - raw_sim) * 100.0)) if raw_sim <= 1.0 else max(0.0, min(100.0, raw_sim * 100.0))
            
            job_meta = res.get("metadata", {})
            required_skills = job_meta.get("required_skills", "").split(",") if isinstance(job_meta.get("required_skills"), str) else []
            required_skills = [s.strip() for s in required_skills if s.strip()]

            if required_skills and user_skills_set:
                matched = [s for s in required_skills if s.lower().strip() in user_skills_set]
                missing = [s for s in required_skills if s.lower().strip() not in user_skills_set]
                skill_score = (len(matched) / len(required_skills)) * 100.0
            else:
                matched = []
                missing = required_skills
                skill_score = 50.0

            blended = 0.60 * sem_score + 0.40 * skill_score

            ranked_jobs.append(
                RankedJob(
                    job_id=res.get("job_id", ""),
                    score=round(blended, 2),
                    semantic_score=round(sem_score, 2),
                    skill_score=round(skill_score, 2),
                    swipe_bias_score=None,
                    matched_skills=matched,
                    missing_skills=missing,
                )
            )

        ranked_jobs.sort(key=lambda x: x.score, reverse=True)
        return FeedJobsResponse(jobs=ranked_jobs, total_found=len(ranked_jobs))

    except Exception as err:
        logger.warning("Vector store feed_jobs_from_index failed: %s", err)
        return FeedJobsResponse(jobs=[], total_found=0)
