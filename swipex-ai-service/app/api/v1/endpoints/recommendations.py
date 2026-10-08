"""API endpoints for job ranking and semantic recommendations."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.security import verify_api_key
from app.schemas.recommendations import (
    FeedJobsRequest,
    FeedJobsResponse,
    RankJobsRequest,
    RankJobsResponse,
)
from app.services.ai.recommendations import feed_jobs_from_index, rank_candidate_jobs

router = APIRouter(prefix="/recommendations", dependencies=[Depends(verify_api_key)])


@router.post(
    "/rank",
    response_model=RankJobsResponse,
    status_code=status.HTTP_200_OK,
    summary="Rank candidate jobs using semantic, skill, and swipe signal weights",
    responses={
        200: {"description": "Candidate jobs ranked successfully"},
        400: {"description": "Invalid payload"},
        401: {"description": "Unauthorized"},
    },
)
async def rank_jobs(
    payload: RankJobsRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> RankJobsResponse:
    """Rank given candidate job postings against candidate profile."""
    return await asyncio.to_thread(rank_candidate_jobs, payload)


@router.post(
    "/jobs",
    response_model=FeedJobsResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend jobs directly from vector store index",
    responses={
        200: {"description": "Recommended jobs retrieved successfully"},
        401: {"description": "Unauthorized"},
    },
)
async def feed_jobs(
    payload: FeedJobsRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> FeedJobsResponse:
    """Query ChromaDB vector store and return top matching jobs for user feed."""
    return await asyncio.to_thread(feed_jobs_from_index, payload)
