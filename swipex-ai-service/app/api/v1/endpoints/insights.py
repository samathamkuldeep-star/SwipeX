"""API endpoints for M4 career intelligence & insights."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.core.security import verify_api_key
from app.schemas.insights import (
    HiringTrendsRequest,
    HiringTrendsResponse,
    ResumePerformanceRequest,
    ResumePerformanceResponse,
    SkillGapRequest,
    SkillGapResponse,
)
from app.services.ai.insights import (
    analyze_skill_gap,
    compute_hiring_trends,
    compute_resume_performance,
)

router = APIRouter(prefix="/insights", dependencies=[Depends(verify_api_key)])


@router.post(
    "/skill-gap",
    response_model=SkillGapResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform skill gap analysis against benchmark jobs",
)
async def get_skill_gap(
    payload: SkillGapRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> SkillGapResponse:
    return await asyncio.to_thread(analyze_skill_gap, payload)


@router.post(
    "/resume-performance",
    response_model=ResumePerformanceResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute ATS score summary distribution for candidate resume",
)
async def get_resume_performance(
    payload: ResumePerformanceRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> ResumePerformanceResponse:
    return await asyncio.to_thread(compute_resume_performance, payload)


@router.post(
    "/hiring-trends",
    response_model=HiringTrendsResponse,
    status_code=status.HTTP_200_OK,
    summary="Compute market hiring trends and top demanded skills",
)
async def get_hiring_trends(
    payload: HiringTrendsRequest,
    _: Annotated[None, Depends(verify_api_key)] = None,
) -> HiringTrendsResponse:
    return await asyncio.to_thread(compute_hiring_trends, payload)
