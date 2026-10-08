"""API v1 router composition."""

from fastapi import APIRouter, Depends

from app.api.v1.endpoints import (
    ats,
    extraction,
    health,
    index,
    insights,
    recommendations,
    suggestions,
)
from app.core.security import verify_api_key

api_router = APIRouter(dependencies=[Depends(verify_api_key)])
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(extraction.router, tags=["Extraction"])
api_router.include_router(index.router, tags=["Indexing"])
api_router.include_router(ats.router, tags=["ATS Scoring"])
api_router.include_router(suggestions.router, tags=["Suggestions"])
api_router.include_router(recommendations.router, tags=["Recommendations"])
api_router.include_router(insights.router, tags=["Insights"])
