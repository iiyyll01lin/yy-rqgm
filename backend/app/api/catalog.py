"""Catalog endpoints: hardware tiers + model configs (from the gatekeeper DB)."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.models import ModelOut, ModelsResponse, TierOut, TiersResponse
from backend.gatekeeper.spec import list_models, list_tiers

router = APIRouter(prefix="/api", tags=["catalog"])


@router.get("/tiers", response_model=TiersResponse)
def get_tiers() -> TiersResponse:
    return TiersResponse(tiers=[TierOut.model_validate(t.to_public_dict()) for t in list_tiers()])


@router.get("/models", response_model=ModelsResponse)
def get_models() -> ModelsResponse:
    return ModelsResponse(models=[ModelOut.model_validate(m.to_public_dict()) for m in list_models()])
