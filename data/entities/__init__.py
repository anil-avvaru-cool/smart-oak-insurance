from __future__ import annotations

from data.entities.entity_vehicle import resolve_vehicles
from data.entities.entity_person import resolve_persons
from data.entities.entity_address import resolve_addresses
from data.entities.entity_phone import resolve_phones
from data.entities.entity_policy import resolve_policies

__all__ = [
    "resolve_vehicles",
    "resolve_persons",
    "resolve_addresses",
    "resolve_phones",
    "resolve_policies",
]
