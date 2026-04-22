from __future__ import annotations

import dash_bootstrap_components as dbc

APP_TITLE = "NeoResist-MD · ResistanceLoop v1"
ASSET_VERSION = "2026041701"

# Keep plot colors aligned with the current clinical UI palette.
TIER_COLORS = {1: "#3FB950", 2: "#D29922", 3: "#F85149"}
TIER_LABELS = {1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}

EXTERNAL_STYLESHEETS = [
    dbc.themes.CYBORG,
    f"https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap&v={ASSET_VERSION}",
    f"https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap&v={ASSET_VERSION}",
]
