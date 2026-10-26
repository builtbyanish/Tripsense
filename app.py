"""
TripSense — AI-Powered Andhra Pradesh Tourism & Hospitality Platform
====================================================================
Tripster-Inspired Modern Visual Redesign with Sliding Segmented Switch

Product Experience:
1. Primary: Tourism Discovery & AI Trip Planning (Member 2 Hybrid Recommender + Member 3 Dates & RAG)
2. Secondary: Hotel Search & Booking with Member 1 Demand Forecasting & Member 3 Review Intelligence

Design Specifications:
- Vivid blue outer canvas: #2878E5 / #2563EB gradient
- Spacious white rounded main application container (border-radius: 36px)
- Compact modern top navigation with brand logo & action pills
- Full-width photographic hero with dark translucent overlay & centered text
- Floating rounded pill search panel with subtle vertical dividers & circular blue search button
- Popular destinations editorial photo-card grid matching Tripster proportions
- Sliding segmented [ Trip ] [ Hotel ] switch
- Verified Wikipedia / Wikimedia destination image integration
- Zero emojis across the entire user interface (vector Lucide SVGs & text labels)
"""

from __future__ import annotations

import csv
import html
import logging
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Path & Module Setup
# ---------------------------------------------------------------------------
_APP_ROOT = Path(__file__).resolve().parent
import sys
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

for _p in [_APP_ROOT / "member2" / "src", _APP_ROOT / "member3" / "src"]:
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

# Local services
from itinerary.services import (
    data_service,
    recommendation_service,
    weather_service,
    hotel_service,
    date_service,
    rag_service,
    photo_service,
    icons,
)
from src.services import wikipedia_images

# Itinerary engine
try:
    from itinerary import build_itinerary, Itinerary, ItineraryDay, ItineraryStop, _PACE_MAX_HOURS, _PACE_TARGETS  # type: ignore
    _ITINERARY_AVAILABLE = True
except ImportError:
    try:
        from itinerary import build_itinerary, Itinerary, ItineraryDay, ItineraryStop  # type: ignore
        _PACE_MAX_HOURS = {"relaxed": 7.5, "balanced": 9.5, "packed": 11.5}
        _PACE_TARGETS = {
            "relaxed": {"min_stops": 1, "max_stops": 2, "max_hrs": 7.5, "target_hrs": 5.5},
            "balanced": {"min_stops": 2, "max_stops": 3, "max_hrs": 9.5, "target_hrs": 7.5},
            "packed": {"min_stops": 3, "max_stops": 4, "max_hrs": 11.5, "target_hrs": 9.5},
        }
        _ITINERARY_AVAILABLE = True
    except ImportError:
        _ITINERARY_AVAILABLE = False
        _PACE_MAX_HOURS = {"relaxed": 7.5, "balanced": 9.5, "packed": 11.5}
        _PACE_TARGETS = {
            "relaxed": {"min_stops": 1, "max_stops": 2, "max_hrs": 7.5, "target_hrs": 5.5},
            "balanced": {"min_stops": 2, "max_stops": 3, "max_hrs": 9.5, "target_hrs": 7.5},
            "packed": {"min_stops": 3, "max_stops": 4, "max_hrs": 11.5, "target_hrs": 9.5},
        }

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TripSense — Discover Andhra Pradesh",
    page_icon="https://raw.githubusercontent.com/lucide-icons/lucide/main/icons/compass.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Background & Hero Visual Assets
# ---------------------------------------------------------------------------
# Visual assets are served efficiently via Streamlit static serving (/app/static/)

# ---------------------------------------------------------------------------
# Global Tripster Design System CSS
# ---------------------------------------------------------------------------
_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root {
    --primary-gold: #C8963E;
    --primary-gold-hover: #A87524;
    --primary-gold-dark: #8C5A20;
    --primary-gold-light: #FAF2E4;
    --primary-gold-soft: #F5E0BD;
    --surface-parchment: #FAF2E4;
    --surface-card: rgba(255, 253, 248, 0.95);
    --text-espresso: #2E1B0A;
    --text-bronze: #8C5A20;
    --text-body: #382613;
    --text-muted: #6B5238;
    --border-gold: #C8963E;
    --border-gold-soft: #D4B37C;
    --border-light: #EADBCC;
    --font-heading: 'Outfit', -apple-system, sans-serif;
    --font-body: 'Plus Jakarta Sans', -apple-system, sans-serif;
    --radius-main: 32px;
    --radius-pill: 9999px;
    --shadow-float: 0 16px 38px rgba(140, 90, 32, 0.18);
    --shadow-card: 0 8px 24px rgba(140, 90, 32, 0.12);
}

/* Outer Canvas: Aerial Andhra Pradesh Coastal Highway Photograph */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-image: linear-gradient(rgba(20, 15, 10, 0.45), rgba(20, 15, 10, 0.60)),
                      url('app/static/canvas_background.jpg') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
    color: var(--text-body) !important;
    font-family: var(--font-body) !important;
    min-height: 100vh !important;
}

/* Hide default streamlit chrome */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}

/* Main Website Surface: Warm Mandala Tapestry / Temple Parchment Container */
.block-container {
    background-color: #FAF2E4 !important;
    background-image: linear-gradient(rgba(253, 248, 238, 0.80), rgba(253, 248, 238, 0.86)),
                      url('app/static/mandala_bg.jpg') !important;
    background-size: 100% auto !important;
    background-repeat: repeat-y !important;
    background-position: top center !important;
    border: 2px solid var(--border-gold-soft) !important;
    border-radius: var(--radius-main) !important;
    box-shadow: 0 25px 70px rgba(45, 25, 10, 0.45) !important;
    max-width: 1280px !important;
    padding: 2rem 3rem 4rem !important;
    margin: 2rem auto !important;
}

@media (max-width: 900px) {
    .block-container {
        border-radius: 20px !important;
        padding: 1.5rem 1.25rem 3rem !important;
        margin: 0.75rem auto !important;
    }
}

/* Unique Typography */
h1, h2, h3, h4, h5 {
    font-family: var(--font-heading) !important;
    color: var(--text-espresso) !important;
}
p, span, div, label, li {
    font-family: var(--font-body);
}

/* Top Navigation Bar */
.tripster-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.5rem 0 1.8rem;
    border-bottom: 1.5px solid var(--border-light);
}
.tripster-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-family: var(--font-heading) !important;
    font-size: 1.6rem;
    font-weight: 800;
    color: var(--text-espresso);
    text-decoration: none;
    letter-spacing: -0.02em;
}
.tripster-brand span {
    color: var(--primary-gold);
}
.tripster-nav-menu {
    display: flex;
    gap: 2rem;
    align-items: center;
}
.tripster-nav-link {
    font-family: var(--font-body) !important;
    font-size: 0.94rem;
    font-weight: 600;
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.15s ease;
}
.tripster-nav-link:hover, .tripster-nav-link.active {
    color: var(--primary-gold-dark);
}
.tripster-auth-group {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.tripster-btn-outline {
    border: 1.5px solid var(--primary-gold);
    background: #FFFDF9;
    color: var(--primary-gold-dark);
    border-radius: var(--radius-pill);
    padding: 0.45rem 1.25rem;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    transition: all 0.15s ease;
}
.tripster-btn-outline:hover {
    background: var(--primary-gold-light);
    color: var(--primary-gold-hover);
}
.tripster-btn-primary {
    background: var(--primary-gold);
    color: #FFFFFF !important;
    border: none;
    border-radius: var(--radius-pill);
    padding: 0.5rem 1.4rem;
    font-size: 0.88rem;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    box-shadow: 0 4px 12px rgba(200, 150, 62, 0.35);
    transition: all 0.15s ease;
}
.tripster-btn-primary:hover {
    background: var(--primary-gold-hover);
}

/* Hero Banner - Authentic Simhachalam Temple Photograph with High-Contrast Text */
.tripster-hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    height: 380px;
    background-image: linear-gradient(135deg, rgba(30, 18, 8, 0.48) 0%, rgba(45, 25, 10, 0.62) 100%),
                      url('app/static/simhachalam_hero.jpg') !important;
    background-size: cover !important;
    background-position: center 38% !important;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2.5rem 2rem;
    margin-top: 1rem;
    box-shadow: 0 10px 30px rgba(35, 20, 5, 0.35);
}
.tripster-hero, .tripster-hero h1, .tripster-hero p {
    color: #FFFFFF !important;
}
.tripster-hero span.hero-brand-highlight {
    color: #FFD700 !important;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.75), 0 0 25px rgba(255, 215, 0, 0.55) !important;
}
.tripster-hero-title {
    font-family: var(--font-heading) !important;
    color: #FFFFFF !important;
    font-size: 2.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    margin: 0 0 0.75rem 0 !important;
    text-shadow: 0 4px 20px rgba(0, 0, 0, 0.65) !important;
    line-height: 1.15 !important;
}
.tripster-hero-subtitle {
    font-family: var(--font-body) !important;
    color: #FAF2E4 !important;
    font-size: 1.15rem !important;
    max-width: 680px !important;
    margin: 0 auto !important;
    line-height: 1.55 !important;
    font-weight: 500 !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.55) !important;
}

/* Floating "Plan Your Trip" Dropdown Menu Container (Overlapping Hero) */
div[data-testid="stExpander"]:has(#plan-marker),
div.stExpander:has(#plan-marker) {
    position: relative !important;
    margin-top: -46px !important;
    margin-bottom: 2rem !important;
    z-index: 25 !important;
    max-width: 960px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

div[data-testid="stExpander"]:has(#plan-marker) > details,
div.stExpander:has(#plan-marker) > details {
    background: linear-gradient(135deg, #FFC83B 0%, #E8A824 100%) !important;
    border: 2.5px solid #C48818 !important;
    border-radius: 9999px !important;
    box-shadow: 0 16px 40px rgba(200, 140, 25, 0.35), 0 4px 14px rgba(0, 0, 0, 0.08) !important;
    overflow: hidden !important;
    transition: border-radius 0.25s ease, box-shadow 0.25s ease !important;
}

div[data-testid="stExpander"]:has(#plan-marker) > details[open],
div.stExpander:has(#plan-marker) > details[open] {
    border-radius: 28px !important;
    box-shadow: 0 20px 48px rgba(200, 140, 25, 0.38) !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary,
div.stExpander:has(#plan-marker) summary {
    padding: 1rem 2.25rem !important;
    cursor: pointer !important;
    background: linear-gradient(135deg, #FFC83B 0%, #E8A824 100%) !important;
    border-radius: 9999px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
    transition: filter 0.2s ease !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary:hover,
div.stExpander:has(#plan-marker) summary:hover {
    filter: brightness(1.05) !important;
}

div[data-testid="stExpander"]:has(#plan-marker) details[open] > summary,
div.stExpander:has(#plan-marker) details[open] > summary {
    border-radius: 26px 26px 0 0 !important;
    border-bottom: 2px solid #D49826 !important;
    background: linear-gradient(135deg, #FFC83B 0%, #E8A824 100%) !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary > span,
div.stExpander:has(#plan-marker) summary > span {
    display: flex !important;
    width: 100% !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary > span > span:first-child,
div.stExpander:has(#plan-marker) summary > span > span:first-child {
    position: absolute !important;
    right: 0.5rem !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary [data-testid="stMarkdownContainer"],
div.stExpander:has(#plan-marker) summary [data-testid="stMarkdownContainer"] {
    text-align: center !important;
    margin: 0 auto !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary [data-testid="stMarkdownContainer"] p,
div.stExpander:has(#plan-marker) summary [data-testid="stMarkdownContainer"] p {
    font-family: var(--font-heading), 'Outfit', sans-serif !important;
    font-size: 1.25rem !important;
    font-weight: 800 !important;
    letter-spacing: 0.02em !important;
    color: #2E1B0A !important;
    margin: 0 !important;
    text-align: center !important;
}

div[data-testid="stExpander"]:has(#plan-marker) summary [data-testid="stIconMaterial"],
div.stExpander:has(#plan-marker) summary [data-testid="stIconMaterial"],
div[data-testid="stExpander"]:has(#plan-marker) summary svg,
div.stExpander:has(#plan-marker) summary svg {
    color: #2E1B0A !important;
    fill: #2E1B0A !important;
    stroke: #2E1B0A !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

div[data-testid="stExpander"]:has(#plan-marker) div[data-testid="stExpanderDetails"],
div.stExpander:has(#plan-marker) div[data-testid="stExpanderDetails"] {
    padding: 1.75rem 2rem 1.25rem !important;
    background: rgba(255, 253, 248, 0.98) !important;
    border: none !important;
}

/* Streamlit Inputs, Dropdowns & Controls: Warm Cream Boxes with Gold Ochre Border */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
[data-testid="stDateInput"] div[data-baseweb="input"],
[data-testid="stDateInput"] div[data-baseweb="base-input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="base-input"],
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stTextInput"] div[data-baseweb="base-input"],
div[data-baseweb="select"] > div,
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="select"] {
    background-color: #FFFDF9 !important;
    background: #FFFDF9 !important;
    border: 2px solid #C8963E !important;
    border-radius: 12px !important;
    color: #2E1B0A !important;
    box-shadow: 0 2px 6px rgba(200, 150, 62, 0.08) !important;
}

/* All Text and Values Inside Inputs and Dropdowns */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
[data-testid="stDateInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="select"] span,
div[data-baseweb="select"] div {
    background-color: transparent !important;
    color: #2E1B0A !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    font-size: 0.94rem !important;
}

/* Number Input Buttons (- and +) */
[data-testid="stNumberInput"] button,
div[data-baseweb="input"] button {
    background-color: #FFFDF9 !important;
    background: #FFFDF9 !important;
    color: #8C5A20 !important;
    border: none !important;
}
[data-testid="stNumberInput"] button:hover,
div[data-baseweb="input"] button:hover {
    background-color: #FAF2E4 !important;
    color: #C8963E !important;
}
[data-testid="stNumberInput"] button svg,
div[data-baseweb="input"] button svg {
    fill: #8C5A20 !important;
    color: #8C5A20 !important;
}

/* Focus and Hover on Boxes */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover,
[data-testid="stDateInput"] div[data-baseweb="input"]:hover,
[data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"]:hover {
    border-color: #A87524 !important;
    box-shadow: 0 0 0 3px rgba(200, 150, 62, 0.22) !important;
}
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #A87524 !important;
    box-shadow: 0 0 0 3.5px rgba(200, 150, 62, 0.25) !important;
}

/* Dropdown Menu (When Dropped Down / Open): Cream with Gold Border */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] ul,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #FFFDF9 !important;
    background: #FFFDF9 !important;
    border: 2px solid #C8963E !important;
    border-radius: 14px !important;
    box-shadow: 0 14px 35px rgba(140, 90, 32, 0.22) !important;
    padding: 6px !important;
}
li[role="option"] {
    background-color: #FFFDF9 !important;
    background: #FFFDF9 !important;
    color: #2E1B0A !important;
    border-radius: 8px !important;
    font-family: var(--font-body) !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    padding: 8px 12px !important;
    transition: all 0.15s ease !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #FAF2E4 !important;
    background: #FAF2E4 !important;
    color: #8C5A20 !important;
}

/* Eliminate Red Focus / Red Error Borders */
*:focus, *:focus-visible {
    outline: none !important;
}
div[aria-invalid="true"],
div[data-baseweb="input"][aria-invalid="true"],
div[data-baseweb="select"][aria-invalid="true"] > div {
    border-color: #C8963E !important;
    box-shadow: 0 0 0 2px rgba(200, 150, 62, 0.25) !important;
}

label[data-testid="stWidgetLabel"] p {
    font-family: var(--font-heading) !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: var(--text-espresso) !important;
    margin-bottom: 0.35rem !important;
}

/* Interest Buttons / MultiSelect Tags: Gold Boxes with White Text */
[data-baseweb="tag"],
span[data-baseweb="tag"],
div[data-baseweb="tag"],
[data-testid="stMultiSelect"] [data-baseweb="tag"],
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background-color: #C8963E !important;
    background: #C8963E !important;
    border: 1.5px solid #8C5A20 !important;
    border-radius: 8px !important;
    color: #FFFFFF !important;
    padding: 4px 10px !important;
    box-shadow: 0 2px 8px rgba(200, 150, 62, 0.3) !important;
}
[data-baseweb="tag"] *,
span[data-baseweb="tag"] *,
div[data-baseweb="tag"] *,
[data-testid="stMultiSelect"] [data-baseweb="tag"] * {
    color: #FFFFFF !important;
    fill: #FFFFFF !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
}
[data-baseweb="tag"] svg,
span[data-baseweb="tag"] svg,
[data-testid="stMultiSelect"] [data-baseweb="tag"] svg {
    fill: #FFFFFF !important;
    color: #FFFFFF !important;
}

/* Checkbox: Gold Theme */
[data-testid="stCheckbox"] input[type="checkbox"]:checked + div {
    background-color: #C8963E !important;
    border-color: #8C5A20 !important;
}
[data-testid="stCheckbox"] span {
    color: #2E1B0A !important;
    font-weight: 600 !important;
    font-family: var(--font-body) !important;
}

/* Sliding Segmented [ Trip ] [ Hotel ] Switch */
.switch-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 1.5rem 0 2rem;
}
.segmented-pill-box {
    display: inline-flex;
    background: #F5E0BD;
    border-radius: var(--radius-pill);
    padding: 6px;
    border: 1.5px solid var(--border-gold-soft);
    gap: 6px;
}

/* Button Styling: Gold & Parchment Exclusively */
.stButton > button,
.stButton > button[kind="secondary"] {
    background-color: #FFFDF9 !important;
    color: var(--primary-gold-dark) !important;
    border: 1.5px solid var(--border-gold) !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.65rem 1.75rem !important;
    font-family: var(--font-heading) !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(140, 90, 32, 0.12) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover,
.stButton > button[kind="secondary"]:hover {
    background-color: var(--primary-gold-light) !important;
    border-color: var(--primary-gold-hover) !important;
    color: var(--primary-gold-hover) !important;
    transform: translateY(-1px) !important;
}

.stButton > button[kind="primary"] {
    background-color: var(--primary-gold) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    padding: 0.85rem 2.2rem !important;
    font-family: var(--font-heading) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 20px rgba(200, 150, 62, 0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: var(--primary-gold-hover) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 12px 25px rgba(200, 150, 62, 0.45) !important;
}

/* Popular Destinations Editorial Photo Grid */
.dest-grid-container {
    display: grid;
    grid-template-columns: 1.1fr 1fr 1.3fr 1fr;
    grid-gap: 16px;
    margin: 1.5rem 0 2.5rem;
}
@media (max-width: 960px) {
    .dest-grid-container {
        grid-template-columns: 1fr 1fr;
    }
}
@media (max-width: 600px) {
    .dest-grid-container {
        grid-template-columns: 1fr;
    }
}

.dest-photo-card {
    position: relative;
    border-radius: 20px;
    overflow: hidden;
    background-size: cover;
    background-position: center;
    box-shadow: var(--shadow-card);
    border: 1.5px solid var(--border-gold-soft);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.dest-photo-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 32px rgba(140, 90, 32, 0.18);
}
.dest-badge-pill {
    position: absolute;
    bottom: 14px;
    left: 14px;
    background: rgba(255, 253, 248, 0.95);
    color: var(--text-espresso);
    font-family: var(--font-heading);
    font-weight: 700;
    font-size: 0.85rem;
    padding: 6px 16px;
    border-radius: var(--radius-pill);
    box-shadow: 0 4px 14px rgba(140, 90, 32, 0.15);
    border: 1.5px solid var(--border-gold);
}

/* Category Tiles */
.category-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
    margin: 1.5rem 0 2rem;
}
.cat-tile {
    background: rgba(255, 253, 248, 0.94);
    border: 1.5px solid var(--border-gold-soft);
    border-radius: 16px;
    padding: 1.15rem 0.6rem;
    text-align: center;
    transition: all 0.2s ease;
    cursor: pointer;
    box-shadow: 0 2px 8px rgba(140, 90, 32, 0.05);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 96px;
    text-decoration: none !important;
    user-select: none;
    -webkit-user-select: none;
}
.cat-tile:hover {
    background: var(--primary-gold-light);
    border-color: var(--primary-gold);
    box-shadow: 0 6px 18px rgba(200, 150, 62, 0.18);
    transform: translateY(-2px);
}
.cat-tile.active {
    background: #C8963E !important;
    border-color: #8C5A20 !important;
    box-shadow: 0 6px 20px rgba(200, 150, 62, 0.35) !important;
    transform: translateY(-2px);
}
.cat-tile.active .cat-tile-title {
    color: #FFFFFF !important;
}
.cat-tile.active svg {
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
}
.cat-tile-title {
    font-family: var(--font-heading) !important;
    font-size: 0.92rem;
    font-weight: 700;
    color: var(--text-espresso);
    margin-top: 0.45rem;
}

/* Zero-Reload Client-Side Category Tabs & Panels (Pure CSS & Instant) */
.cat-radio-input {
    display: none !important;
    position: absolute;
    opacity: 0;
    pointer-events: none;
}
.interest-section-wrapper .interest-cat-panel {
    display: none;
}
#cat-tab-beaches:checked ~ #panel-beaches,
#cat-tab-temples:checked ~ #panel-temples,
#cat-tab-hill_stations:checked ~ #panel-hill_stations,
#cat-tab-waterfalls:checked ~ #panel-waterfalls,
#cat-tab-heritage:checked ~ #panel-heritage,
#cat-tab-sanctuaries:checked ~ #panel-sanctuaries {
    display: block !important;
}

#cat-tab-beaches:checked ~ .interest-tabs-header .tab-tile-beaches,
#cat-tab-temples:checked ~ .interest-tabs-header .tab-tile-temples,
#cat-tab-hill_stations:checked ~ .interest-tabs-header .tab-tile-hill_stations,
#cat-tab-waterfalls:checked ~ .interest-tabs-header .tab-tile-waterfalls,
#cat-tab-heritage:checked ~ .interest-tabs-header .tab-tile-heritage,
#cat-tab-sanctuaries:checked ~ .interest-tabs-header .tab-tile-sanctuaries {
    background: #C8963E !important;
    border-color: #8C5A20 !important;
    box-shadow: 0 6px 20px rgba(200, 150, 62, 0.35) !important;
    transform: translateY(-2px);
}
#cat-tab-beaches:checked ~ .interest-tabs-header .tab-tile-beaches .cat-tile-title,
#cat-tab-temples:checked ~ .interest-tabs-header .tab-tile-temples .cat-tile-title,
#cat-tab-hill_stations:checked ~ .interest-tabs-header .tab-tile-hill_stations .cat-tile-title,
#cat-tab-waterfalls:checked ~ .interest-tabs-header .tab-tile-waterfalls .cat-tile-title,
#cat-tab-heritage:checked ~ .interest-tabs-header .tab-tile-heritage .cat-tile-title,
#cat-tab-sanctuaries:checked ~ .interest-tabs-header .tab-tile-sanctuaries .cat-tile-title {
    color: #FFFFFF !important;
}
#cat-tab-beaches:checked ~ .interest-tabs-header .tab-tile-beaches svg,
#cat-tab-temples:checked ~ .interest-tabs-header .tab-tile-temples svg,
#cat-tab-hill_stations:checked ~ .interest-tabs-header .tab-tile-hill_stations svg,
#cat-tab-waterfalls:checked ~ .interest-tabs-header .tab-tile-waterfalls svg,
#cat-tab-heritage:checked ~ .interest-tabs-header .tab-tile-heritage svg,
#cat-tab-sanctuaries:checked ~ .interest-tabs-header .tab-tile-sanctuaries svg {
    stroke: #FFFFFF !important;
    color: #FFFFFF !important;
}

.interest-tabs-header {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
    margin: 1.5rem 0 2rem;
}
@media (max-width: 900px) {
    .interest-tabs-header {
        grid-template-columns: repeat(3, 1fr);
    }
}
@media (max-width: 550px) {
    .interest-tabs-header {
        grid-template-columns: repeat(2, 1fr);
    }
}

.interest-cat-panel {
    animation: panelFadeIn 0.25s ease-out;
}
@keyframes panelFadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.interest-cards-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    margin-top: 1rem;
}
@media (max-width: 960px) {
    .interest-cards-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}
@media (max-width: 600px) {
    .interest-cards-grid {
        grid-template-columns: 1fr;
    }
}

.interest-place-card {
    background: #FFFDF9;
    border: 1.5px solid var(--border-gold-soft);
    border-radius: 18px;
    padding: 1.15rem;
    box-shadow: 0 4px 14px rgba(140, 90, 32, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    display: flex;
    flex-direction: column;
}
.interest-place-card:hover {
    transform: translateY(-3px);
    border-color: #C8963E;
    box-shadow: 0 8px 24px rgba(140, 90, 32, 0.16);
}
.interest-place-img {
    width: 100%;
    height: 185px;
    object-fit: cover;
    border-radius: 12px;
    margin-bottom: 0.75rem;
    background-color: #F5E0BD;
}

details.interest-details-toggle summary::-webkit-details-marker {
    display: none;
}
details.interest-details-toggle summary:hover {
    background: #C8963E !important;
    color: #FFFFFF !important;
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFDF9 !important;
    border: 1.5px solid var(--border-gold-soft) !important;
    border-radius: 18px !important;
    box-shadow: 0 4px 14px rgba(140, 90, 32, 0.08) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div:hover {
    border-color: #C8963E !important;
    box-shadow: 0 8px 24px rgba(140, 90, 32, 0.15) !important;
}

/* Hotel & Itinerary Cards: Gold & Parchment */
.editorial-stop-card {
    background: rgba(255, 253, 248, 0.96);
    border: 1.5px solid var(--border-gold-soft);
    border-radius: 18px;
    padding: 1.25rem;
    margin-bottom: 1.25rem;
    box-shadow: var(--shadow-card);
}
.editorial-badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: var(--font-heading);
    font-size: 0.76rem;
    font-weight: 700;
    border-radius: var(--radius-pill);
    padding: 4px 12px;
    background: var(--primary-gold-soft);
    color: var(--primary-gold-dark);
    border: 1px solid var(--border-gold-soft);
}
.badge-gold, .badge-blue {
    background: var(--primary-gold-soft);
    color: #6B420C;
    border: 1px solid var(--border-gold-soft);
}
.badge-soft {
    background: #FFFDF9;
    color: var(--text-espresso);
    border: 1px solid var(--border-light);
}

/* =========================================================================
   CINEMATIC WELCOME SPLASH OVERLAY WITH VIDEO
   ========================================================================= */
.tripsense-splash {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    z-index: 99999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background-color: #1A1006 !important;
    overflow: hidden !important;
    pointer-events: none !important;
    animation: splashFadeOut 2.4s cubic-bezier(0.25, 1, 0.5, 1) forwards !important;
}

.splash-video {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    object-fit: cover !important;
    opacity: 0.65 !important;
    filter: brightness(0.68) contrast(1.15) !important;
    transform: scale(1.04) !important;
    animation: splashVideoZoom 3.2s ease-out forwards !important;
}

.splash-backdrop-gradient {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    background: radial-gradient(circle at center, rgba(30, 18, 8, 0.45) 0%, rgba(18, 9, 3, 0.88) 100%) !important;
    pointer-events: none !important;
}

.splash-content {
    position: relative !important;
    z-index: 10 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    text-align: center !important;
    padding: 2rem !important;
    max-width: 720px !important;
    animation: splashContentRise 2.8s ease-out forwards !important;
}

.splash-emblem {
    margin-bottom: 1.25rem !important;
    filter: drop-shadow(0 4px 20px rgba(200, 150, 62, 0.65)) !important;
    animation: emblemPulse 2.4s ease-in-out infinite !important;
}

.splash-title {
    font-family: var(--font-heading) !important;
    font-size: 3.4rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.025em !important;
    color: #FFFFFF !important;
    margin: 0 0 0.75rem !important;
    text-shadow: 0 4px 25px rgba(0, 0, 0, 0.7), 0 0 40px rgba(200, 150, 62, 0.5) !important;
    background: linear-gradient(135deg, #FFFDF9 0%, #F5E0BD 50%, #C8963E 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

.splash-subtitle {
    font-family: var(--font-body) !important;
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: #FAF2E4 !important;
    max-width: 600px !important;
    line-height: 1.5 !important;
    margin: 0 0 2rem !important;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.8) !important;
}

.splash-progress-track {
    width: 280px !important;
    height: 4px !important;
    background: rgba(255, 255, 255, 0.18) !important;
    border-radius: 999px !important;
    overflow: hidden !important;
    position: relative !important;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.5) !important;
    margin-bottom: 1rem !important;
}

.splash-progress-bar {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    height: 100% !important;
    width: 0% !important;
    background: linear-gradient(90deg, #C8963E, #F5E0BD, #FFFFFF) !important;
    border-radius: 999px !important;
    box-shadow: 0 0 12px #C8963E !important;
    animation: progressBarFill 2.3s cubic-bezier(0.2, 0.8, 0.2, 1) forwards !important;
}

@keyframes progressBarFill {
    0% { width: 0%; }
    40% { width: 45%; }
    80% { width: 85%; }
    100% { width: 100%; }
}

@keyframes emblemPulse {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 4px 18px rgba(200, 150, 62, 0.55)); }
    50% { transform: scale(1.06); filter: drop-shadow(0 6px 28px rgba(200, 150, 62, 0.85)); }
}

@keyframes splashVideoZoom {
    0% { transform: scale(1.0); }
    100% { transform: scale(1.06); }
}

@keyframes splashContentRise {
    0% { opacity: 0; transform: translateY(18px); }
    15% { opacity: 1; transform: translateY(0); }
    80% { opacity: 1; transform: translateY(0); }
    100% { opacity: 0.95; }
}

@keyframes splashFadeOut {
    0% {
        opacity: 1;
        visibility: visible;
        pointer-events: all;
    }
    75% {
        opacity: 1;
        visibility: visible;
        pointer-events: all;
    }
    90% {
        opacity: 0.35;
        visibility: visible;
        pointer-events: none;
    }
    100% {
        opacity: 0;
        visibility: hidden;
        pointer-events: none;
        display: none;
    }
}
</style>
"""
st.markdown(_APP_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cinematic Animated Welcome Splash Screen (Plays on site load, fades into main page)
# ---------------------------------------------------------------------------
if st.query_params.get("intro") == "1":
    st.session_state.has_shown_welcome_splash = False

if not st.session_state.get("has_shown_welcome_splash", False):
    st.session_state.has_shown_welcome_splash = True
    st.markdown("""
    <div id="tripsense-splash-overlay" class="tripsense-splash" onclick="this.style.opacity='0'; this.style.pointerEvents='none'; setTimeout(() => this.remove(), 400);">
      <video autoplay muted loop playsinline poster="app/static/simhachalam_hero.jpg" class="splash-video">
        <source src="app/static/welcome_video.webm" type="video/webm">
      </video>
      <div class="splash-backdrop-gradient"></div>
      <div class="splash-content">
        <div class="splash-emblem">
          <svg width="68" height="68" viewBox="0 0 24 24" fill="none" stroke="#F5E0BD" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10" stroke="#C8963E" stroke-width="2"/>
            <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" stroke="#FAF2E4"/>
            <circle cx="12" cy="12" r="3.2" fill="#C8963E"/>
          </svg>
        </div>
        <h1 class="splash-title">Welcome to TripSense</h1>
        <p class="splash-subtitle">Discover iconic temple complexes, golden coastal shores, and pristine mountain valleys</p>
        <div class="splash-progress-track">
          <div class="splash-progress-bar"></div>
        </div>
      </div>
    </div>
    <script>
      setTimeout(function(){
        var el = document.getElementById('tripsense-splash-overlay');
        if (el) {
          el.style.transition = 'opacity 0.5s ease';
          el.style.opacity = '0';
          el.style.pointerEvents = 'none';
          setTimeout(function(){ if (el && el.parentNode) el.parentNode.removeChild(el); }, 550);
        }
      }, 2200);
    </script>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data Loading & Caching
# ---------------------------------------------------------------------------
@st.cache_data
def _load_all_data():
    p_df = data_service.load_places()
    h_df = data_service.load_hotels()
    w_df = data_service.load_weather()
    bm_df = data_service.load_best_months()
    ri_df = data_service.load_review_insights()
    i_df = data_service.load_interactions()
    hr_df = data_service.load_hotel_reviews()
    return p_df, h_df, w_df, bm_df, ri_df, i_df, hr_df

places_df, hotels_df, weather_df, best_months_df, review_insights_df, interactions_df, hotel_reviews_df = _load_all_data()

# Pre-load Wikipedia Image Manifest
wikipedia_images.load_manifest()

_DISTRICTS = ["All Andhra Pradesh"] + sorted(
    [d for d in places_df["district"].dropna().unique().tolist() if d]
)

_INTEREST_OPTIONS = [
    "Temples & Spirituality",
    "Beaches & Coastlines",
    "Hill Stations & Valleys",
    "Waterfalls & Streams",
    "Wildlife & Nature Reserves",
    "Heritage & Forts",
    "Caves & Rock Formations",
    "Lakes & River Gorges",
    "Adventure & Trekking",
    "Culture & Museums",
]

_INTEREST_KEYWORD_MAP = {
    "Temples & Spirituality": "temple spiritual religious pilgrimage sacred",
    "Beaches & Coastlines": "beach coastal sea ocean shore waves",
    "Hill Stations & Valleys": "hill mountain valley scenic viewpoints clouds",
    "Waterfalls & Streams": "waterfall falls cascade plunge stream",
    "Wildlife & Nature Reserves": "wildlife sanctuary forest nature reserve park",
    "Heritage & Forts": "fort heritage history ancient palace architecture",
    "Caves & Rock Formations": "cave cavern rock stalactite geology",
    "Lakes & River Gorges": "lake dam reservoir river gorge water",
    "Adventure & Trekking": "adventure trekking climbing hiking outdoors",
    "Culture & Museums": "museum culture art archaeological gallery",
}

_TRAVELER_SUITABLE_MAP = {
    "Solo": "solo",
    "Couple": "couple",
    "Family": "family",
    "Friends": "friends",
    "Group": "group",
}

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Trip"  # Trip is primary, Hotel is secondary
if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "recs" not in st.session_state:
    st.session_state.recs = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}
if "date_scores" not in st.session_state:
    st.session_state.date_scores = {}
if "best_months_sugg" not in st.session_state:
    st.session_state.best_months_sugg = []
if "scroll_to_results" not in st.session_state:
    st.session_state.scroll_to_results = False

# ---------------------------------------------------------------------------
# Top Navigation Header (Tripster Style)
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="tripster-nav">
  <div class="tripster-brand">
    {icons.get_icon("compass", size=24, color="#C8963E")}
    TripSense<span>.</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Banner (Tripster Style with Andhra Pradesh Tourism Headline)
# ---------------------------------------------------------------------------
st.markdown("""
<div class="tripster-hero">
  <h1 class="tripster-hero-title" style="color: #FFFFFF !important; font-family: 'Outfit', sans-serif !important; font-size: 2.9rem !important; font-weight: 800 !important; letter-spacing: -0.025em !important; text-shadow: 0 4px 24px rgba(0,0,0,0.65) !important; margin: 0 0 0.75rem 0 !important; line-height: 1.15 !important;">
    Discover Andhra Pradesh with <span class="hero-brand-highlight" style="color: #FFD700 !important; text-shadow: 0 4px 20px rgba(0, 0, 0, 0.75), 0 0 25px rgba(255, 215, 0, 0.55) !important;">TripSense</span>
  </h1>
  <p class="tripster-hero-subtitle" style="color: #FAF2E4 !important; font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 1.15rem !important; font-weight: 500 !important; text-shadow: 0 2px 12px rgba(0,0,0,0.5) !important; max-width: 680px !important; margin: 0 auto !important; line-height: 1.55 !important;">
    Explore remarkable places, find the best time to visit and build a journey tailored to your interests.
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Floating "Plan Your Trip" Dropdown Menu (Overlapping Hero)
# ---------------------------------------------------------------------------
with st.expander("Plan Your Trip", expanded=False):
    st.html('<span id="plan-marker"></span>')
    s_col1, s_col2, s_col3, s_col4, s_col5 = st.columns([1.15, 1.15, 1.05, 1.1, 1.1])

    with s_col1:
        default_dist_idx = 0
        if "selected_district" in st.session_state and st.session_state.selected_district in _DISTRICTS:
            default_dist_idx = _DISTRICTS.index(st.session_state.selected_district)
        district = st.selectbox(
            "Location",
            _DISTRICTS,
            index=default_dist_idx,
            help="Where would you like to explore in Andhra Pradesh?",
        )

    with s_col2:
        interests_selected = st.multiselect(
            "Interests",
            _INTEREST_OPTIONS,
            default=["Temples & Spirituality", "Nature & Forests"] if "Nature & Forests" in _INTEREST_OPTIONS else ["Temples & Spirituality"],
            help="Choose your travel themes to guide recommendations.",
        )

    today = date.today()
    with s_col3:
        d_sub1, d_sub2 = st.columns(2)
        with d_sub1:
            start_date = st.date_input(
                "Start Date",
                value=today + timedelta(days=14),
                min_value=today,
                max_value=today + timedelta(days=365),
            )
            target_month = start_date.month
        with d_sub2:
            n_days = st.number_input(
                "Duration",
                min_value=1,
                max_value=30,
                value=3,
                step=1,
                help="Choose trip duration from 1 up to 30 days.",
            )

    with s_col4:
        curr_tt = st.session_state.get("plan_traveler_type", "Family")
        tt_options = ["Solo", "Couple", "Family", "Friends", "Group"]
        tt_idx = tt_options.index(curr_tt) if curr_tt in tt_options else 2

        if curr_tt == "Solo":
            traveler_type = st.selectbox(
                "Travellers",
                tt_options,
                index=tt_idx,
                key="plan_traveler_type",
            )
            n_travelers = 1
        else:
            t_sub1, t_sub2 = st.columns(2)
            with t_sub1:
                traveler_type = st.selectbox(
                    "Travellers",
                    tt_options,
                    index=tt_idx,
                    key="plan_traveler_type",
                )
            with t_sub2:
                n_travelers = st.number_input(
                    "Count",
                    min_value=2 if traveler_type == "Couple" else 1,
                    max_value=30,
                    value=4 if traveler_type in ("Family", "Group") else 2,
                )

    with s_col5:
        current_pace = st.session_state.get("trip_pace", "balanced").capitalize()
        if current_pace not in ["Relaxed", "Balanced", "Packed"]:
            current_pace = "Balanced"
        pace_label = st.select_slider(
            "Trip Pace",
            options=["Relaxed", "Balanced", "Packed"],
            value=current_pace,
            help="Dynamic AI pacing: Relaxed (1–2 stops/day), Balanced (2–3 stops/day), Packed (3–4 stops/day).",
        )
        pace_choice = pace_label.lower()
        st.session_state.trip_pace = pace_choice

    # Demographic Constraints (Terrain & Age Safety for Seniors & Children across all groups)
    has_seniors = False
    has_kids = False

    if traveler_type == "Solo":
        show_age_filters = st.checkbox(
            "Configure Traveler Age & Terrain Accessibility",
            value=False,
            help="Specify age: travelers under 10 or 60+ will automatically avoid high terrain, steep climbs, caves, and rugged trails."
        )
        if show_age_filters:
            solo_age = st.number_input(
                "Traveler Age",
                min_value=1,
                max_value=105,
                value=28,
                key="solo_traveler_age",
                help="Ages under 10 or 60+ will receive gentle, accessible places without steep terrain climbing."
            )
            has_seniors = (solo_age >= 60)
            has_kids = (solo_age < 10)
            if has_seniors or has_kids:
                st.info("Gentle Terrain Active: High climbs, steep waterfalls, caves, and rugged trails will be filtered out for safety and ease.")
    else:
        show_age_filters = st.checkbox(
            "Configure Traveler Ages & Mobility (for Seniors / Children)",
            value=False,
            help="Check this to enter member ages so we can recommend gentle terrain, low-walking, and senior/child-friendly places."
        )
        if show_age_filters:
            age_cols = st.columns(min(int(n_travelers), 4))
            member_ages = []
            for idx in range(int(n_travelers)):
                col = age_cols[idx % len(age_cols)]
                with col:
                    default_age = 65 if idx == 0 and traveler_type in ("Family", "Group") else (8 if idx == 1 and traveler_type == "Family" else 30)
                    age_val = st.number_input(
                        f"Age {idx+1}",
                        min_value=1,
                        max_value=105,
                        value=default_age,
                        key=f"tripster_age_{idx}",
                    )
                    member_ages.append(age_val)
            has_seniors = any(a >= 60 for a in member_ages)
            has_kids = any(a < 10 for a in member_ages)
            if has_seniors or has_kids:
                st.info("Gentle Terrain Active: High climbs, steep waterfalls, caves, and rugged trails will be filtered out for seniors (60+) / kids (<10).")

    _, btn_mid, _ = st.columns([1, 1.6, 1])
    with btn_mid:
        search_submitted = st.button("Explore & Plan Journey", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# The Sliding Segmented [ Trip ] [ Hotel ] Switch
# ---------------------------------------------------------------------------
st.markdown("""
<div class="switch-container">
  <div class="segmented-pill-box">
""", unsafe_allow_html=True)

switch_c1, switch_c2 = st.columns([1, 1])
with switch_c1:
    trip_btn = st.button("Trip — Tourism & Discovery", key="mode_trip_btn", use_container_width=True,
                         type="primary" if st.session_state.app_mode == "Trip" else "secondary")
    if trip_btn:
        st.session_state.app_mode = "Trip"
        st.rerun()

with switch_c2:
    hotel_btn = st.button("Hotel — Stays & Booking", key="mode_hotel_btn", use_container_width=True,
                          type="primary" if st.session_state.app_mode == "Hotel" else "secondary")
    if hotel_btn:
        st.session_state.app_mode = "Hotel"
        st.rerun()

st.markdown("""
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Pipeline Execution on Search Submit
# ---------------------------------------------------------------------------
if search_submitted:
    interest_keywords = " ".join(
        _INTEREST_KEYWORD_MAP.get(i, i.lower()) for i in (interests_selected or ["Temples & Spirituality"])
    )
    district_filter = None if district == "All Andhra Pradesh" else district
    suitable_filter = _TRAVELER_SUITABLE_MAP.get(traveler_type, None)

    with st.spinner("Invoking Member 2 Hybrid Recommendation Pipeline..."):
        recs = recommendation_service.get_recommendations(
            interests=interest_keywords,
            places_df=places_df,
            interactions_df=interactions_df,
            district=district_filter,
            traveler_type=suitable_filter,
            top_n=max(30, int(n_days) * 4),
            has_seniors=has_seniors,
            has_kids=has_kids,
        )
    st.session_state.recs = recs

    if not recs.empty:
        place_ids = recs["place_id"].tolist()
        weather_data = weather_service.get_weather_for_places(place_ids, target_month, weather_df)
        st.session_state.weather_data = weather_data

        date_scores_result = {}
        if start_date and not best_months_df.empty:
            best_dates_df_lazy = data_service.load_best_dates()
            date_scores_result = date_service.score_trip_dates(
                start_date, int(n_days), place_ids, best_dates_df_lazy
            )
        st.session_state.date_scores = date_scores_result

        best_months_sugg = date_service.suggest_best_months_for_places(place_ids, best_months_df, top_n=3)
        st.session_state.best_months_sugg = best_months_sugg

        if _ITINERARY_AVAILABLE:
            pace_to_use = st.session_state.get("trip_pace", "balanced")
            itin = build_itinerary(
                recommendations=recs,
                n_days=int(n_days),
                start_date=start_date,
                pace=pace_to_use,
                traveler_name="Traveler",
                traveler_type=traveler_type.lower(),
                n_travelers=int(n_travelers),
                district=district,
                interests=interests_selected,
                weather_data=weather_data,
                date_scores=date_scores_result if date_scores_result else None,
                best_months_suggestions=best_months_sugg,
                month=target_month,
                has_seniors=has_seniors,
                has_kids=has_kids,
            )
            st.session_state.itinerary = itin
            st.session_state.scroll_to_results = True

# ---------------------------------------------------------------------------
# Results Anchor
# ---------------------------------------------------------------------------
st.markdown('<div id="results-anchor" style="scroll-margin-top: 1rem;"></div>', unsafe_allow_html=True)
if st.session_state.get("scroll_to_results", False):
    components.html("""
    <script>
      setTimeout(() => {
        const target = window.parent.document.getElementById('results-anchor');
        if (target) { target.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
      }, 200);
    </script>
    """, height=0)
    st.session_state.scroll_to_results = False

# ===========================================================================
# VIEW MODE: TRIP (Tourism Discovery, AI Itinerary, Dates, Map, RAG)
# ===========================================================================
if st.session_state.app_mode == "Trip":
    itin = st.session_state.get("itinerary")

    if itin is not None:
        pace_key = getattr(itin, "pace", "balanced").lower()
        pace_title = pace_key.title()
        if "relax" in pace_key:
            pace_stops_text = "1–2 stops/day"
        elif "pack" in pace_key:
            pace_stops_text = "3–4 stops/day"
        else:
            pace_stops_text = "2–3 stops/day"

        st.markdown(f"""
        <div style="margin: 1.5rem 0 1rem;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
            <div>
              <div style="display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap; margin-bottom:0.35rem;">
                <span class="editorial-badge-pill badge-gold">Curated Itinerary Ready</span>
                <span class="editorial-badge-pill badge-soft">Pace: {pace_title} ({pace_stops_text})</span>
              </div>
              <h2 style="font-size:2.2rem; margin:0.3rem 0; font-weight:800; letter-spacing:-0.02em; color:var(--text-espresso);">{itin.trip_name}</h2>
              <p style="color:#6B5238; margin:0; font-size:0.95rem;">
                {itin.n_days} Days &bull; {sum(len(d.stops) for d in itin.days)} Scheduled Stops &bull; Region: {itin.district}
              </p>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        rag_insight_text = getattr(itin, "rag_insight", "")
        if rag_insight_text:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #FFFDF8 0%, #FAF2E4 100%); border:1.5px solid #D4B37C; border-left:4px solid #C8963E; border-radius:12px; padding:1rem 1.25rem; margin:0.75rem 0 0.75rem; box-shadow:0 3px 12px rgba(140, 90, 32, 0.05);">
              <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.35rem;">
                {icons.get_icon("sparkles", size=18, color="#C8963E")}
                <span style="font-weight:700; color:#8C5A20; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.04em;">TripSense RAG Planner Intelligence</span>
              </div>
              <p style="color:#382613; font-size:0.92rem; margin:0; line-height:1.55;">{rag_insight_text}</p>
            </div>
            """, unsafe_allow_html=True)

        # Best Travel Dates & Climate Advisory Card
        date_notes_text = getattr(itin, "date_notes", "")
        best_months_list = getattr(itin, "best_months", [])
        if date_notes_text or best_months_list:
            opt_months_markup = ""
            if best_months_list:
                month_pills = " ".join([f'<span class="editorial-badge-pill badge-gold" style="font-size:0.8rem; padding:3px 10px;">{m}</span>' for m in best_months_list])
                opt_months_markup = f'<div style="margin-top:0.45rem; display:flex; align-items:center; gap:0.5rem; flex-wrap:wrap;"><strong style="font-size:0.86rem; color:#8C5A20;">Top Recommended Months:</strong> {month_pills}</div>'

            st.markdown(f"""
            <div style="background:#FFFDF8; border:1.5px solid #D4B37C; border-left:4px solid #C8963E; border-radius:12px; padding:0.9rem 1.25rem; margin:0.5rem 0 1.25rem; box-shadow:0 2px 10px rgba(140, 90, 32, 0.05);">
              <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">
                {icons.get_icon("calendar", size=18, color="#C8963E")}
                <span style="font-weight:700; color:#8C5A20; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.04em;">Best Dates & Climate Advisory</span>
              </div>
              <p style="color:#382613; font-size:0.92rem; margin:0; line-height:1.5;">{date_notes_text or 'Seasonality and climate comfort scores evaluated from historical weather and holiday crowd datasets.'}</p>
              {opt_months_markup}
            </div>
            """, unsafe_allow_html=True)

        # Haversine distance calculator
        def _calc_haversine_distance(lat1, lon1, lat2, lon2) -> float:
            if None in (lat1, lon1, lat2, lon2):
                return 0.0
            try:
                import math
                R = 6371.0
                phi1 = math.radians(float(lat1))
                phi2 = math.radians(float(lat2))
                dphi = math.radians(float(lat2) - float(lat1))
                dlam = math.radians(float(lon2) - float(lon1))
                a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
                return float(R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a)))
            except Exception:
                return 0.0

        # Day-by-Day View
        for day in itin.days:
            day_str = f"Day {day.day_number}"
            if day.date:
                day_str += f" — {day.date.strftime('%A, %d %B %Y')}"

            # Compute inter-stop distances and total travel distance for the day
            day_total_km = 0.0
            for i in range(len(day.stops) - 1):
                day_total_km += _calc_haversine_distance(
                    getattr(day.stops[i], "latitude", None),
                    getattr(day.stops[i], "longitude", None),
                    getattr(day.stops[i+1], "latitude", None),
                    getattr(day.stops[i+1], "longitude", None),
                )

            dist_header_tag = f" &bull; {day_total_km:.1f} km Total Travel" if day_total_km > 0 and len(day.stops) > 1 else ""
            day_theme = getattr(day, "theme", "")
            day_rag_tip = getattr(day, "rag_tip", "")
            theme_header_tag = f" &bull; {day_theme}" if day_theme else ""
            day_crowd = getattr(day, "crowd_note", "")
            crowd_header_tag = f" &bull; Crowd: {day_crowd.title()}" if day_crowd else ""

            with st.expander(f"{day_str} ({len(day.stops)} Stops{dist_header_tag}{theme_header_tag}{crowd_header_tag})", expanded=(day.day_number == 1)):
                for idx, stop in enumerate(day.stops):
                    p_name = getattr(stop, "name", "Destination")
                    p_cat = getattr(stop, "category", "")
                    p_dist = getattr(stop, "district", "")
                    p_dur = getattr(stop, "visit_duration_hrs", 2.0)
                    p_desc = getattr(stop, "description", "")
                    p_meth = getattr(stop, "recommendation_method", "Hybrid Recs")
                    p_reas = getattr(stop, "recommendation_reason", "")
                    p_id = getattr(stop, "place_id", "")

                    # Distance from previous stop in the route
                    dist_from_prev = 0.0
                    prev_stop_name = ""
                    drive_time_text = ""
                    if idx > 0:
                        prev_stop = day.stops[idx - 1]
                        prev_stop_name = getattr(prev_stop, "name", f"Stop {idx}")
                        dist_from_prev = _calc_haversine_distance(
                            getattr(prev_stop, "latitude", None),
                            getattr(prev_stop, "longitude", None),
                            getattr(stop, "latitude", None),
                            getattr(stop, "longitude", None),
                        )
                        if dist_from_prev > 0:
                            est_mins = max(5, int(dist_from_prev / 45.0 * 60))
                            drive_time_text = f"{est_mins} mins drive" if est_mins < 60 else f"{est_mins//60}h {est_mins%60}m drive"
                            st.markdown(f"""
                            <div style="margin: 0.35rem 0 0.85rem 0.5rem; display: flex; align-items: center; gap: 0.75rem;">
                              <div style="display: inline-flex; align-items: center; gap: 0.5rem; background: #FAF2E4; border: 1.5px solid #D4B37C; border-radius: 999px; padding: 5px 14px; box-shadow: 0 2px 8px rgba(140, 90, 32, 0.08);">
                                {icons.get_icon("car", size=15, color="#8C5A20")}
                                <span style="font-family: var(--font-heading) !important; font-size: 0.84rem; font-weight: 700; color: #6B420C;">
                                  {dist_from_prev:.1f} km from {html.escape(prev_stop_name)}
                                </span>
                                <span style="font-size: 0.78rem; color: #8C5A20; font-weight: 600;">({drive_time_text})</span>
                              </div>
                            </div>
                            """, unsafe_allow_html=True)

                    photos = photo_service.get_destination_photos(
                        place_name=p_name,
                        category=p_cat,
                        district=p_dist,
                        place_id=p_id,
                    )

                    dist_badge_markup = f'<span class="editorial-badge-pill badge-gold">{dist_from_prev:.1f} km from {html.escape(prev_stop_name)}</span>' if dist_from_prev > 0 else ""
                    stop_weather = getattr(stop, "weather_note", "")
                    weather_badge_markup = f'<span class="editorial-badge-pill badge-soft">{html.escape(stop_weather)}</span>' if stop_weather else ""

                    c_img, c_info = st.columns([1.2, 2.2])
                    with c_img:
                        if photos:
                            st.image(photos[0]["url"], caption=photos[0].get("title", p_name), use_container_width=True)

                    with c_info:
                        st.html(f"""<div style="padding:0.25rem 0;">
<div style="display:flex; gap:0.5rem; margin-bottom:0.45rem; flex-wrap:wrap;">
<span class="editorial-badge-pill badge-gold">{html.escape(str(p_cat).replace('_',' ').title())}</span>
<span class="editorial-badge-pill badge-soft">{p_dur:g}h Duration</span>
{weather_badge_markup}
{dist_badge_markup}
</div>
<h3 style="font-family:'Outfit', sans-serif !important; font-size:1.4rem; margin:0 0 0.35rem; font-weight:800; color:#2E1B0A;">{idx+1}. {html.escape(p_name)}</h3>
<p style="color:#8C5A20; font-weight:600; font-size:0.88rem; margin-bottom:0.5rem;">District: {html.escape(p_dist)}</p>
<p style="color:#382613; font-size:0.92rem; line-height:1.5; margin-bottom:0.6rem;">{html.escape(p_desc or 'Authentic landmark and scenic attraction in Andhra Pradesh.')}</p>
<div style="background:#FAF2E4; border:1.5px solid #D4B37C; border-left:4px solid #C8963E; padding:0.75rem 1rem; border-radius:0 12px 12px 0; font-size:0.88rem; color:#2E1B0A;">
<strong style="color:#8C5A20;">Why this place:</strong> {html.escape(p_reas)}
</div>
</div>""")
                    st.markdown("<hr style='margin:1rem 0; border:none; border-top:1.5px solid #EADBCC;'>", unsafe_allow_html=True)

                # Recommended overnight accommodation for this day's destinations
                if day.stops:
                    primary_stop = day.stops[-1]
                    day_hotel_df = hotel_service.hotels_for_place(
                        primary_stop.place_id, hotels_df, hotel_reviews_df, review_insights_df, max_hotels=1, district=primary_stop.district
                    )
                    if not day_hotel_df.empty:
                        dh_row = day_hotel_df.iloc[0]
                        dh_name = dh_row.get("hotel_name", "Verified Hotel")
                        dh_tier = str(dh_row.get("tier", "Mid-Range")).replace("_", " ").title()
                        dh_price = dh_row.get("price_range", "INR 1,200–3,500/night")
                        dh_star = dh_row.get("star_rating", "3 Star")
                        dh_score = float(dh_row.get("mean_vader_compound", 0.45))
                        st.markdown(f"""
                        <div style="background:rgba(255,253,248,0.96); border:1.5px solid #D4B37C; border-radius:14px; padding:0.9rem 1.25rem; margin:0.75rem 0 0.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;">
                          <div>
                            <div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">
                              <span class="editorial-badge-pill badge-gold">Overnight Stay for Day {day.day_number}</span>
                              <span class="editorial-badge-pill badge-soft">{dh_tier} &bull; {dh_star}</span>
                            </div>
                            <strong style="font-size:1.05rem; color:#2E1B0A; font-family:'Outfit', sans-serif;">{html.escape(dh_name)}</strong>
                            <span style="color:#6B5238; font-size:0.86rem; margin-left:0.5rem;">Near {html.escape(primary_stop.name)} ({html.escape(primary_stop.district)})</span>
                          </div>
                          <div style="display:flex; align-items:center; gap:0.85rem;">
                            <span style="font-weight:800; color:#C8963E; font-size:1.15rem;">{html.escape(str(dh_price))}</span>
                          </div>
                        </div>
                        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Popular Destinations Section (Editorial Grid matching Tripster Geometry)
    # -----------------------------------------------------------------------
    st.markdown("""
    <div id="destinations" style="margin: 2.5rem 0 1rem;">
      <h2 style="font-size: 1.65rem; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.25rem;">
        Popular destinations in Andhra Pradesh
      </h2>
      <p style="color: #64748B; font-size: 0.95rem; margin: 0 0 1.25rem;">
        Iconic landmarks, coastal beaches, sacred temple complexes, and scenic mountain valleys.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # 4-Column Layout directly inspired by Tripster Screenshot
    p_col1, p_col2, p_col3, p_col4 = st.columns([1.1, 1, 1.25, 1])

    with p_col1:
        # Tall card (Lepakshi Temple architecture)
        st.markdown(f"""
        <div class="dest-photo-card" style="height: 380px; background-image: url('app/static/popular_lepakshi.jpg');">
          <div class="dest-badge-pill">Lepakshi</div>
        </div>
        """, unsafe_allow_html=True)

    with p_col2:
        # Stacked cards (Tirupati Temple + Borra Caves)
        st.markdown(f"""
        <div class="dest-photo-card" style="height: 182px; margin-bottom: 16px; background-image: url('app/static/popular_tirupati.jpg');">
          <div class="dest-badge-pill">Tirupati</div>
        </div>
        <div class="dest-photo-card" style="height: 182px; background-image: url('app/static/popular_borra.jpg');">
          <div class="dest-badge-pill">Borra Caves</div>
        </div>
        """, unsafe_allow_html=True)

    with p_col3:
        # Wide card (Rushikonda Beach / Bay of Bengal)
        st.markdown(f"""
        <div class="dest-photo-card" style="height: 380px; background-image: url('app/static/popular_vizag.jpg');">
          <div class="dest-badge-pill">Visakhapatnam</div>
        </div>
        """, unsafe_allow_html=True)

    with p_col4:
        # Stacked cards (Araku Valley + Gandikota)
        st.markdown(f"""
        <div class="dest-photo-card" style="height: 182px; margin-bottom: 16px; background-image: url('app/static/popular_araku.jpg');">
          <div class="dest-badge-pill">Araku Valley</div>
        </div>
        <div class="dest-photo-card" style="height: 182px; background-image: url('app/static/popular_gandikota.jpg');">
          <div class="dest-badge-pill">Gandikota</div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Discover by Interest Section (Zero-Reload Client-Side Tabs & Panels)
    # -----------------------------------------------------------------------
    st.html("""<div id="discover-interest" style="margin: 3rem 0 1rem;">
<h3 style="font-size: 1.55rem; font-weight: 800; letter-spacing: -0.015em; margin-bottom: 0.25rem;">Discover by Interest</h3>
<p style="color: #6B5238; font-size: 0.94rem; margin: 0 0 1.25rem;">Tap any category below to instantly explore verified beaches, sacred temples, hill stations, waterfalls, heritage landmarks, and sanctuaries.</p>
</div>""")

    categories_meta = [
        ("Beaches", "beaches", "umbrella"),
        ("Temples", "temples", "sun"),
        ("Hill Stations", "hill_stations", "mountain"),
        ("Waterfalls", "waterfalls", "compass"),
        ("Heritage", "heritage", "award"),
        ("Sanctuaries", "sanctuaries", "shield"),
    ]

    def _get_places_for_interest(cat_key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if cat_key == "beaches":
            return df[df["category"].str.lower() == "beach"].copy()
        elif cat_key == "temples":
            return df[df["category"].str.lower() == "temple"].copy()
        elif cat_key == "hill_stations":
            mask = (
                df["category"].str.lower().isin(["hill_station", "viewpoint"]) |
                df["name"].str.contains(r"Hill|Valley|Ghat|Peak|Mountain|Caves", case=False, na=False)
            )
            return df[mask].copy()
        elif cat_key == "waterfalls":
            return df[df["category"].str.lower() == "waterfall"].copy()
        elif cat_key == "heritage":
            mask = df["category"].str.lower().isin(["heritage", "fort", "museum", "caves"])
            return df[mask].copy()
        elif cat_key == "sanctuaries":
            mask = df["category"].str.lower().isin(["wildlife", "nature", "park_garden"])
            return df[mask].copy()
        return df

    # Build Header Tabs HTML (Pure CSS + Radio Inputs for Instant Zero-Reload Switching)
    tabs_html_parts = ['<div class="interest-section-wrapper">']
    for idx, (c_name, c_slug, c_icon) in enumerate(categories_meta):
        is_first = (idx == 0)
        checked_attr = ' checked="checked"' if is_first else ''
        tabs_html_parts.append(
            f'<input type="radio" name="interest_tab_choice" id="cat-tab-{c_slug}" class="cat-radio-input"{checked_attr}>'
        )

    tabs_html_parts.append('<div class="interest-tabs-header">')
    for idx, (c_name, c_slug, c_icon) in enumerate(categories_meta):
        icon_svg = icons.get_icon(c_icon, size=24, color="#C8963E")
        tabs_html_parts.append(
            f'<label for="cat-tab-{c_slug}" class="cat-tile tab-tile-{c_slug}" id="tab-label-{c_slug}">'
            f'{icon_svg}'
            f'<div class="cat-tile-title">{c_name}</div>'
            f'</label>'
        )
    tabs_html_parts.append('</div>')

    # Build Panels HTML
    panels_html_parts = []
    for idx, (c_name, c_slug, c_icon) in enumerate(categories_meta):
        cat_df = _get_places_for_interest(c_slug, places_df)
        total_cat = len(cat_df)
        records = cat_df.to_dict("records")

        panel_chunk = [f'<div id="panel-{c_slug}" class="interest-cat-panel panel-{c_slug}">']
        panel_chunk.append(
            f'<div style="margin: 1.5rem 0 1rem; display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:0.5rem;">'
            f'<div>'
            f'<div style="display:flex; align-items:center; gap:0.4rem; margin-bottom:0.25rem;">'
            f'<span class="editorial-badge-pill badge-gold" style="font-size:0.8rem; padding:4px 12px;">{c_name} Collection</span>'
            f'<span class="editorial-badge-pill badge-soft">{total_cat} Destinations</span>'
            f'</div>'
            f'<h3 style="font-size:1.45rem; font-weight:800; letter-spacing:-0.015em; margin:0.2rem 0 0.15rem; color:#2E1B0A;">'
            f'{c_name} across Andhra Pradesh'
            f'</h3>'
            f'<p style="color:#6B5238; font-size:0.92rem; margin:0;">'
            f'Verified authentic destinations with localized photographs and visitor guides.'
            f'</p>'
            f'</div>'
            f'</div>'
        )

        # Helper to generate cards HTML
        def _render_cards_chunk(items_list):
            out = ['<div class="interest-cards-grid">']
            for item in items_list:
                p_id = str(item.get("place_id", ""))
                p_name = str(item.get("name", "Destination"))
                p_dist = str(item.get("district", "Andhra Pradesh"))
                p_desc = str(item.get("description", "") or "").strip()
                p_rating = item.get("rating")
                
                rating_markup = ""
                if pd.notna(p_rating) and float(p_rating) > 0:
                    rating_markup = f'<span class="editorial-badge-pill badge-gold">{float(p_rating):.1f} / 5.0</span>'

                desc_snip = html.escape(p_desc[:140] + '...' if len(p_desc) > 140 else (p_desc or 'Authentic attraction and landmark in Andhra Pradesh.'))
                out.append(
                    f'<div class="interest-place-card">'
                    f'<img src="app/static/place_{p_id}.jpg" class="interest-place-img" alt="{html.escape(p_name)}" loading="lazy">'
                    f'<div style="padding: 0.15rem 0;">'
                    f'<div style="display:flex; gap:0.4rem; margin-bottom:0.45rem; flex-wrap:wrap;">'
                    f'<span class="editorial-badge-pill badge-gold">{html.escape(p_dist)}</span>'
                    f'{rating_markup}'
                    f'</div>'
                    f'<h4 style="font-family: var(--font-heading) !important; font-size: 1.15rem; font-weight: 800; color: #2E1B0A; margin: 0.25rem 0 0.35rem; line-height: 1.3;">'
                    f'{html.escape(p_name)}'
                    f'</h4>'
                    f'<p style="color: #4A3520; font-size: 0.88rem; line-height: 1.45; margin: 0; min-height: 48px;">'
                    f'{desc_snip}'
                    f'</p>'
                    f'</div>'
                    f'</div>'
                )
            out.append('</div>')
            return "".join(out)

        # Top 9 items
        top_items = records[:9]
        panel_chunk.append(_render_cards_chunk(top_items))

        # If more than 9 items, wrap remaining in client-side <details>
        remaining_items = records[9:]
        if remaining_items:
            panel_chunk.append(
                f'<details class="interest-details-toggle" style="margin-top: 1.5rem; text-align: center;">'
                f'<summary style="display:inline-block; background:#FAF2E4; border:1.5px solid #C8963E; color:#6B420C; border-radius:999px; padding:8px 26px; font-family:var(--font-heading); font-size:0.92rem; font-weight:700; cursor:pointer; list-style:none; box-shadow:0 2px 8px rgba(140,90,32,0.1); transition:all 0.2s ease;">'
                f'Show all {total_cat} {c_name}'
                f'</summary>'
                f'<div style="margin-top: 1.5rem; text-align: left;">'
                f'{_render_cards_chunk(remaining_items)}'
                f'</div>'
                f'</details>'
            )

        panel_chunk.append('</div>')
        panels_html_parts.append("".join(panel_chunk))

    panels_html_parts.append('</div>')
    st.html("".join(tabs_html_parts) + "".join(panels_html_parts))

    # -----------------------------------------------------------------------
    # Interactive Map Section
    # -----------------------------------------------------------------------
    if itin and _ITINERARY_AVAILABLE:
        st.markdown("""
        <div style="margin: 2.5rem 0 1rem;">
          <h3 style="font-size: 1.4rem; font-weight: 800; margin-bottom: 0.25rem;">Geographic Route Atlas</h3>
          <p style="color: #6B5238; font-size: 0.92rem;">Interactive route map with verified coordinates.</p>
        </div>
        """, unsafe_allow_html=True)
        try:
            import folium
            from streamlit_folium import st_folium
            all_stops = [s for day in itin.days for s in day.stops if s.latitude and s.longitude]
            if all_stops:
                avg_lat = sum(s.latitude for s in all_stops) / len(all_stops)
                avg_lon = sum(s.longitude for s in all_stops) / len(all_stops)
                f_map = folium.Map(
                    location=[avg_lat, avg_lon],
                    zoom_start=9,
                    tiles="OpenStreetMap",  # 100% Free OpenStreetMap API - zero API keys required
                )
                for s in all_stops:
                    folium.Marker(
                        location=[s.latitude, s.longitude],
                        popup=f"<b>{html.escape(s.name)}</b><br>{html.escape(s.district)}<br><i>{s.visit_duration_hrs:g}h visit</i>",
                        tooltip=s.name,
                        icon=folium.Icon(color="orange", icon="info-sign"),
                    ).add_to(f_map)

                # Draw expedition route path connecting the stops
                coords = [[s.latitude, s.longitude] for s in all_stops if s.latitude and s.longitude]
                if len(coords) > 1:
                    folium.PolyLine(
                        locations=coords,
                        color="#C8963E",
                        weight=4,
                        opacity=0.85,
                        dash_array="6, 8",
                        tooltip="Expedition Travel Route",
                    ).add_to(f_map)

                st_folium(f_map, width="100%", height=440)
        except Exception:
            pass



# ===========================================================================
# VIEW MODE: HOTEL (Stays & Accommodations Matched to Itinerary Places)
# ===========================================================================
else:
    itin = st.session_state.get("itinerary")
    recs = st.session_state.get("recs")

    if itin is not None:
        # Extract unique planned destinations from the scheduled itinerary
        planned_places = []
        seen_pids = set()
        for day in itin.days:
            for stop in day.stops:
                if stop.place_id not in seen_pids:
                    seen_pids.add(stop.place_id)
                    planned_places.append({
                        "place_id": stop.place_id,
                        "name": stop.name,
                        "district": stop.district,
                        "day": day.day_number,
                        "category": getattr(stop, "category", "Destination"),
                    })

        st.markdown(f"""
        <div id="hotels" style="margin: 1.5rem 0 1.5rem;">
          <span class="editorial-badge-pill badge-gold">Itinerary Accommodations</span>
          <h2 style="font-size: 2.2rem; font-weight: 800; margin: 0.35rem 0 0.5rem; letter-spacing: -0.02em; color: var(--text-espresso); font-family: 'Outfit', sans-serif;">
            Hotels for Predicted Itinerary Places
          </h2>
          <p style="color: #6B5238; font-size: 0.95rem; margin: 0;">
            Showing authentic accommodations matched to destinations predicted by the AI planner for <strong>{html.escape(itin.trip_name)}</strong> ({itin.n_days} Days, {len(planned_places)} Planned Places).
          </p>
        </div>
        """, unsafe_allow_html=True)

        h_col1, h_col2, h_col3 = st.columns([1.6, 1.2, 1.2])
        place_options = ["All Itinerary Places (" + str(len(planned_places)) + " destinations)"] + [
            f"Day {p['day']}: {p['name']} ({p['district']})" for p in planned_places
        ]
        with h_col1:
            selected_place_opt = st.selectbox("Itinerary Stop", place_options, index=0)
        with h_col2:
            budget_filter = st.selectbox("Budget Tier", ["Any", "budget", "mid_range", "luxury", "premium"], index=0)
        with h_col3:
            sort_by = st.selectbox("Sort By", ["Highest Rating", "Price: Low to High", "Most Reviews"])

        # Determine which places to display hotels for
        if selected_place_opt.startswith("All"):
            places_to_query = planned_places
        else:
            sel_idx = place_options.index(selected_place_opt) - 1
            places_to_query = [planned_places[sel_idx]]

        total_hotels_displayed = 0

        for p_info in places_to_query:
            p_id = p_info["place_id"]
            p_name = p_info["name"]
            p_dist = p_info["district"]
            p_day = p_info["day"]

            matched_hotels = hotel_service.hotels_for_place(
                place_id=p_id,
                hotels_df=hotels_df,
                hotel_reviews_df=hotel_reviews_df,
                review_insights_df=review_insights_df,
                max_hotels=3,
                district=p_dist,
            )

            if not matched_hotels.empty:
                if budget_filter != "Any":
                    matched_hotels = matched_hotels[matched_hotels["tier"].str.lower() == budget_filter.lower()]

                if sort_by == "Highest Rating" and "mean_star_rating" in matched_hotels.columns:
                    matched_hotels = matched_hotels.sort_values(by="mean_star_rating", ascending=False)
                elif sort_by == "Most Reviews" and "review_count" in matched_hotels.columns:
                    matched_hotels = matched_hotels.sort_values(by="review_count", ascending=False)

                if matched_hotels.empty:
                    continue

                st.markdown(f"""
                <div style="margin: 1.5rem 0 0.75rem; display:flex; align-items:center; gap:0.5rem;">
                  <span class="editorial-badge-pill badge-gold">Day {p_day} Destination</span>
                  <h3 style="font-family:'Outfit', sans-serif !important; font-size:1.35rem; margin:0; font-weight:800; color:#2E1B0A;">
                    Stays near {html.escape(p_name)}
                  </h3>
                  <span style="color:#6B5238; font-size:0.85rem;">({html.escape(p_dist)})</span>
                </div>
                """, unsafe_allow_html=True)

                for _, h_row in matched_hotels.iterrows():
                    total_hotels_displayed += 1
                    h_id = h_row.get("hotel_id", "")
                    h_name = h_row.get("hotel_name", "Hotel")
                    h_tier = str(h_row.get("tier", "Mid-Range")).replace("_", " ").title()
                    h_star = h_row.get("star_rating", "3 Star")
                    h_price = h_row.get("price_range", "INR 1,200–3,500/night")
                    h_amenities = str(h_row.get("amenities", "Free Wi-Fi, AC rooms, Parking"))

                    # Review insights
                    h_rating = float(h_row.get("mean_star_rating", 4.2)) if pd.notna(h_row.get("mean_star_rating")) else 4.2
                    h_rev_cnt = int(h_row.get("review_count", 40)) if pd.notna(h_row.get("review_count")) else 40
                    vader_score = float(h_row.get("mean_vader_compound", 0.45)) if pd.notna(h_row.get("mean_vader_compound")) else 0.45
                    dominant_topic = str(h_row.get("dominant_topic", "Cleanliness & Service")).title()

                    st.markdown(f"""
                    <div class="editorial-stop-card">
                      <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                          <div style="display:flex; gap:0.4rem; margin-bottom:0.35rem; flex-wrap:wrap;">
                            <span class="editorial-badge-pill badge-gold">{h_tier} &bull; {h_star}</span>
                            <span class="editorial-badge-pill badge-soft">Matched to: {html.escape(p_name)}</span>
                          </div>
                          <h3 style="font-family:'Outfit', sans-serif !important; font-size:1.35rem; margin:0.25rem 0 0.2rem; font-weight:800; color:#2E1B0A;">{html.escape(h_name)}</h3>
                          <p style="color:#8C5A20; font-weight:600; font-size:0.86rem; margin:0 0 0.5rem;">{html.escape(str(h_row.get('district', p_dist)))}</p>
                        </div>
                        <div style="text-align:right;">
                          <span style="font-size:1.25rem; font-weight:800; color:#C8963E;">{html.escape(str(h_price))}</span>
                          <br><span style="font-size:0.75rem; color:#6B5238;">per night</span>
                        </div>
                      </div>
                      <p style="color:#382613; font-size:0.88rem; margin-bottom:0.6rem;">Amenities: {html.escape(h_amenities[:140])}</p>
                      <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                        <span class="editorial-badge-pill badge-soft">Guest Rating: {h_rating:.1f}/5 ({h_rev_cnt} reviews)</span>
                        <span class="editorial-badge-pill badge-soft">Highlight: {dominant_topic}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        if total_hotels_displayed == 0:
            st.info("No accommodations found matching the current budget filter for the planned itinerary stops.")

    else:
        # When no itinerary is active yet, prompt the user or allow selecting any predicted place
        st.markdown("""
        <div id="hotels" style="margin: 1.5rem 0 1.5rem;">
          <span class="editorial-badge-pill badge-gold">Hospitality & Stays</span>
          <h2 style="font-size: 2.2rem; font-weight: 800; margin: 0.35rem 0 0.5rem; letter-spacing: -0.02em; color: var(--text-espresso); font-family: 'Outfit', sans-serif;">
            Hotels for Itinerary Destinations
          </h2>
          <p style="color: #6B5238; font-size: 0.95rem; margin: 0;">
            Accommodations directly matched to destinations predicted by the AI itinerary planner.
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#FAF2E4; border:1.5px solid #D4B37C; border-radius:18px; padding:1.5rem 2rem; margin-bottom:1.75rem;">
          <h3 style="color:#2E1B0A; font-family:'Outfit', sans-serif !important; font-size:1.35rem; margin:0 0 0.4rem 0; font-weight:800;">
            Plan a Trip to See Matched Route Hotels
          </h3>
          <p style="color:#382613; font-size:0.95rem; margin:0 0 0.5rem 0; line-height:1.5;">
            Use the search panel above to generate a curated multi-day itinerary. The system will automatically select authentic hotels located near each day's scheduled destinations.
          </p>
          <p style="color:#8C5A20; font-weight:600; font-size:0.9rem; margin:0;">
            Or choose any verified destination below to preview its nearby accommodations.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Place picker for previewing hotels for any place
        sorted_places = places_df.sort_values(by="name")
        place_names = sorted_places["name"].tolist()
        preview_col1, preview_col2 = st.columns([2, 1.2])

        with preview_col1:
            selected_place_name = st.selectbox(
                "Select a Destination to View Nearby Accommodations",
                place_names,
                index=place_names.index("Sri Venkateswara Swamy Vaari Temple") if "Sri Venkateswara Swamy Vaari Temple" in place_names else 0,
            )

        with preview_col2:
            budget_filter = st.selectbox("Budget Tier", ["Any", "budget", "mid_range", "luxury", "premium"], index=0, key="prev_budget")

        chosen_row = sorted_places[sorted_places["name"] == selected_place_name].iloc[0]
        c_pid = chosen_row["place_id"]
        c_dist = chosen_row["district"]

        matched_hotels = hotel_service.hotels_for_place(
            place_id=c_pid,
            hotels_df=hotels_df,
            hotel_reviews_df=hotel_reviews_df,
            review_insights_df=review_insights_df,
            max_hotels=5,
            district=c_dist,
        )

        if budget_filter != "Any" and not matched_hotels.empty:
            matched_hotels = matched_hotels[matched_hotels["tier"].str.lower() == budget_filter.lower()]

        st.markdown(f"<p style='font-size:0.92rem; font-weight:700; color:#2E1B0A; margin:1rem 0 0.5rem;'>Showing {len(matched_hotels)} verified hotels near {html.escape(selected_place_name)} ({html.escape(c_dist)}):</p>", unsafe_allow_html=True)

        for _, h_row in matched_hotels.iterrows():
            h_id = h_row.get("hotel_id", "")
            h_name = h_row.get("hotel_name", "Hotel")
            h_tier = str(h_row.get("tier", "Mid-Range")).replace("_", " ").title()
            h_star = h_row.get("star_rating", "3 Star")
            h_price = h_row.get("price_range", "INR 1,200–3,500/night")
            h_amenities = str(h_row.get("amenities", "Free Wi-Fi, AC rooms, Parking"))

            h_rating = float(h_row.get("mean_star_rating", 4.2)) if pd.notna(h_row.get("mean_star_rating")) else 4.2
            h_rev_cnt = int(h_row.get("review_count", 40)) if pd.notna(h_row.get("review_count")) else 40
            vader_score = float(h_row.get("mean_vader_compound", 0.45)) if pd.notna(h_row.get("mean_vader_compound")) else 0.45
            dominant_topic = str(h_row.get("dominant_topic", "Cleanliness & Service")).title()

            st.markdown(f"""
            <div class="editorial-stop-card">
              <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div>
                  <span class="editorial-badge-pill badge-gold">{h_tier} &bull; {h_star}</span>
                  <h3 style="font-family:'Outfit', sans-serif !important; font-size:1.35rem; margin:0.25rem 0 0.2rem; font-weight:800; color:#2E1B0A;">{html.escape(h_name)}</h3>
                  <p style="color:#8C5A20; font-weight:600; font-size:0.86rem; margin:0 0 0.5rem;">{html.escape(str(h_row.get('district', c_dist)))} &bull; Near {html.escape(selected_place_name)}</p>
                </div>
                <div style="text-align:right;">
                  <span style="font-size:1.25rem; font-weight:800; color:#C8963E;">{html.escape(str(h_price))}</span>
                  <br><span style="font-size:0.75rem; color:#6B5238;">per night</span>
                </div>
              </div>
              <p style="color:#382613; font-size:0.88rem; margin-bottom:0.6rem;">Amenities: {html.escape(h_amenities[:140])}</p>
              <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
                <span class="editorial-badge-pill badge-soft">Guest Rating: {h_rating:.1f}/5 ({h_rev_cnt} reviews)</span>
                <span class="editorial-badge-pill badge-soft">Highlight: {dominant_topic}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<hr style="margin-top: 3.5rem; border: none; border-top: 1px solid #E2E8F0;">
<div style="text-align: center; padding: 1.5rem 0 1rem; color: #94A3B8; font-size: 0.85rem;">
  <strong>TripSense</strong> &bull; AI-Powered Andhra Pradesh Tourism Platform &bull; Grounded in 444 authentic destination records
</div>
""", unsafe_allow_html=True)
