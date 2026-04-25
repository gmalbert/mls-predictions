"""
themes.py
Injects the permanent Oracle Pitch CSS theme into every page.

Usage:
    from themes import apply_theme
    apply_theme()   # call once at module level before pg.run()
"""

import streamlit as st

_CSS = """
<style>
/* ── Oracle Pitch — permanent theme ───────────────────────────────────── */

/* Dark sidebar */
[data-testid="stSidebar"] {
    background-color: #1a2610 !important;
}
[data-testid="stSidebar"] * {
    color: #c8f060 !important;
}
[data-testid="stSidebarNavLink"] {
    color: #c8f060 !important;
}
[data-testid="stSidebarNavLink"]:hover,
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background-color: #5ec40033 !important;
    color: #5ec400 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: #3a8a00 !important;
}

/* Tabs */
[data-testid="stTab"] {
    color: #4e7e28 !important;
    border-bottom: 2px solid #c0e088 !important;
}
[data-testid="stTab"][aria-selected="true"] {
    color: #5ec400 !important;
    border-bottom-color: #5ec400 !important;
}

/* Buttons */
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"] {
    background-color: #5ec400 !important;
    border-color: #5ec400 !important;
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #c0e088 !important;
    border-radius: 8px;
    padding: 10px;
}

/* DataFrames */
[data-testid="stDataFrame"] {
    border: 1px solid #c0e088 !important;
}

/* Links */
a {
    color: #3a8a00 !important;
}
</style>
"""


def apply_theme() -> None:
    """Inject the Oracle Pitch CSS theme. Call once before pg.run()."""
    st.markdown(_CSS, unsafe_allow_html=True)
