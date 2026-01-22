"""
FastAPI application exposing the News‑Aware Portfolio Optimization agent.

This module defines a simple API with two endpoints:

* ``GET /`` serves a minimal web UI built with HTML and JavaScript. The
  interface allows users to input a recent return and a news headline, then
  calls the API to obtain a trading recommendation.
* ``POST /predict`` accepts form‑encoded data containing ``return_value`` and
  ``news_text`` fields and returns a JSON object with the recommended action
  (0 for cash, 1 for invest) and a human‑readable suggestion.

The agent itself is represented by a Q‑table stored in ``model.pkl`` in the
project root. On startup, this table is loaded from disk. Sentiment
computation and state discretization are delegated to helper functions in
``env.py``.
"""

from __future__ import annotations

import os
import pickle

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from env import compute_sentiment, discretize_state

# Determine paths relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


def load_q_table(path: str) -> list:
    """Load the Q‑table from a pickle file.

    Parameters
    ----------
    path : str
        Path to the pickled Q‑table.

    Returns
    -------
    list
        A nested list or NumPy array containing Q‑values.
    """
    with open(path, "rb") as f:
        q_table = pickle.load(f)
    return q_table


app = FastAPI(title="News‑Aware Portfolio Optimization Agent")

# Mount static files (CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Jinja2 templates for HTML pages
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
templates = Jinja2Templates(directory=templates_dir)

# Load Q‑table at startup
Q_TABLE = load_q_table(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the web interface.

    Parameters
    ----------
    request : Request
        The incoming HTTP request (unused here but required by Jinja2).

    Returns
    -------
    fastapi.responses.HTMLResponse
        The rendered HTML page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(return_value: float = Form(...), news_text: str = Form(...)) -> dict:
    """Compute a trading recommendation.

    Accepts form‑encoded data with two fields:
    ``return_value`` (a decimal representing the most recent asset return) and
    ``news_text`` (a short news headline or description). The function
    computes a sentiment score, discretizes the state and selects the best
    action from the Q‑table. It returns a JSON response with the action and
    a human‑readable suggestion.

    Parameters
    ----------
    return_value : float
        The return of the asset on the previous day (e.g. 0.01 for +1%).
    news_text : str
        A short text summarising relevant news or events.

    Returns
    -------
    dict
        A dictionary containing the action (0 or 1) and a suggestion string.
    """
    sentiment_score = compute_sentiment(news_text)
    state_id = discretize_state(return_value, sentiment_score)
    # Choose the best action based on Q‑values
    action = int(float(Q_TABLE[state_id].argmax()))
    suggestion = "Invest (go long)" if action == 1 else "Stay in cash"
    return {"action": action, "suggestion": suggestion}