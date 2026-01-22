from pathlib import Path
import pickle
import numpy as np

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Flat-layout import: env.py is in repo root
from env import compute_sentiment, discretize_state


BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MODEL_PATH = BASE_DIR.parent / "model.pkl"

app = FastAPI(title="News-Aware Portfolio Optimization Agent")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def load_q_table() -> np.ndarray:
    """
    Load a saved Q-table from disk. We load on startup for simplicity.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run `python train.py` first to generate model.pkl."
        )
    with open(MODEL_PATH, "rb") as f:
        q = pickle.load(f)
    return q


# Load Q table once at startup
Q_TABLE = load_q_table()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(return_value: float = Form(...), news_text: str = Form(...)) -> dict:
    """
    Accepts:
      - return_value: float (e.g., 0.01 for +1%)
      - news_text: string headline or short snippet

    Returns:
      - action: 0 (cash) or 1 (invest)
      - suggestion: user-friendly string
      - developer_insights: model internals for transparency/debugging
    """
    sentiment_score = compute_sentiment(news_text)
    state_id = discretize_state(return_value, sentiment_score)

    q_values = Q_TABLE[state_id]
    action = int(q_values.argmax())
    suggestion = "Invest (go long)" if action == 1 else "Stay in cash"

    developer_insights = {
        "return_value": float(return_value),
        "sentiment_score": int(sentiment_score),
        "state_id": int(state_id),
        "q_values": [float(x) for x in q_values],
        "chosen_action": int(action),
        "action_map": {"0": "cash", "1": "invest"},
    }

    return {
        "action": int(action),
        "suggestion": suggestion,
        "developer_insights": developer_insights,
    }
