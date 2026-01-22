# News‑Aware Portfolio Optimization Agent

This project is a self‑contained demonstration of how *reinforcement learning* and
*sentiment analysis* can be combined to make simple trading decisions. The
approach is motivated by research showing that adding qualitative signals from
financial news to an RL trading system can improve performance compared to
models that rely solely on price data. For example, a 2024 study showed that
RL models enhanced with sentiment features achieved higher net worth and
cumulative profits than those without sentiment and even outperformed a
buy‑and‑hold baseline in portfolio experiments【930120649791441†L70-L79】. The same
study emphasised that RL excels at adapting strategies in complex markets but
traditionally ignores qualitative factors such as market mood【930120649791441†L89-L99】;
integrating sentiment helps bridge this gap【930120649791441†L103-L109】.

## Project overview

The repository contains a minimal end‑to‑end implementation of a *news‑aware
portfolio optimization agent*. It includes:

* **Synthetic data generator**: Creates price and sentiment series for training. A
  random walk with a small upward drift simulates the asset price, while
  discrete sentiment scores \(-1, 0, 1\) model daily market mood.
* **Trading environment**: Implements a simple reinforcement learning
  environment (`TradingEnv` in `env.py`) where the agent decides between two
  actions: invest fully in the asset or stay in cash. Rewards are proportional
  to the daily return if the agent invests.
* **Q‑learning trainer**: Uses Q‑learning to learn a policy that maps
  discretised price returns and sentiment signals to actions. After training,
  the resulting Q‑table is saved as `model.pkl`.
* **FastAPI server**: Exposes a small API (`api/main.py`) with two endpoints:
  - `GET /` serves a dark‑themed web interface where users can input the
    previous day’s return and a news headline and obtain a recommendation.
  - `POST /predict` accepts form data and returns a JSON object with the
    recommended action and a human‑readable suggestion.
* **Web UI**: A minimal HTML/JavaScript page (`api/templates/index.html`) and
  associated CSS (`api/static/style.css`) implementing a dark background with
  blue and silver accents. The front‑end sends requests to the API and displays
  the resulting recommendation.

Despite the simplicity of the demo—synthetic data, discrete states and
coarse sentiment—the project illustrates how one can combine time‑series data,
NLP techniques and RL in a cohesive system. It serves as a template that can
be extended with real price feeds, richer sentiment analysis (e.g. FinBERT
embeddings) and more sophisticated RL algorithms (e.g. deep Q‑networks or
policy gradient methods)【930120649791441†L70-L79】.

## Installation and setup

This project was developed using Python 3.9 and requires only a few third‑party
libraries. To set up a virtual environment and install dependencies, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn jinja2 numpy scikit‑learn
```

Because the environment uses only NumPy and no deep learning libraries, it can
run on modest hardware without GPU support. FastAPI and Uvicorn are used to
serve the API and web UI.

## Training the agent

To train the Q‑learning agent and generate a Q‑table, run:

```bash
cd news_portfolio_rl
python train.py
```

This command produces a file called `model.pkl` in the `news_portfolio_rl`
directory. The script uses 200 training episodes by default; you can adjust
`EPISODES` in `train.py` if you wish to train longer.

### Understanding the synthetic environment

The agent observes a discrete state representing two features:

1. **Price return**: Categorised into down, flat and up buckets using ±0.2 % thresholds.
2. **Sentiment score**: Computed from news text via a simple keyword list
   (`env.POSITIVE_KEYWORDS` and `env.NEGATIVE_KEYWORDS`), producing −1,
   0 or +1.

These two dimensions create 9 possible states. The action space is {0 = stay
in cash, 1 = invest fully}. Rewards are calculated as the product of the
agent’s action and the asset’s return. Over many episodes the Q‑learning
algorithm learns a table of expected returns for each state–action pair and
tends to favour investing when recent returns and sentiment are positive.

## Running the API and UI

After training, start the FastAPI server from the project root:

```bash
uvicorn news_portfolio_rl.api.main:app --reload
```

Visit `http://127.0.0.1:8000/` in your browser to load the web interface. The
page allows you to enter a recent return and a news headline. Press “Get
Recommendation” and the agent will respond with either **Invest (go long)** or
**Stay in cash**.

### Deploying to GitHub Pages

To publish the web UI via GitHub Pages, you can generate a static version of
`index.html` and include the contents of `api/static` and `api/templates` in
your repository’s `/docs` folder. GitHub Pages serves static assets from the
`docs` directory by default. The API requires a Python server and therefore
cannot run on GitHub Pages alone; however, you can host the API separately or
use the static UI to call an API deployed on another service.

## Customising the agent

This project is intentionally simple to make it easy to understand and extend.
Here are some ideas for further development:

* **Use real data:** Replace `create_synthetic_data` with a loader for
  historical price series (e.g. from Yahoo Finance or a CSV) and parse
  timestamps of actual news articles. See the literature for examples of
  sentiment‑enhanced RL outperforming baselines【930120649791441†L70-L79】.
* **Improve sentiment analysis:** Swap the keyword list for a more
  sophisticated model such as VADER or a transformer (e.g. BERT). Financial
  news sentiment has been shown to correlate with short‑term price movements
  and volatility【930120649791441†L143-L149】.
* **Enhance the RL algorithm:** Replace the discrete Q‑learning with a
  function‑approximation method (e.g. deep Q‑networks or PPO) to handle
  continuous state spaces. Research demonstrates that RL agents adaptively
  optimise trading strategies by maximising cumulative returns【930120649791441†L129-L134】.

## Directory structure

```
news_portfolio_rl/
├── api/
│   ├── main.py         # FastAPI application
│   ├── static/
│   │   └── style.css   # Dark‑themed CSS
│   └── templates/
│       └── index.html  # Web UI template
├── env.py              # Environment and helper functions
├── train.py            # Q‑learning training script
├── model.pkl           # Saved Q‑table (generated after training)
├── README.md           # Project documentation
└── requirements.txt    # Dependencies (optional)
```

## Requirements file

A `requirements.txt` file is provided for convenience. Install dependencies via:

```bash
pip install -r requirements.txt
```

## License

This project is released under the MIT License. See `LICENSE` for details.