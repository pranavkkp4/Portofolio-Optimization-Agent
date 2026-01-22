News-Aware Portfolio Optimization Agent

Reinforcement Learning • Time Series • NLP • FastAPI

Overview

This project is an end-to-end machine learning system that demonstrates how sequential decision-making models can combine time-series market signals with natural-language news sentiment to produce conservative, interpretable portfolio recommendations.

The system is designed to showcase model design and optimization, feature engineering, reinforcement learning, and production-oriented deployment—all core skills for a Machine Learning Engineer role.

Problem Statement

Portfolio decisions are inherently sequential and uncertain.
In real-world ML systems, teams must:

Combine structured time-series data (returns, momentum)

Incorporate unstructured signals (news text)

Optimize decisions under delayed rewards

Deploy models in a reliable, inspectable production environment

This project addresses those challenges with a compact but realistic ML pipeline.

Solution Architecture

The system consists of four main components:

Feature Extraction

Time-series signal: recent return regime

NLP signal: sentiment polarity from news text

Reinforcement Learning Policy

Tabular Q-learning agent

Learns a policy mapping market states → portfolio actions

Inference & Deployment

FastAPI backend serving predictions

Stateless, production-style API design

Interpretability & UI

Web interface for interactive inference

“Developer Insights” panel exposing model internals

How It Works
1. Time-Series Processing

The user provides a recent market return (e.g., previous day return).
This value is discretized into a return regime (negative / neutral / positive), forming part of the RL state.

2. NLP Feature Engineering

A news headline is processed into a sentiment polarity score (negative / neutral / positive).
This unstructured signal is fused with the time-series regime.

3. State Representation

The combined features form a compact discrete state:

state = f(return_regime, sentiment_polarity)

4. Reinforcement Learning Inference

A trained Q-learning policy evaluates the state and outputs Q-values for each action:

0: Stay in cash

1: Invest (go long)

The action with the highest Q-value is selected.

5. Interpretability

The UI exposes:

State ID

Sentiment score

Q-values per action

Selected action

This makes the decision process transparent and debuggable, which is critical for production ML systems.

Machine Learning Details
State Space

Return regime: {negative, neutral, positive}

Sentiment polarity: {negative, neutral, positive}

Combined into a discrete state index

Action Space

0: Stay in cash

1: Invest

Learning Algorithm

Tabular Q-learning

Off-policy, value-based reinforcement learning

Exploration during training, greedy policy at inference

Reward Signal

Positive reward when invested during positive return regimes

Neutral/negative reward when invested during adverse conditions

Cash action provides a conservative baseline

Why This Project Is Relevant for ML Engineering

This project demonstrates:

Model design & optimization

State abstraction and reward shaping

Conservative policy behavior under uncertainty

Time-series reasoning

Regime-based discretization

Sequential decision context

NLP processing

Feature extraction from unstructured text

Fusion of text + numeric signals

Reinforcement learning

Policy learning over time

Action-value estimation

Production deployment

FastAPI inference service

Clean separation of training vs inference

Inspectable outputs for debugging

MLOps awareness

Deterministic inference

Model artifact loading (model.pkl)

Clear retraining workflow

Web Interface

The web UI allows users to:

Enter a return value and news headline

Receive a portfolio recommendation

Inspect internal model reasoning via Developer Insights

Screenshots of the UI (before input and after inference with insights) are included in the project’s GitHub Pages site.

Running the Project Locally
1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies
pip install -r requirements.txt
pip install python-multipart

3. Train the model

This generates the Q-table used for inference.

python train.py

4. Start the FastAPI server
uvicorn api.main:app --reload

5. Open the app
http://127.0.0.1:8000/

GitHub Pages Project Site

This repository also includes a static project site used as a portfolio showcase.

Preview locally
python3 -m http.server 5500


Open:

http://127.0.0.1:5500/docs/

Publish on GitHub Pages

Repo → Settings → Pages

Source: Deploy from branch

Branch: main

Folder: /docs

Limitations & Future Improvements

To move from a portfolio demonstration to a production-ready trading system:

Replace synthetic inputs with point-in-time aligned market and news data

Add transaction costs and slippage

Implement walk-forward backtesting

Expand action space to include position sizing

Replace tabular Q-learning with function approximation (e.g., DQN)

These extensions are intentionally left out to keep the project focused on clarity, interpretability, and core ML concepts.

Technologies Used

Python

NumPy

Scikit-learn

FastAPI

Jinja2

Reinforcement Learning (Q-learning)

Time-series feature engineering

NLP sentiment processing

One-Line Summary

Reinforcement-learning portfolio optimization agent combining time-series market signals and NLP sentiment, deployed via FastAPI with an interpretable web interface.
