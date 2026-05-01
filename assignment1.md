# Assignment 1 – Predicting NBA Player Performance

## Project Overview

The goal of this project is to build a supervised regression model to predict NBA player performance based on historical and contextual statistics.

Instead of focusing only on raw statistics, this project aims to understand how different player attributes and game-related features influence overall performance.

The target variable will be a continuous performance metric such as Player Efficiency Rating (PER) or plus/minus per game.

---

## Dataset

This project will use an existing publicly available dataset, such as:

- NBA Player Stats dataset (Kaggle)
- Historical NBA performance datasets (1950–2023)
- Advanced basketball statistics datasets

The dataset contains player-level statistics across multiple seasons and games.

---

## Target Variable

The target variable is a continuous measure of player performance, such as:

- Player Efficiency Rating (PER), or
- Plus/Minus per game

This makes the problem a supervised regression task.

---

## Feature Engineering

Several types of features will be constructed:

### 1. Basic performance statistics
- Points per game
- Assists
- Rebounds
- Turnovers
- Minutes played

### 2. Shooting efficiency
- Field goal percentage (FG%)
- Three-point percentage (3P%)
- Free throw percentage (FT%)

### 3. Player profile features
- Age
- Height and weight
- Position encoding

### 4. Recent performance trends
- Rolling averages over the last 5 games
- Performance variability

---

## Models

The following regression models will be tested:

- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- Random Forest Regressor
- XGBoost Regressor (if available)

---

## Evaluation Metrics

Model performance will be evaluated using:

- Mean Absolute Error (MAE)
- R² score

The results will be compared to a baseline model using average player performance.

---

## Objective

The objective of this project is to apply supervised learning techniques to a real-world sports dataset and understand how different features contribute to predicting player performance in basketball.
