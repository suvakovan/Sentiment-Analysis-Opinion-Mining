# Sentiment Analysis & Opinion Mining

<p align="center">
	<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=28&pause=1000&center=true&vCenter=true&width=760&lines=Sentiment+Analysis+%26+Opinion+Mining;Naive+Bayes+%7C+Logistic+Regression+%7C+LSTM;Amazon+Review+Classification+Dashboard" alt="Animated typing header" />
</p>

<p align="center">
	<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,100:1d4ed8&height=160&section=header&text=Opinion%20Mining%20Project&fontSize=38&fontColor=ffffff&animation=fadeIn" alt="Animated wave header" />
</p>

<p align="center">
	Comparative sentiment classification project for Amazon product reviews using <strong>Naive Bayes</strong>, <strong>Logistic Regression</strong>, and <strong>LSTM</strong>.
</p>

<p align="center">
	<img src="https://img.shields.io/badge/Models-3-0f766e?style=for-the-badge" alt="Models count" />
	<img src="https://img.shields.io/badge/Task-Binary%20Sentiment%20Classification-1d4ed8?style=for-the-badge" alt="Task badge" />
	<img src="https://img.shields.io/badge/Interface-Flask%20Dashboard-111827?style=for-the-badge" alt="Interface badge" />
</p>

## What This Repo Contains

- a sentiment analysis pipeline for Amazon product reviews
- model training for Naive Bayes, Logistic Regression, and LSTM
- a Flask dashboard for live predictions and comparison
- notebooks, results, plots, and saved artifacts

## Reported Metrics

The current comparison file at `sentiment-analysis-project/results/model_comparison.csv` reports the following scores:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| Naive Bayes | 1.00 | 1.00 | 1.00 | 1.00 |
| Logistic Regression | 1.00 | 1.00 | 1.00 | 1.00 |
| LSTM | 1.00 | 1.00 | 1.00 | 1.00 |

## Main Project

The full project lives in [sentiment-analysis-project/](sentiment-analysis-project/).

## Run the App

```bash
cd sentiment-analysis-project
python app.py
```

Then open http://127.0.0.1:5000 in your browser.