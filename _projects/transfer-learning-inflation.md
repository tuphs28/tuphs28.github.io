---
layout: page
title: Improve Inflation Forecasts With BERT
description: Using a fine-tuned language model to extract measures of economic activity in order to improve inflation forecasts.
img: assets/img/projects/inflation/pctpos.png
importance: 1
category: data-science
related_publications: true
---

This post overviews one of my undergraduate dissertations. This dissertation investigated using a fine-tuned foundation model to extract proxies for economic indicators from naturally-occuring text data. This is of interest since leveraging transfer learning could plausibly allow us to extract high-fidelity measurements of economic phenomena that would otherwise be too expensive or too impractical to obtain. I focused on extracting measures of economic sentiment - since consumer and producer sentiment is usually measured through infrequent surveys that only imperfectly capture the desired phenomenon - for use in improving inflation forecasts in Nigeria. The measures of economic sentiment I constructed were taken from the headlines of all major Nigerian newspapers over a 10 year period.

### Background

Before outlining this project in further detail, it is useful to first briefly overview *why* we would expect newspaper headlines to improve inflation forecasts. One reason for this is that, empirically, newspaper sentiment has already been used to improve forecasts of macroeconomic indicators in countries like the UK {% cite Kalamara2022making %} and USA {% cite shapiro2022measuring %}.