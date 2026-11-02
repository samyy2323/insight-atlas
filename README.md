# Insight Atlas

An interactive dashboard that explores 30 years of global socioeconomic trends and adds three small machine-learning views.

Built for the Data Visualization Lab (BISL504-C), The National Institute of Engineering, Mysuru.

- 10 interactive graphs: world map, animated bubble chart, trends, rankings, treemap and more
- Country Explorer: compare up to four countries on a radar chart
- Machine learning: life-expectancy predictor (Ridge regression), country clusters (K-Means) and a forecast to 2030
- Data: practice dataset, 194 countries, 1994-2024 (slimmed to the columns the app uses)

Run locally:

```
pip install -r requirements.txt
streamlit run insight_atlas.py
```
