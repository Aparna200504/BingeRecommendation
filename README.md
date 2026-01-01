# Binge Recommendation

A lightweight project that provides show/movie recommendations from a Netflix titles dataset and a minimal Flask UI to browse and discover similar titles.

## Accomplishments

- Implemented a basic recommendation engine using the `data/netflix_titles.csv` dataset.
- Exposed recommendations via a small Flask web app (`app.py`) with templates in `templates/`.
- Organized code into a dedicated recommendation module (`recommender.py`) and static assets under `static/`.

## Tech Stack

- Python 3.x
- Flask (web UI)
- pandas (data loading / manipulation)
- scikit-learn (feature extraction / similarity)
- Jinja2 (templating)

## Project Structure

- `app.py` — Flask application entrypoint
- `recommender.py` — Recommendation logic and helper functions
- `requirements.txt` — Python dependencies
- `data/netflix_titles.csv` — Source dataset
- `templates/` — HTML templates (`base.html`, `index.html`)
- `static/` — CSS and other static assets

## Flow

1. Load and preprocess dataset from `data/netflix_titles.csv`.
2. Build feature representations (e.g., text features, metadata) and compute similarity in `recommender.py`.
3. `app.py` serves the web UI where users can search or select a title.
4. On selection, the recommender returns similar titles which are rendered in the template.

## Run locally

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Flask app:

```bash
python app.py
```

Open http://localhost:5000 in your browser.


"# BingeRecommendation" 
