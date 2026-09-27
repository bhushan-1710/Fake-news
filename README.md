# Fake News Detector 📰

A web app that predicts whether a news headline is **Fake** or **Real** using machine learning. Built with Flask and scikit-learn.

This was built as a college project — it's intentionally kept simple and easy to understand.

## What it does

- You enter a news headline on the homepage
- The ML model (TF-IDF + Logistic Regression) classifies it as FAKE or REAL
- The prediction gets saved to a Supabase database
- There's a basic admin panel to view prediction history

## Tech stack

- **Backend:** Python, Flask
- **ML:** scikit-learn (TF-IDF vectorizer + Logistic Regression)
- **Frontend:** HTML, CSS, vanilla JS
- **Database:** Supabase (PostgreSQL)
- **Other:** pandas, joblib, python-dotenv

## How the model works

Pretty straightforward ML pipeline:

1. Load a labeled dataset of news headlines (FAKE/REAL)
2. Clean the text (lowercase, drop nulls)
3. Convert text to numerical features using TF-IDF
4. Train a Logistic Regression classifier
5. Save the trained model using joblib
6. Flask loads the saved model at startup and uses it for predictions

The model gets ~89% accuracy on the test set which is decent for a simple approach.

## Project structure

```
fake-news-detector/
├── app.py                  # main Flask app
├── train_model.py          # trains and saves the ML model
├── requirements.txt
├── .env                    # your credentials (not committed)
├── .gitignore
├── README.md
├── model/                  # generated model files
│   ├── model.pkl
│   └── vectorizer.pkl
├── dataset/
│   └── news.csv            # training data
├── templates/
│   ├── index.html          # homepage
│   ├── result.html         # shows prediction
│   ├── admin_login.html    # admin login form
│   └── admin.html          # prediction history dashboard
└── static/
    ├── style.css
    └── script.js
```

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/bhushan-1710/Fake-news.git
cd Fake-news
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123

SECRET_KEY=pick-any-random-string
```

### 3. Set up Supabase

Create a free project on [supabase.com](https://supabase.com), then run this SQL in the SQL Editor:

```sql
CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    headline TEXT NOT NULL,
    prediction TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Grab your project URL and anon key from **Settings → API** and put them in `.env`.

### 4. Train the model

```bash
python train_model.py
```

This trains the model on the dataset and saves `model.pkl` and `vectorizer.pkl` to the `model/` folder. You only need to do this once.

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## Admin dashboard

Go to `/admin/login` and use the credentials from your `.env` file. The dashboard shows all past predictions with some basic stats (total, fake count, real count).

## Notes

- The model needs to be trained before you can run the app (step 4)
- If Supabase isn't configured, the app still works — it just won't save prediction history
- This is a demo project with a small dataset. Don't use it for actual fact-checking
