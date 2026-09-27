"""
app.py
------
Flask web application for Fake News Detection.

Loads the pre-trained TF-IDF vectorizer and Logistic Regression model,
accepts news headlines from users, predicts FAKE/REAL, saves results
to Supabase, and provides an admin dashboard.

Run with: python app.py
"""

import os
from datetime import datetime
from functools import wraps

from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
import joblib

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fake-news-detector-secret-key-change-me")

# ── Load ML model and vectorizer ──────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
else:
    print("WARNING: Model files not found. Please run 'python train_model.py' first.")

# ── Supabase setup ────────────────────────────────────────────────────
supabase_client = None

try:
    from supabase import create_client

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if SUPABASE_URL and SUPABASE_KEY:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase client initialized successfully.")
    else:
        print("WARNING: Supabase credentials not set in .env file.")
        print("  Predictions will still work, but history won't be saved.")
except ImportError:
    print("WARNING: supabase package not installed. History saving disabled.")
except Exception as e:
    print(f"WARNING: Failed to initialize Supabase: {e}")

# ── Admin credentials ────────────────────────────────────────────────
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")


# ── Helper: save prediction to Supabase ──────────────────────────────
def save_prediction(headline, prediction):
    """Save a prediction record to Supabase. Fails silently if DB unavailable."""
    if supabase_client is None:
        print("  Supabase not available — skipping save.")
        return False

    try:
        data = {
            "headline": headline,
            "prediction": prediction,
            "created_at": datetime.now().isoformat()
        }
        supabase_client.table("predictions").insert(data).execute()
        print(f"  Prediction saved to Supabase.")
        return True
    except Exception as e:
        print(f"  ERROR saving to Supabase: {e}")
        return False


# ── Helper: get all predictions from Supabase ────────────────────────
def get_predictions():
    """Fetch all predictions from Supabase, ordered by newest first."""
    if supabase_client is None:
        return []

    try:
        response = (
            supabase_client.table("predictions")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"  ERROR fetching from Supabase: {e}")
        return []


# ── Decorator: require admin login ───────────────────────────────────
def login_required(f):
    """Simple decorator to protect admin routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for("admin_login"))
        return f(*args, **kwargs)
    return decorated_function


# ══════════════════════════════════════════════════════════════════════
#   ROUTES
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    """Display the homepage with the headline input form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive a headline, predict FAKE/REAL, save to DB, show result."""

    headline = request.form.get("headline", "").strip()

    # Validate input
    if not headline:
        flash("Please enter a news headline.", "error")
        return redirect(url_for("home"))

    # Check model is loaded
    if model is None or vectorizer is None:
        flash("Model not available. Please run 'python train_model.py' first.", "error")
        return redirect(url_for("home"))

    # Make prediction
    headline_lower = headline.lower()
    headline_tfidf = vectorizer.transform([headline_lower])
    prediction = model.predict(headline_tfidf)[0]  # "FAKE" or "REAL"

    print(f"  Headline: {headline}")
    print(f"  Prediction: {prediction}")

    # Save to Supabase (non-blocking — doesn't affect result display)
    db_saved = save_prediction(headline, prediction)
    if not db_saved:
        print("  Note: Prediction was not saved to database.")

    return render_template("result.html", headline=headline, prediction=prediction)


# ── Admin routes ──────────────────────────────────────────────────────

@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    """Simple admin login page."""
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["admin_logged_in"] = True
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid username or password.", "error")

    return render_template("admin_login.html")


@app.route("/admin")
@login_required
def admin_dashboard():
    """Display the admin dashboard with prediction history."""
    predictions = get_predictions()

    # Calculate statistics
    total = len(predictions)
    fake_count = sum(1 for p in predictions if p.get("prediction") == "FAKE")
    real_count = sum(1 for p in predictions if p.get("prediction") == "REAL")

    return render_template(
        "admin.html",
        predictions=predictions,
        total=total,
        fake_count=fake_count,
        real_count=real_count
    )


@app.route("/admin/logout")
def admin_logout():
    """Log out the admin."""
    session.pop("admin_logged_in", None)
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=5000)
