# AutoML Trainer

A full-stack web application that lets you upload any CSV dataset, configure and train a real scikit-learn machine learning model, and receive AI-powered interpretation of your results — all through a clean, step-by-step interface.

**Live demo:** [arnavbabel.github.io/AutoMLGenerator](https://arnavbabel.github.io/AutoMLGenerator)

---

## Features

- Upload any CSV dataset (up to 5,000 rows)
- Auto-detects numeric and categorical columns
- Supports 8 ML algorithms across classification and regression
- One-hot encodes categorical features automatically
- Returns performance metrics, feature importance, and an AI-generated interpretation of results
- Downloadable training report

## Supported Models

**Regression**
- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest

**Classification**
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest

## Tech Stack

**Frontend:** HTML, CSS, JavaScript — hosted on GitHub Pages

**Backend:** Python, FastAPI, scikit-learn, pandas — hosted on Render

**AI Interpretation:** Groq API (Llama 3.3 70B)

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/arnavbabel/AutoMLGenerator.git
cd AutoMLGenerator
```

**2. Set up the backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your Groq API key**

Create a `.env` file in the `backend/` folder:
```
GROQ_API_KEY=your_key_here
```

Or export it directly:
```bash
export GROQ_API_KEY=your_key_here
```

**4. Start the backend**
```bash
uvicorn main:app --reload
```

**5. Open the frontend**

Open `index.html` in your browser. Make sure the `API` variable at the top of the script points to `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/upload` | Parse CSV and return column types |
| POST | `/train` | Train a model and return metrics |
| POST | `/interpret` | Generate AI interpretation of results |

---

## Project Structure

```
AutoMLGenerator/
├── index.html          # Frontend
└── backend/
    ├── main.py         # FastAPI app
    ├── requirements.txt
    └── Procfile
```