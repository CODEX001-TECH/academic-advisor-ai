
from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "academic_advisor_model.pkl"

app = Flask(__name__)

# Names expected by the current saved model.
MODEL_FEATURES = [
    "cgpa",
    "attendance",
    "study_hours",
    "failed_courses",
    "credits_completed",
]

ADVICE = {
    "Low": {
        "title": "Good academic standing",
        "items": [
            "Keep attending lectures and maintain your study routine.",
            "Review lecture notes weekly and start assignments early.",
            "Set one measurable academic goal for the next two weeks.",
        ],
    },
    "Medium": {
        "title": "Some areas need attention",
        "items": [
            "Create a weekly timetable with focused study sessions.",
            "Ask your lecturer or academic adviser about difficult topics.",
            "Improve attendance and practise past questions consistently.",
        ],
    },
    "High": {
        "title": "Priority academic support recommended",
        "items": [
            "Contact your department's academic adviser.",
            "Identify difficult courses and create a recovery plan.",
            "Seek help from a tutor or study group and study daily.",
        ],
    },
}

HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Academic Advisor AI</title>
<style>
body {
    font-family: Arial, sans-serif;
    background: #f3f6fb;
    color: #172033;
    margin: 0;
}
.wrap {
    max-width: 760px;
    margin: 35px auto;
    padding: 20px;
}
.card {
    background: white;
    padding: 26px;
    border-radius: 16px;
    box-shadow: 0 8px 30px #14213d12;
}
h1 { margin: 0 0 8px; color: #173f8a; }
.muted { color: #596579; line-height: 1.5; }
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
}
label {
    display: block;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 6px;
}
input {
    box-sizing: border-box;
    width: 100%;
    padding: 11px;
    border: 1px solid #ccd5e2;
    border-radius: 8px;
}
.field { margin: 8px 0 10px; }
button {
    margin-top: 15px;
    background: #1769e0;
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 13px 18px;
    font-weight: bold;
    cursor: pointer;
    width: 100%;
}
.result {
    margin-top: 22px;
    padding: 18px;
    background: #eef5ff;
    border-left: 5px solid #1769e0;
    border-radius: 8px;
}
.error {
    margin-top: 18px;
    color: #9b1c1c;
    background: #fff0f0;
    padding: 12px;
    border-radius: 8px;
}
li { margin: 8px 0; }
@media (max-width: 600px) {
    .grid { grid-template-columns: 1fr; }
    .wrap { margin: 10px auto; }
}
</style>
</head>
<body>
<div class="wrap">
<div class="card">
<h1>Academic Advisor AI</h1>

<p class="muted">
Enter your academic information to receive an estimated academic-risk
category and practical study recommendations. This is a learning
prototype, not an official university decision tool.
</p>

<form method="post">
<div class="grid">
<div class="field">
<label for="cgpa">Current CGPA (1.00–5.00)</label>
<input id="cgpa" name="cgpa" type="number"
min="1" max="5" step="0.01"
value="{{ values.get('cgpa', '3.50') }}" required>
</div>

<div class="field">
<label for="attendance_pct">Attendance (%)</label>
<input id="attendance_pct" name="attendance_pct" type="number"
min="0" max="100" step="0.1"
value="{{ values.get('attendance_pct', '80') }}" required>
</div>

<div class="field">
<label for="study_hours_per_day">Study hours per day</label>
<input id="study_hours_per_day" name="study_hours_per_day"
type="number" min="0" max="24" step="0.1"
value="{{ values.get('study_hours_per_day', '2') }}" required>
</div>

<div class="field">
<label for="failed_courses">Number of failed courses</label>
<input id="failed_courses" name="failed_courses" type="number"
min="0" max="30" step="1"
value="{{ values.get('failed_courses', '0') }}" required>
</div>

<div class="field">
<label for="credits_completed">Credits completed</label>
<input id="credits_completed" name="credits_completed" type="number"
min="0" max="300" step="1"
value="{{ values.get('credits_completed', '45') }}" required>
</div>
</div>

<button type="submit">Generate Academic Advice</button>
</form>

{% if error %}
<div class="error">{{ error }}</div>
{% endif %}

{% if prediction %}
<div class="result">
<h2>Predicted risk: {{ prediction }}</h2>
<strong>{{ advice.title }}</strong>
<ul>
{% for item in advice["items"] %}
<li>{{ item }}</li>
{% endfor %}
</ul>
<p class="muted">
Model confidence: {{ confidence }}%.
Discuss important academic decisions with your department adviser.
</p>
</div>
{% endif %}

</div>
</div>
</body>
</html>
"""


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file is missing. Place academic_advisor_model.pkl "
            "in the same folder as app.py."
        )

    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)


@app.route("/", methods=["GET", "POST"])
def home():
    values = request.form.to_dict() if request.method == "POST" else {}

    prediction = None
    advice = None
    confidence = None
    error = None

    if request.method == "POST":
        try:
            # Read values from the web form.
            data = {
                "cgpa": float(request.form["cgpa"]),
                "attendance_pct": float(
                    request.form["attendance_pct"]
                ),
                "study_hours_per_day": float(
                    request.form["study_hours_per_day"]
                ),
                "failed_courses": int(
                    request.form["failed_courses"]
                ),
                "credits_completed": int(
                    request.form["credits_completed"]
                ),
            }

            # Validate the submitted information.
            if not 1 <= data["cgpa"] <= 5:
                raise ValueError("CGPA must be between 1 and 5.")

            if not 0 <= data["attendance_pct"] <= 100:
                raise ValueError(
                    "Attendance must be between 0 and 100."
                )

            if not 0 <= data["study_hours_per_day"] <= 24:
                raise ValueError(
                    "Study hours must be between 0 and 24."
                )

            if data["failed_courses"] < 0:
                raise ValueError(
                    "Failed courses cannot be negative."
                )

            if not 0 <= data["credits_completed"] <= 300:
                raise ValueError(
                    "Credits completed must be between 0 and 300."
                )

            # Load the trained model.
            model = load_model()

            # Map form values to the saved model's feature names.
            input_data = pd.DataFrame(
                [[
                    data["cgpa"],
                    data["attendance_pct"],
                    data["study_hours_per_day"],
                    data["failed_courses"],
                    data["credits_completed"],
                ]],
                columns=MODEL_FEATURES,
            )

            # Generate the prediction.
            prediction = str(model.predict(input_data)[0])

            # Calculate confidence when supported by the model.
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = round(
                    float(max(probabilities) * 100), 1
                )
            else:
                confidence = "N/A"

            # Select recommendations for the predicted category.
            if prediction not in ADVICE:
                raise ValueError(
                    f"Unexpected model category: {prediction}. "
                    "Check the labels used to train the model."
                )

            advice = ADVICE[prediction]

        except Exception as exc:
            error = str(exc)

    return render_template_string(
        HTML,
        values=values,
        prediction=prediction,
        advice=advice,
        confidence=confidence,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
