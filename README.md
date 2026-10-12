# Academic Advisor AI (CSC 303)

A small academic-advising prototype combining standard software engineering practices with a machine-learning classifier. It predicts a synthetic academic-risk category (Low, Medium, High) from CGPA, attendance, study hours, failed courses, and credits completed, then displays study recommendations.

## Project files
- `app.py` — Flask web application
- `student_data.csv` — synthetic practice dataset; not real student records
- `Academic_Advisor_Model_Training.ipynb` — run in Google Colab to train/evaluate the model and export the `.pkl` file
- `academic_advisor_model.pkl` — model artifact; generate it from the notebook and put it beside `app.py`
- `requirements.txt` — Python dependencies

## Run the model training in Google Colab
1. Upload `Academic_Advisor_Model_Training.ipynb` and `student_data.csv` to Google Colab.
2. Run all notebook cells in order.
3. Download `academic_advisor_model.pkl` from the Colab Files panel and place it in this project folder.
4. The notebook uses a synthetic dataset for demonstration. Do not claim it was trained on real university records.

## Run the web app locally
1. Install Python 3.11+ and Git.
2. Open this folder in VS Code.
3. In the VS Code terminal:
   ```bash
   python -m venv .venv
   ```
   Windows PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python app.py
   ```
4. Open `http://127.0.0.1:5000` in your browser.

## Git and GitHub
Create an empty GitHub repository named `academic-advisor-ai`, then run in this folder:
```bash
git init
git add .
git commit -m "Build academic advisor AI prototype"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/academic-advisor-ai.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your GitHub username. If Git asks you to authenticate, complete the browser sign-in flow.

## Important notes
- Git is installed on the computer; GitHub is an online hosting service (you create an account/repository on GitHub.com; there is no separate GitHub desktop installation required for this assignment).
- If you commit the `.pkl` model, do so only after generating it and checking the file is not too large.
- This is an educational prototype, not a validated academic decision system. Predictions are based on a synthetic demonstration dataset and should not determine real student outcomes.
