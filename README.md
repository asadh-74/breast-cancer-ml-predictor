# 🩺 Breast Cancer Diagnosis Predictor

A machine learning classifier that predicts whether a breast tumor is **malignant** or **benign**
from fine needle aspirate (FNA) measurements — trained, evaluated, and deployed end-to-end
(Colab → Streamlit → GitHub).

**Why this matters:** Early, accurate diagnosis directly affects treatment outcomes. This project
builds and compares three model families on a well-known clinical benchmark and ships the best one
as an interactive web app.

## 🔗 Live demo
> Deploy this repo on [Streamlit Community Cloud](https://share.streamlit.io) (free) and drop the
> link here. See **Deployment** below — it takes about 2 minutes.

## 📊 Results

Trained on the [Breast Cancer Wisconsin (Diagnostic)](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-wisconsin-diagnostic-dataset)
dataset (569 samples, 30 features), 80/20 stratified split, 5-fold cross-validation:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.983 | 0.986 | 0.986 | **0.986** | 0.995 |
| XGBoost | 0.947 | 0.946 | 0.972 | 0.959 | 0.993 |
| Random Forest | 0.947 | 0.958 | 0.958 | 0.958 | 0.994 |

⭐ Logistic Regression was selected as the deployed model — highest F1 and ROC-AUC, and it's the
most interpretable of the three (relevant when the stakeholder is a clinician, not just an engineer).

Confusion matrix (best model, test set of 114): 1 false positive, 1 false negative.

## 🛠 Tech stack
- **Modeling:** scikit-learn, XGBoost, pandas, numpy
- **Notebook:** Google Colab (`breast_cancer_classifier.ipynb`)
- **App / deployment:** Streamlit
- **Explainability:** feature importance / coefficient analysis

## 📁 Repo structure
```
.
├── breast_cancer_classifier.ipynb   # Full pipeline: EDA → training → evaluation → export
├── app.py                           # Streamlit prediction app
├── requirements.txt
└── README.md
```

## ▶️ Run it yourself

**1. Explore the full pipeline in Colab**
- Open `breast_cancer_classifier.ipynb` in [Google Colab](https://colab.research.google.com/) (or click "Open in Colab" once this repo is pushed).
- Run all cells. It downloads no external data — the dataset ships inside scikit-learn.

**2. Run the app locally**
```bash
pip install -r requirements.txt
streamlit run app.py
```
`app.py` trains the Logistic Regression model itself on startup (it's a small dataset — training
takes well under a second) rather than loading a pickled model file. This sidesteps a common
deployment headache: a `.pkl` saved with one scikit-learn version can fail to load with another
installed elsewhere. Training fresh means the app always works with whatever compatible
scikit-learn version is installed.

## 🚀 Deploy to GitHub + Streamlit Cloud (for your portfolio link)

```bash
# from this project folder
git init
git add .
git commit -m "Breast cancer diagnosis predictor: EDA, model comparison, Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/breast-cancer-ml-predictor.git
git push -u origin main
```

Then:
1. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
2. Click **New app**, pick this repo, set the main file to `app.py`, click **Deploy**.
3. Copy the live URL into the **Live demo** section above and into your resume/LinkedIn/portfolio.

## 📌 Notes for reviewers
- This is a portfolio/demo project, not a clinical tool — no medical claims are made.
- The pipeline (compare multiple models → pick by F1 → explain → export → serve) generalizes to
  any tabular classification problem; swapping in another dataset mainly changes Section 1 of the
  notebook.

---
Built by **Asad Hussain** — Electrical Engineering (NUST CEME) · [Portfolio](https://asad-portfolio-flax-pi.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/asad-hussain92/)
