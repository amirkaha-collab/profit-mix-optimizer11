# Streamlit – מנוע הקצאה (שילובים של 1–3 מסלולים)

## מה יש פה
- `app.py` אפליקציית Streamlit
- `template_data.xlsx` תבנית לדוגמה (אפשר להחליף בקובץ שלך)
- `requirements.txt`
- `.streamlit/config.toml` (דארק מוד)

## עמודות חובה בקובץ המסלולים
חייבות להיות בדיוק העמודות הבאות (באנגלית, באותיות קטנות):
- `provider` – שם הגוף
- `fund` – שם המסלול
- `stocks` – אחוז מניות
- `foreign` – אחוז חו״ל
- `fx` – אחוז חשיפת מט״ח
- `illiquid` – אחוז לא-סחיר
- `sharpe` – שארפ (מספר)
- `service` – ציון שירות (מספר)

> ישראל מחושב באפליקציה: `100 - foreign`

## הרצה מקומית
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## העלאה ל-Streamlit Cloud
1. העלה את התיקייה ל-GitHub (כולל `template_data.xlsx`).
2. Streamlit Cloud → New app → בחר Repo → `app.py`.
3. Deploy.

## תכונות מרכזיות
- בחירת שילובים: 1 / 2 / 3 מסלולים (אפשר לבחור כמה אפשרויות יחד).
- אופציה: "רק חלופות מאותו גוף מנהל".
- 3 חלופות עם סט מנהלים שונה (אין שתי חלופות עם אותו סט).
- דארק מוד + RTL.
- שירות: אפשר להעלות CSV `provider,score` כדי לדרוס ציוני שירות.
