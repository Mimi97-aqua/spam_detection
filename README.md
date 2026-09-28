## Spam Detection ML Model

A simple SMS spam classifier built with scikit-learn. It trains and compares a Multinomial Naive Bayes model and a Logistic Regression model, using bag-of-words features, to label messages as **spam** or **ham** (not spam).

### Dataset

`spam.xlsx` holds 5,572 labelled SMS messages:

- `v1`: the label (`spam` or `ham`)
- `v2`: the message text

### Workflow

All the work is in [`model.ipynb`](model.ipynb):

1. **Load data**: read `spam.xlsx` with pandas.
2. **Clean data**: drop the mostly empty `Unnamed` columns, remove 409 duplicate rows (5,163 remain), rename the columns to `label` / `message`, and encode the labels (`spam` → 1, `ham` → 0).
3. **Prepare features**: split 80/20 into train and test sets (`random_state=42`), then vectorize the text with `CountVectorizer` (English stop words removed).
4. **Train models**: `MultinomialNB` and `LogisticRegression(max_iter=1000)`.
5. **Evaluate**: accuracy, confusion matrix, and a classification report for each model.

### Results (Naive Bayes, test set)

| Metric          | Ham (0) | Spam (1) |
|-----------------|---------|----------|
| Precision       | 0.98    | 0.91     |
| Recall          | 0.99    | 0.88     |
| F1-score        | 0.99    | 0.89     |

Overall accuracy: **~97.7%** on 1,033 test messages.

### Setup

Requires Python 3.14+. The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run --with jupyter jupyter lab model.ipynb
```

### Dependencies
- pandas
- scikit-learn
