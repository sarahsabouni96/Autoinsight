"""Local retrieval over supported questions. Answers use fitted model statistics."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

QUESTIONS = {
    "accuracy": ["How accurate is the model?", "What are the test errors MAE RMSE and R2?", "Compare the baseline quality"],
    "drivers": ["Which features matter most?", "What drives the price?", "Explain important features and predictors"],
    "uncertainty": ["How reliable is the prediction interval?", "How uncertain are estimates?", "What is the coverage?"],
    "segments": ["What are the car clusters?", "Compare vehicle segments", "Which groups of cars are in the data?"],
    "training": ["How many cars were used for training?", "What data did you use?", "How was the dataset split?"],
}
labels = [key for key, phrases in QUESTIONS.items() for phrase in phrases]
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
vectors = vectorizer.fit_transform([q for phrases in QUESTIONS.values() for q in phrases])


def answer(question, report):
    similarities = (vectors @ vectorizer.transform([question]).T).toarray().ravel()
    if similarities.max() < .16:
        return "Try a question about accuracy, feature importance, prediction intervals, clusters, or training data."
    intent = labels[int(np.argmax(similarities))]
    if intent == "accuracy":
        return (f"{report['model']} has test MAE {report['test']['MAE']:,.0f} price units. "
                f"Always predicting the training median gives MAE {report['baseline']['MAE']:,.0f}.")
    if intent == "drivers":
        top = sorted(report["importance"], key=report["importance"].get, reverse=True)[:2]
        return "Largest increases in MAE after shuffling a feature: " + ", ".join(
            f"{name} ({report['importance'][name]:,.0f} price units)" for name in top) + ". This measures predictive reliance."
    if intent == "uncertainty":
        return (f"The 90% interval uses a calibration radius of {report['interval_radius']:,.0f} price units. "
                f"It covered {report['test_coverage']:.1%} of held-out cars. Coverage can vary by car type.")
    if intent == "segments":
        return "Training medians by segment: " + "; ".join(
            f"{label}: year {values['Year of manufacture']:.0f}, mileage {values['Mileage']:,.0f}, price {values['Price']:,.0f}"
            for label, values in report["segment_medians"].items()) + "."
    rows = report["rows"]
    return (f"Removed {report['duplicates_removed']} duplicate records. Used {rows['fit']:,} for fitting "
            f"and model selection, {rows['calibration']:,} for interval calibration, and {rows['test']:,} for testing.")
