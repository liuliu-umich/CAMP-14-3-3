from flask import Flask, render_template, request
from flask import Response

from features import (
    is_positive_prediction,
    predict_all_sites,
    predict_score,
    predict_sequence_binding,
    validate_input,
    validate_sequence,
)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/predict_score", methods=["POST"])
def predict_binding_site():
    sequence = request.form.get("sequence", "").strip()
    site = request.form.get("site", "").strip()

    try:
        sequence, site = validate_input(sequence, site)
        prediction, probability, prediction_features = predict_score(sequence, site)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            sequence=sequence,
            site=site,
        )

    negative_probability = float(probability[0]) if len(probability) > 0 else 0.0
    positive_probability = float(probability[1]) if len(probability) > 1 else float(max(probability))
    predicted_positive = is_positive_prediction(prediction)

    return render_template(
        "result.html",
        sequence=sequence,
        site=site,
        residue=sequence[site - 1],
        prediction=prediction,
        predicted_positive=predicted_positive,
        positive_probability=positive_probability,
        negative_probability=negative_probability,
        probability=max(positive_probability, negative_probability),
        feature_count=prediction_features.shape[1],
    )


@app.route("/predict_all_sites", methods=["POST"])
def predict_all_binding_sites():
    sequence = request.form.get("sequence", "").strip()

    try:
        sequence = validate_sequence(sequence)
        results_df, prediction_features, filtered_sequence = predict_all_sites(sequence)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            sequence=sequence,
        )

    positive_count = int(results_df["predicted_positive"].sum())

    return render_template(
        "all_sites_result.html",
        sequence=filtered_sequence,
        sequence_length=len(filtered_sequence),
        site_count=len(results_df),
        positive_count=positive_count,
        feature_count=prediction_features.shape[1],
        rows=results_df.to_dict(orient="records"),
    )


@app.route("/predict_sequence_binding", methods=["POST"])
def predict_protein_binding():
    sequence = request.form.get("sequence", "").strip()

    try:
        sequence = validate_sequence(sequence)
        summary, results_df, prediction_features, filtered_sequence = predict_sequence_binding(sequence)
    except Exception as exc:
        return render_template(
            "index.html",
            error=str(exc),
            sequence=sequence,
        )

    return render_template(
        "sequence_result.html",
        sequence=filtered_sequence,
        sequence_length=len(filtered_sequence),
        feature_count=prediction_features.shape[1],
        predicted_positive=summary["predicted_positive"],
        threshold=summary["threshold"],
        max_probability=summary["max_probability"],
        best_site=summary["best_site"],
        best_residue=summary["best_residue"],
        best_site_prediction=summary["best_site_prediction"],
        site_count=summary["site_count"],
        rows=results_df.to_dict(orient="records"),
    )


if __name__ == "__main__":
    app.run(debug=False, use_reloader=False, threaded=False)
