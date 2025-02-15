from flask import Flask, request, render_template

from src.logger import logging
from src.pipeline.pipeline_prediction import CustomData, PredictionPipeline

application = Flask(__name__)
app = application


@app.route("/home")
def home():
    return render_template("home.html")


@app.route("/eda")
def eda():
    return render_template("EDA.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        logging.info("Getting user input from web form")
        data = CustomData(
            longitude=request.form.get("longitude"),
            latitude=request.form.get("latitude"),
            housing_median_age=request.form.get("housing_median_age"),
            total_rooms=request.form.get("total_rooms"),
            total_bedrooms=request.form.get("total_bedrooms"),
            population=request.form.get("population"),
            households=request.form.get("households"),
            median_income=request.form.get("median_income"),
            ocean_proximity=request.form.get("ocean_proximity"),
        )
        x_pred = data.get_DataFrame()

        logging.info("Calculating prediction for user input")
        ppln_pred = PredictionPipeline()
        y_pred = ppln_pred.predict(features=x_pred)
        logging.info("Returning prediction to user")

        return render_template("home.html", results=y_pred[0].round(2))


if __name__ == "__main__":
    app.run(host="0.0.0.0")
