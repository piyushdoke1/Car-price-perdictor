from flask import Flask, request, render_template
import pandas as pd
from pickle import load

app = Flask(__name__)

# Load the trained regression model and expected input columns
model = load(open("car.pkl", "rb"))
cols = load(open("carcols.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Get input values from the form
            name = request.form.get("name")
            age = int(request.form.get("age"))
            km = int(request.form.get("km"))

            # Create input DataFrame
            input_df = pd.DataFrame([{
                "car_name": name,
                "age_years": age,
                "kms_driven": km
            }])

            # making dummies
            input_df = pd.get_dummies(input_df)

            # Ensure all model columns exist in the input
            for col in cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[cols]

            # Predict price
            predicted_price = model.predict(input_df)[0]
            predicted_price = round(predicted_price, 2)

            return render_template("home.html", price=predicted_price, brand=name)

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
