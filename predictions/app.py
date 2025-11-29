from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import subprocess
import os
import json
import sys

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure Python 3.6 compatibility
if sys.version_info < (3, 7):
    print("Running on Python 3.6 - some features may be limited")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.json
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        start_date = data.get("start_date")

        # Validate inputs
        if not all([latitude, longitude, start_date]):
            return jsonify({"error": "Missing required fields"}), 400

        # Path to your weather prediction script
        script_path = "/home/zef/html/predictions/energy_weather_node_past_future.py.py"  # Update this path

        # Run your Python script with the provided parameters
        # Adjust the command based on how your script accepts arguments
        cmd = [
            "python",
            script_path,
            "--lat",
            str(latitude),
            "--lon",
            str(longitude),
            "--date",
            start_date,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            return (
                jsonify({"error": "Script execution failed", "details": result.stderr}),
                500,
            )

        # Assuming your script outputs HTML graph
        # Read the generated HTML file
        output_file = "/home/zef/html/predictions/output.html"  # Update based on your script's output
        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                html_content = f.read()

            return jsonify({"success": True, "html": html_content})
        else:
            return jsonify({"error": "Output file not found"}), 500

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Script execution timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
