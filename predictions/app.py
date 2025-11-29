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
        api_key = data.get("api_key", "")  # Optional API key

        # Validate inputs
        if not all([latitude, longitude, start_date]):
            return jsonify({"error": "Missing required fields"}), 400

        # Path to your weather prediction script
        script_path = "energy_weather_node_past_future.py"

        # Build command with arguments
        cmd = [
            "python",
            script_path,
            "--latitude",
            str(latitude),
            "--longitude",
            str(longitude),
            "--first_date",
            start_date,
        ]

        # Add API key if provided
        if api_key:
            cmd.extend(["--api_key", api_key])

        # Run the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        if result.returncode != 0:
            return (
                jsonify(
                    {
                        "error": "Script execution failed",
                        "details": result.stderr,
                        "output": result.stdout,
                    }
                ),
                500,
            )

        # Look for HTML output files (your script might generate these)
        # Check common output filenames
        possible_outputs = [
            "weather_forecast.html",
            "output.html",
            "weather_plot.html",
            "forecast.html",
        ]

        html_content = None
        for output_file in possible_outputs:
            if os.path.exists(output_file):
                with open(output_file, "r", encoding="utf-8") as f:
                    html_content = f.read()
                break

        # If no file found, check if stdout contains HTML
        if not html_content and "<html>" in result.stdout.lower():
            html_content = result.stdout

        if html_content:
            return jsonify({"success": True, "html": html_content})
        else:
            return (
                jsonify(
                    {
                        "error": "No HTML output found",
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Script execution timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
