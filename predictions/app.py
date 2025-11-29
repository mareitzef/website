from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import subprocess
import os
import json
import sys

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # Enable CORS for all routes

# Ensure Python 3.6 compatibility
if sys.version_info < (3, 7):
    print("Running on Python 3.6 - some features may be limited")


@app.route("/")
def index():
    return open("index.html", "r", encoding="utf-8").read()


@app.route("/plots")
def get_plots():
    """Serve the static HTML plots file"""
    try:
        plots_file = "Meteostat_and_openweathermap_plots_only.html"
        if os.path.exists(plots_file):
            with open(plots_file, "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content
        else:
            return jsonify({"error": "Plots file not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.json
        latitude = data.get("latitude", "47.9161926")
        longitude = data.get("longitude", "7.70911552")
        start_date = data.get("start_date", "")

        # Path to your weather prediction script
        script_path = os.path.join(
            os.path.dirname(__file__), "energy_weather_node_past_future.py"
        )
        work_dir = os.path.dirname(__file__)

        # Build command with arguments
        cmd = ["python", script_path]

        if start_date:
            cmd.extend(["--start_date", start_date])

        if latitude and longitude:
            cmd.extend(["--latitude", str(latitude), "--longitude", str(longitude)])

        # Run the script in the correct directory
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd=work_dir,
        )

        if result.returncode != 0:
            return (
                jsonify(
                    {
                        "error": "Script execution failed",
                        "details": result.stderr,
                        "output": result.stdout,
                        "command": " ".join(cmd),
                    }
                ),
                500,
            )

        # Look for the plots-only HTML file in the working directory
        plots_filename = os.path.join(
            work_dir, "Meteostat_and_openweathermap_plots_only.html"
        )

        if os.path.exists(plots_filename):
            with open(plots_filename, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Return the plots directly (already just divs, no body tags)
            return jsonify(
                {"success": True, "html": html_content, "filename": plots_filename}
            )
        else:
            # List files to help debug
            files = [f for f in os.listdir(work_dir) if f.endswith(".html")]
            return (
                jsonify(
                    {
                        "error": "Output file not found",
                        "expected": plots_filename,
                        "available_files": files,
                        "work_dir": work_dir,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                ),
                500,
            )

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Script execution timeout"}), 500
    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
