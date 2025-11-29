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

        # Look for the plots-only HTML file
        plots_filename = "Meteostat_and_openweathermap_plots_only.html"

        if os.path.exists(plots_filename):
            with open(plots_filename, "r", encoding="utf-8") as f:
                html_content = f.read()

            # Extract just the plot divs from the generated HTML
            # We'll embed these into our page instead of the full HTML
            import re

            # Try to extract the body content or the plots
            # This depends on your template structure
            body_match = re.search(
                r"<body[^>]*>(.*?)</body>", html_content, re.DOTALL | re.IGNORECASE
            )
            if body_match:
                plot_content = body_match.group(1)
            else:
                # If no body tag, use the whole content
                plot_content = html_content

            return jsonify(
                {"success": True, "html": plot_content, "filename": plots_filename}
            )
        else:
            # List files to help debug
            files = [f for f in os.listdir(".") if f.endswith(".html")]
            return (
                jsonify(
                    {
                        "error": "Output file not found",
                        "expected": plots_filename,
                        "available_files": files,
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
