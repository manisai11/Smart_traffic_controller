
from flask import Flask, render_template, request, jsonify
import os
from process_video import process_video  # Vehicle counting function
from app1 import process_video_amb      # Ambulance detection function
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads/"

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

executor = ThreadPoolExecutor(max_workers=4)  # For parallel processing of each lane

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():
    results = {}
    ambulance_lane = None
    lane_vehicle_counts = {}

    for lane in range(1, 5):  # Loop through four lanes
        file_key = f"file{lane}"
        if file_key not in request.files:
            return jsonify({"error": f"No file uploaded for lane {lane}"}), 400

        file = request.files[file_key]
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"lane{lane}_{file.filename}")
        file.save(file_path)

        # Run vehicle counting and ambulance detection in parallel
        future_vehicle_count = executor.submit(process_video, file_path)
        future_ambulance_detected = executor.submit(process_video_amb, file_path)

        vehicle_count = future_vehicle_count.result()
        ambulance_detected = future_ambulance_detected.result()

        # Store results per lane
        results[f"lane{lane}"] = {
            "vehicle_count": vehicle_count,
            "ambulance_detected": ambulance_detected
        }
        lane_vehicle_counts[f"lane{lane}"] = vehicle_count

        # Check if ambulance is detected in this lane
        if ambulance_detected:
            ambulance_lane = f"lane{lane}"

    # Determine green signal allocation
    if ambulance_lane:
        green_signal_lane = ambulance_lane  # Prioritize the ambulance lane
    else:
        # Select the lane with the maximum vehicle count
        green_signal_lane = max(lane_vehicle_counts, key=lane_vehicle_counts.get)

    # Update results to reflect signal allocation
    for lane, data in results.items():
        if lane == green_signal_lane:
            data["signal"] = "green"
        else:
            data["signal"] = "red"

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
