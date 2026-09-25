from flask import Flask, render_template, request, jsonify, send_file
import os

from src.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")

def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    image = request.files.get("image")

    if not image:
        return jsonify({
            "success": False,
            "message": "No image uploaded"
        })

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "current_image.jpg"
    )

    image.save(image_path)

    return jsonify({
        "success": True
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "current_image.jpg"
    )

    if not os.path.exists(image_path):
        return jsonify({
            "success": False,
            "message": "No image found"
        })

    predict_class, confidence = predict_image(
        image_path
    )

    return jsonify({
        "success": True,
        "class": predict_class,
        "confidence": round(confidence, 2)
    })

# SHOW UPLOADED IMAGE
@app.route("/uploaded-image")
def uploaded_image():
    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        "current_image.jpg"
    )

    return send_file(image_path)

if __name__ == "__main__":
    app.run(debug=True)