import cv2
import os
from flask import Flask, render_template, request, redirect, url_for, send_file
# from Models.lib_detection import load_model, detect_lp, im2single
from processing import detect_recognize_plate  

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def save_uploaded_file(file):
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)
    return file_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        # Lưu file ảnh tạm thời
        temp_image_path = save_uploaded_file(file)
        image_name = file.filename
        vehicle = "static/" + image_name

        # Xử lý ảnh và nhận diện biển số
        path_vehicle, final_string, path_drawn = detect_recognize_plate(vehicle)


        return render_template("index.html", vehicle=path_vehicle, final_string=final_string, drawn_image=path_drawn)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
