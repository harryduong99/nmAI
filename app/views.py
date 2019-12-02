from app import app
from flask import request, redirect, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
import sys
import os
from pprint import pprint
from imageai.Detection import ObjectDetection
from .core.classify_dog_breed import predict

app.config["IMAGE_UPLOADS"] = "/home/duongnam/Desktop/python-service/app/app/core/detect_object/input_data"
app.config["IMAGE_STATIC"] = 'app/static/img/object_detected'
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG", "GIF"]
app.config["MAX_IMAGE_FILESIZE"] = 0.5 * 1024 * 1024

@app.route("/")
def index():
    return render_template("public/index.html")

@app.route("/about")
def about():
    return """
    <h1 style='color: red;'>I'm a red H1 heading!</h1>
    <p>This is a lovely little paragraph</p>
    <code>Flask is <em>awesome</em></code>
    """

@app.route("/jinja")
def jinja():

    # Strings
    my_name = "Julian"

    # Integers
    my_age = 30

    # Lists
    langs = ["Python", "JavaScript", "Bash", "Ruby", "C", "Rust"]

    # Dictionaries
    friends = {
        "Tony": 43,
        "Cody": 28,
        "Amy": 26,
        "Clarissa": 23,
        "Wendell": 39
    }

    # Tuples
    colors = ("Red", "Blue")

    # Booleans
    cool = True

    # Classes
    class GitRemote:
        def __init__(self, name, description, domain):
            self.name = name
            self.description = description 
            self.domain = domain

        def pull(self):
            return f"Pulling repo '{self.name}'"

        def clone(self, repo):
            return f"Cloning into {repo}"

    my_remote = GitRemote(
        name="Learning Flask",
        description="Learn the Flask web framework for Python",
        domain="https://github.com/Julian-Nash/learning-flask.git"
    )

    # Functions
    def repeat(x, qty=1):
        return x * qty

    return render_template(
        "public/jinja.html", my_name=my_name, my_age=my_age, langs=langs,
        friends=friends, colors=colors, cool=cool, GitRemote=GitRemote, 
        my_remote=my_remote, repeat=repeat
    )

@app.route("/sign-up", methods=["GET", "POST"])
def sign_up():

    if request.method == "POST":
        # print(request.url)
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")

        # Alternatively

        # username = request.form["username"]
        # email = request.form["email"]
        # password = request.form["password"]

        # return redirect(request.url)
        return jsonify({'name': username,
                        'email': email,
                        'password': password
                        })
    else:
        return render_template("public/sign_up.html")

def allowed_image(filename):
    
    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
        return True
    else:
        return False


def allowed_image_filesize(filesize):

    if int(filesize) <= app.config["MAX_IMAGE_FILESIZE"]:
        return True
    else:
        return False


@app.route("/upload-image", methods=["GET", "POST"])
def upload_image():

    if request.method == "POST":
        if request.files:
            if "filesize" in request.cookies:
                if not allowed_image_filesize(request.cookies["filesize"]):
                    print("Filesize exceeded maximum limit")
                    return redirect(request.url)

                image = request.files["image"]

                if image.filename == "":
                    print("No filename")
                    return redirect(request.url)

                if allowed_image(image.filename):

                    execution_path = os.getcwd()

                    filename = secure_filename(image.filename)
                    # image_path = os.path.join(app.config["IMAGE_UPLOADS"], filename)
                    uploaded_path = os.path.join(execution_path, app.config["IMAGE_STATIC"], 'input', filename)
                    image.save(uploaded_path)

                    detector = ObjectDetection()
                    detector.setModelTypeAsRetinaNet()
                    detector.setModelPath( os.path.join(execution_path , 'app/core/detect_object/detect_model/', "resnet50_coco_best_v2.0.1.h5"))
                    detector.loadModel()

                    detections = detector.detectObjectsFromImage(input_image=uploaded_path, 
                    output_image_path=os.path.join(execution_path, app.config["IMAGE_STATIC"], 'output/' + filename ), extract_detected_objects=True)

                    accepted_dogs = filter_detect_dog(detections)
                    # return jsonify(detections)
                    dog_predictor = predict.DogBreedPrediction(accepted_dogs)
                    results = dog_predictor.predict()

                    img_public_path = os.path.join('static', 'img/object_detected')
                    data_final = {}
                    for i in range(len(accepted_dogs)):
                        data_final[os.path.join(img_public_path, accepted_dogs[i][73:])] = results[i]
                    show_uploaded_path = uploaded_path[46:]
                    return render_template("public/result_detect.html", results=data_final, show_uploaded_path=show_uploaded_path )
                else:
                    print("That file extension is not allowed")
                    return redirect(request.url)

    return render_template("public/upload_image.html")


def filter_detect_dog(detection):
    result = []
    for i, image in enumerate(detection[0]):
        if image["percentage_probability"] > 90 and image["name"] == "dog":
            result.append(detection[1][i])
    return result
                
