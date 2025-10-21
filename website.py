from flask import Flask, render_template, request, redirect, url_for




app = Flask(__name__)

@app.route("/")
def hello_world(name = "Jessalyn"):
    return render_template("hello.html", person=name)

@app.route('/button_action', methods=['POST'])
def button_action():
    print("I GOT PRESSED")
    message = "I GOT PRESSED"

    return render_template("hello.html", message=message)