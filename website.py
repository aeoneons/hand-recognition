from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename



app = Flask(__name__)
gestureslist = ["THUMBS UP", "POINTER UP", "RING UP", "FIST", "PALMS SPREAD"]
name = "Jessalyn"

@app.route("/")
def hello_world(input = "Jess"):
    global name 
    name = input
    return render_template("hello.html", person=name, gestures = gestureslist)

@app.route('/button_action', methods=['POST'])
def button_action():
    print("I GOT PRESSED")
    message = "I GOT PRESSEDs"
    
    return render_template("hello.html", person=name,  message=message, gestures = gestureslist)

@app.route('/drop_down', methods=['GET', 'POST'])
def dropdown():
    
    return render_template("hello.html", person = name, message = "NEW CLICK", gestures = gestureslist)