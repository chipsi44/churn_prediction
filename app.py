from flask import Flask, render_template,request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
# This function is executed when a request is made to the root URL.
def predict():
    if request.method == "GET":
        # If the request method is "GET", the string "ALIVE" is printed and the "index.html" template is rendered.
        print("ALIVE")
        return render_template("index.html")
    
    else:
        print("From Post") 
        return render_template("index.html")

app.run(debug=False)
