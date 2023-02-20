from flask import Flask, request, render_template
app = Flask(__name__)

# defines a route that is accessible at the root URL ("/").
@app.route("/", methods=["GET", "POST"])
# This function is executed when a request is made to the root URL.
def predict():
    if request.method == "GET":
        # If the request method is "GET", the string "ALIVE" is printed and the "index.html" template is rendered.
        print("ALIVE")
        return render_template("index.html")
        
app.run(debug=False)