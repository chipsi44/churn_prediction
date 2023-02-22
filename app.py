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
        list_val = []
        for key,value in request.form.items() :
            list_val.append(value)
            print(value)
        return render_template("index.html", prediction = "waiting for ML", args = list_val)

app.run(debug=False)
