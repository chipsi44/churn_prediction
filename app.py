# Import required libraries
from flask import Flask, render_template, request
import pandas as pd
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Define the route for the root URL
@app.route("/", methods=["GET", "POST"])
def predict():
    # Check the request method
    if request.method == "GET":
        # If the request method is GET, return the index.html template
        print("ALIVE")
        return render_template("index.html")
    else:
        # If the request method is POST, load the pre-trained model and use it to make a prediction
        file_pckl = open('Classification_model_pipeline.pkl', 'rb')
        file_pckl_cluster = open('Clustering_model_pipeline.pkl', 'rb')
        model = pickle.load(file_pckl)
        model_cluster = pickle.load(file_pckl_cluster)
        #Cluster proportion
        dic_prop_cluster = {
            0:'9',
            1:'2',
            2 :"6.5",
            3 : "58",
            4 :"12",
            5 : "11.5"
        }
        # Get the user inputs from the form and create a dictionary of values
        dic_val = {}
        list_val = []
        for key, value in request.form.items():
            list_val.append(value)
            dic_val[key] = value
        
        # Convert the dictionary of values to a Pandas DataFrame
        value_to_predict = pd.DataFrame([dic_val])
        
        # Use the pre-trained model to make a prediction
        reponse = model.predict(value_to_predict)
        reponse_cluster = model_cluster.predict(value_to_predict)
        # Convert the prediction to a human-readable string and return it to the user
        if reponse[0] == 1: 
            reponse = 'highly'
        else: 
            reponse = "not"
        prediction_cluster = reponse_cluster[0]
        # Return the index.html template with the prediction and user inputs
        print(reponse_cluster)
        return render_template("index.html", prediction=reponse, prediction_cluster=prediction_cluster,cluster_rate = dic_prop_cluster[prediction_cluster], args=list_val)

# Start the Flask app
app.run(debug=False)
