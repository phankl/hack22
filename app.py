from flask import Flask, request, render_template, Response
import pandas as pd
import graph

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

#direction for displaying causal graph based on user inputted csv
@app.route('/causal_graph', methods=['POST'])
def causal_graph():
	#extract file path entry and read contents into dataframe
	file = request.files['file']
	df = pd.read_csv(file)

	## pass df into graph.py

	##need to access b in order to get the appropriate image
	
	#return the graph image wrapped in html
	graph_name = 'graph.png'
	return render_template('graph.html', url=graph_name)
	#return "test"

if __name__ == "__main__":
	app.run()




