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

	#column_names = [col for col in df.columns]
	#return {"columns":column_names}

	#return the graph image wrapped in html
	return render_template('graph.html', url='graph.png')

if __name__ == "__main__":
	app.run()




