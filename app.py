from flask import Flask, request, render_template, Response
import pandas as pd
import graph
#import io
#from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#from matplotlib.figure import Figure

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

#direction for extracting CSV file
@app.route('/csv_data', methods=['POST'])
def csv_data():
	#extract file path entry and read contents into dataframe
	file = request.files['file']
	df = pd.read_csv(file)

	column_names = [col for col in df.columns]
	return {"columns":column_names}

#path to image of causal graph
@app.route('/graphs')
def graph_image():
	return render_template('graph_image.html', url='graph.png')

if __name__ == "__main__":
	app.run()




