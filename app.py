from flask import Flask, request, render_template, Response
import pandas as pd

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

	#get column types and column names
	#col_types = (df.dtypes).to_dict()
	column_names = [col for col in df.columns]
	return {"columns":column_names}

if __name__ == "__main__":
	app.run()




