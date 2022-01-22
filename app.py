from flask import Flask, request, render_template, Response

import sqlite3
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

#direction for uploading CSV file
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
	#extract file path entry and read contents into dataframe
	file = request.files['file']
	df = pd.read_csv(file)

	#get column types and column names
	col_types = (df.dtypes).to_dict()
	#check for unnamed columns?
	column_names = [col for col in df.columns]

	#connect to database
	conn = sqlite3.connect('CSV_DATA.db')
	c = conn.cursor()

	#right now, only supports one table of data
	table_vals = ''
	for col in column_names:
		table_vals = table_vals+" "+col
		if col_types[col] == 'float64':
			table_vals += " FLOAT,"
		elif col_types[col] == 'int64':
			table_vals += " INT,"
		elif col_types[col] == 'object':
			table_vals += " VARCHAR(255),"
	table_vals = table_vals[1:len(table_vals)-1]

	#create table
	c.execute('CREATE TABLE if not exists T1 ('+table_vals+')')

	#insert csv data into the table
	for i, j in df.iterrows():
		row_data = '('
		for col in column_names:
			row_data = row_data + "'" + str(j[col])+"',"
		row_data = row_data[:len(row_data)-1]
		if row_data != '':
			c.execute("INSERT INTO T1 VALUES "+row_data+")")
			conn.commit()
	return "csv data in db!"

#direction to display the CSV data
@app.route('/display_csv', methods=['POST', 'GET'])
def display_csv():
	#connect
	cn = sqlite3.connect('CSV_DATA.db')
	df = pd.read_sql_query("SELECT * FROM T1", cn)
	return render_template("display_csv.html", data=df.to_html())

if __name__ == "__main__":
	app.run()




