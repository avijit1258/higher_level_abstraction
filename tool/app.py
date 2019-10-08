from flask import Flask, jsonify, send_from_directory, render_template
from flask import abort
from flask import make_response
from flask import request
from flask import url_for
from flask_cors import CORS



from ClusteringCallGraph import *

from datetime import datetime
import json


app = Flask(__name__, static_url_path='/static')
CORS(app)

@app.route('/',methods=['GET'])
def root():
    # return 'Hello world'
    return render_template('home.html')


@app.route('/get_cluster', methods=['GET'])
def get_cluster():

    c = ClusteringCallGraph()

    cluster = c.python_analysis()
    # print(cluster)

    # with open('myfile.txt', 'r') as f:
    #     content = f.read()
    #     dic = eval(content)

    del c
    return jsonify(cluster)


if __name__ == '__main__':
    app.run()
