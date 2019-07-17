from flask import Flask, jsonify, send_from_directory, render_template
from flask import abort
from flask import make_response
from flask import request
from flask import url_for

from ClusteringCallGraph import *

from datetime import datetime
import json


app = Flask(__name__, static_url_path='/static')


@app.route('/',methods=['GET'])
def root():
    # return 'Hello world'
    return render_template('home.html')


@app.route('/api/v1/get_cluster', methods=['GET'])
def get_cluster():

    c = ClusteringCallGraph()

    cluster = c.python_analysis()
    del c
    return cluster


if __name__ == '__main__':
    app.run()
