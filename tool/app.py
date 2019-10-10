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

subject_system = ['treedetectronSept7.txt', 'treedetectronSept7.txt', 'treedetectronSept7.txt']

ss_count = 0

@app.route('/',methods=['GET'])
def root():
    # return 'Hello world'
    return render_template('home.html')


@app.route('/get_cluster', methods=['GET'])
def get_cluster():
    # c = ClusteringCallGraph()
    #
    # cluster = c.python_analysis()
    # print(cluster)

    with open('treedetectronSept7.txt', 'r') as f:
        print(subject_system[ss_count])
        content = f.read()
        cluster = eval(content)

    # if ss_count == 1:
    #     print('First Subject system')
    # elif ss_count == 2:
    #     print('Second Subject system')
    #
    # with open(subject_system[ss_count], 'r') as f:
    #     print(subject_system[ss_count])
    #     ss_count = ss_count + 1
    #     content = f.read()
    #     cluster = eval(content)

    # del c
    return jsonify(cluster)


@app.route('/save_response', methods=['POST'])
def save_response():

    # if ss_count == 1:
    #     print('First Subject system')
    # elif ss_count == 2:
    #     print('Second Subject system')

    print(request.form['n_t1'])
    print(request.form['n_t2'])
    print(request.form['n_t3'])
    print(request.form['n_t4'])
    print(request.form['n_t5'])
    print(request.form['n_t6'])
    return 'Successfully saved'


if __name__ == '__main__':
    app.run()
