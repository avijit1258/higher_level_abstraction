from flask import Flask, jsonify, send_from_directory, render_template
from flask import abort
from flask import make_response
from flask import request
from flask import url_for
from flask_cors import CORS

import xlsxwriter



from ClusteringCallGraph import *

from datetime import datetime
import json


app = Flask(__name__, static_url_path='/static')
CORS(app)

subject_system = ['treedetectronSept7.txt', 'treedetectronSept7.txt', 'treedetectronSept7.txt']

ss_count = 0
cluster_count = 0
row = 1
username = ''

workbook = ''
worksheet = ''


@app.route('/',methods=['GET'])
def root():
    # return 'Hello world'
    return render_template('home.html')


@app.route('/get_cluster', methods=['GET'])
def get_cluster():
    global ss_count

    # c = ClusteringCallGraph()
    #
    # cluster = c.python_analysis()
    # print(cluster)

    # with open('treedetectronSept7.txt', 'r') as f:
    #     print(subject_system[ss_count])
    #     content = f.read()
    #     cluster = eval(content)

    if ss_count == 1:
        print('First Subject system')
    elif ss_count == 2:
        print('Second Subject system')

    with open(subject_system[ss_count], 'r') as f:
        print(subject_system[ss_count])
        ss_count = ss_count + 1
        content = f.read()
        cluster = eval(content)

    # del c
    return jsonify(cluster)


@app.route('/save_response', methods=['POST'])
def save_response():
    global worksheet, row, cluster_count
    print('From save',worksheet, ss_count)
    # if ss_count == 1:
    #     print('First Subject system')
    # elif ss_count == 2:
    #     print('Second Subject system')
    print(request.form['key'])
    print(request.form['n_t1'])
    print(request.form['n_t2'])
    print(request.form['n_t3'])
    print(request.form['n_t4'])
    print(request.form['n_t5'])
    print(request.form['n_t6'])
    print(request.form['user_summary'])
    print(request.form['comment'])


    worksheet.write(row, 0, request.form['key'])
    worksheet.write(row, 1, request.form['n_t1'])
    worksheet.write(row, 2, request.form['n_t2'])
    worksheet.write(row, 3, request.form['n_t3'])
    worksheet.write(row, 4, request.form['n_t4'])
    worksheet.write(row, 5, request.form['n_t5'])
    worksheet.write(row, 6, request.form['n_t6'])
    worksheet.write(row, 7, request.form['user_summary'])
    worksheet.write(row, 8, request.form['comment'])

    if ss_count == 3 and cluster_count == 12:
        print('Get 3rd subject system')
        workbook.close()

    row = row + 1
    cluster_count = cluster_count + 1

    return 'Subject: '+ str(ss_count) +'Cluster: '+ str(cluster_count)


def create_csv():
    global username, workbook, worksheet
    username = input('Enter name of the user:')
    print(username+'.xlsx')
    workbook = xlsxwriter.Workbook(username+'.xlsx')
    print(workbook)
    worksheet = workbook.add_worksheet()
    print('create_csv_worksheet', worksheet)
    worksheet.write(0, 0, 'Cluster Id')
    worksheet.write(0, 1, 'tfidf_word')
    worksheet.write(0, 2, 'tfidf_method')
    worksheet.write(0, 3, 'lda_word')
    worksheet.write(0, 4, 'lda_method')
    worksheet.write(0, 5, 'lsi_word')
    worksheet.write(0, 6, 'lsi_method')
    worksheet.write(0, 7, 'user_summary')
    worksheet.write(0, 8, 'comment')




if __name__ == '__main__':
    create_csv()
    app.run()


