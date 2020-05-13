import dash
import dash_core_components as dcc
import dash_html_components as html
import chart_studio.plotly as py
import plotly
import plotly.graph_objs as go
import pandas as pd
import cv2
import os
from flask import Flask, Response
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
from decimal import Decimal
from dash.dependencies import Output,Input
from random import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import count
import csv
from plotly.tools import mpl_to_plotly
from flask import current_app


server = Flask(__name__)
app = dash.Dash(__name__, server=server)
X=[]
X.append(1)
Y=[]


def video_output():
    #img=gen(VideoCamera())
    #cap=cv2.VideoCapture('http://192.168.43.1:8080/video')
    model = model_from_json(open("model_4layer_2_2_pool5.json", "r").read())
    #load weights
    model.load_weights('model_4layer_2_2_pool5.h5')
    

    result = np.array((1,3))
    once = False
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap=cv2.VideoCapture(0)
    while True:
        ret,imga=cap.read()
        img=cv2.flip(imga,1)
        if not ret:
            continue
        gray_img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        
        for (x,y,w,h) in faces_detected:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),thickness=4)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
            roi_gray=cv2.resize(roi_gray,(48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis = 0)
            img_pixels /= 255      
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            
            emotions = ('not interested','interested','neutral')
            predicted_emotion = emotions[max_index]
            
            
            row=[predicted_emotion,str(int(predictions[0][max_index] * 100))]
            update_graph_scatter(predicted_emotion)
            update_graph_scatter2(predicted_emotion)
            cv2.putText(img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
        total_sum = np.sum(result[0])
        resized_img = cv2.resize(img, (500, 400))
        ret,image11=cv2.imencode('.jpg',resized_img )
        
        #yield bytes(resized_img)
        #return Response(resized_img)
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +(image11.tobytes()) + b'\r\n\r\n')
    
@server.route('/video_output2')
def video_output2():
    return Response(video_output(),mimetype='multipart/x-mixed-replace; boundary=frame')



    #return("Predited emotion is'{}'".format(emot))


app.layout = html.Div([
    html.Div([
        html.Div([
            html.Img(src="/video_output2")
        ]),
        
        
        html.Div([
           # dcc.Graph(figure=plotly_fig)
           #dcc.Graph(id= 'matplotlib-graph', figure=plotly_fig)
           #html.Img(src="/live_graph")
           dcc.Graph(id='live-graph', animate=True),
            dcc.Interval(
                id='graph-update',
                interval=1*1000
                 ),
        ]),
        html.Div([
               #  html.Img(src="/ytv")
              # dcc.Input(id='prediction_output',value=calc),
               #html.Div(id='out')
                dcc.Graph(id='live-graph2', animate=True),
                dcc.Interval(
                id='graph-update2',
                interval=1*1000
                 ),
            ])
        
    ])
    
 

    ])        

@app.callback(Output('live-graph', 'figure'),
              [Input('graph-update', 'interval')])
def update_graph_scatter(k):
    X.append(X[-1]+1)
    Y.append(k)
    data = go.Scatter(
            x=list(X),
            y=list(Y),
            name='Scatter',
            mode= 'lines+markers'
            )
    return {'data': [data]}

@app.callback(Output('live-graph2', 'figure'),
              [Input('graph-update2', 'interval')])
def update_graph_scatter2(k):
    X.append(X[-1]+1)
    Y.append(k)
    data1 = go.Bar(
            x=list(Y),
            y=list(X),
            name='Scatter',
            
            )
    return {'data': [data1]}

if __name__ == '__main__':
    app.run_server(debug=True)
