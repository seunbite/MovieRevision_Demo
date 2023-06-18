# TODO
# 1. 'update' function - revision   2. inference speed (GPU)   3. Highest Aspect > High Aspects

from flask import Flask, render_template, request
import pandas as pd
import json
# from . import InstructMyselfm
from app_mod import main, revision_

app = Flask(__name__)

'''Main page'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/processing', methods = ['GET', 'POST'])
def processing():
    if request.method == 'POST':

        # input
        user_input = request.form['input']
        print("******* input script *******")
        print(user_input)

        rating, movie_aspects, high_aspects, scene_df = main(user_input)
        scenes = list(scene_df['scene'])

        # revision
        revisions = scene_df.sort_values(by=['total score'], ascending = False).head(3)[['idx','scene','max aspect']]
        revisions = revisions.set_index('idx').to_dict()
        revisions['idx'] = revisions['scene'].keys() # dict = {'Idx' : [22, 51, 40] , 'Scenes' : {22 : scene22, 51: scene51, 40: scene40}, 'Max Aspect' : {22: 'Profanity', 51: 'Violence', 40: 'Violence'}}
        revised = dict()
        for i in revisions['Idx']:
            #revised[i] = "RRR"
            revised[i] = revision_(revisions['Scene'][i], revisions['Max Aspect'][i], "rated as R")
        revisions['Revised Scene']=revised


    return render_template('processing.html', input = user_input, movie_rating = rating, 
                           highest_aspect = high_aspects, movie_aspects = movie_aspects,
                           N = len(scenes), scenes = scenes, revisions = revisions
                           )



@app.route('/update', methods = ['GET', 'POST'])
def update():
    if request.method == 'POST':

        # new input
        user_input = ""

        for key in request.form.keys():
            print(key)
            if key.startswith('sceneLevel-'):
                print("****")
                user_input += request.form[key] #string
            
            print(user_input)

        # script-rating, script-aspects
        movie_rating = auxbprm_(user_input)
        print("rating: ", movie_rating)
        highest_aspect, movie_aspects = aux_(user_input)
        movie_aspects = json.dumps(movie_aspects)
        print("highest is", highest_aspect, "aspects: ", movie_aspects)

        # scene-aspects
        df = aux_scenes_(user_input)
        N = len(df)
        scenes = {k+1:v for k, v in enumerate(list(df['Scene']))}
        
        # revision
        df_ = df.sort_values(by=['Scene Score'], ascending = False).head(3)[['Idx','Scene','Max Aspect']]
        revisions = df_.set_index('Idx').to_dict()
        revisions['Idx'] = revisions['Scene'].keys() # dict = {'Idx' : [22, 51, 40] , 'Scenes' : {22 : scene22, 51: scene51, 40: scene40}, 'Max Aspect' : {22: 'Profanity', 51: 'Violence', 40: 'Violence'}}
        revised = dict()
        for i in revisions['Idx']:
            #revised[i] = "RRR" # not billing
            revised[i] = revision_(revisions['Scene'][i], revisions['Max Aspect'][i], "rated as R")
        revisions['Revised Scene']=revised


    return render_template('processing.html', input = user_input, movie_rating = movie_rating, 
                           highest_aspect = highest_aspect, movie_aspects = movie_aspects,
                           N = N, scenes = scenes, revisions = revisions
                           )





if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True, port=3065)

