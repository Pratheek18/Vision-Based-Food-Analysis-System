from flask import Flask, render_template, request
import numpy as np
from skimage.transform import resize
import pickle
from PIL import Image

import recipe.recipe as r

# load the model
model = pickle.load(open('img_model_100img.p', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/new_dish')
def new_dish():
    return render_template('new_dish.html')


@app.route('/tracking')
def tracking():
    return render_template('tracking.html')


@app.route('/new_dish', methods=['POST'])
def predict():
    imagefile = request.files['imagefileID']
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)
    CATEGORIES = ['pizza', 'burger', 'samosa']
    image = Image.open(imagefile)
    image = np.array(image)
    img_resized = resize(image, (150, 150, 3))
    flat_data = []
    flat_data.append(img_resized.flatten())
    flat_data = np.array(flat_data)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    global prediction
    prediction=y_out

    return render_template('new_dish.html', prediction=y_out)


@app.route('/nutri')
def nutri():
    nutri_value=prediction
    return render_template('nutri.html',nut=nutri_value)


@app.route('/recipe', methods=['GET'])
def recipe():
    payload = {
        "recipe": r.food_type[prediction.lower()]
    }

    print(payload)

    return render_template('recipe.html', recipe = payload)


@app.route('/recipe_ingredients/<number>', methods=['GET'])
def recipe_ingredients(number):
    print(number)

    payload = {
        "ingredients": r.food_ingredients[str(number)]
    }

    return render_template('ingredients.html', payload = payload)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
