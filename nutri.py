import requests
import pandas as pd
import csv


def get_nutrition(food_name):
    nutrition_data = pd.DataFrame(columns=[
        'name', 'protein', 'calcium', 'fat', 'carbohydrates', 'vitamins'
    ])
    for name in food_name:
        url = "https://api.nal.usda.gov/fdc/v1/foods/search?api_key=d4D6dSOc81pTAOY2gsNZ0YhjkMlhStLJRoII5SJu&query=" + name
        response = requests.get(url)
        data = response.json()
        flatten_json = pd.json_normalize(data["foods"])
        first_food = flatten_json.iloc[0]
        first_food_nutrition_list = first_food.foodNutrients
        for item in first_food_nutrition_list:
            if item['nutrientNumber'] == "208":
                calorie = item['value']
                continue
            if item['nutrientNumber'] == "203":
                protein = item['value']
                continue
            if item['nutrientNumber'] == "301":
                calcium = item['value']
                continue
            if item['nutrientNumber'] == "204":
                fat = item['value']
                continue
            if item['nutrientNumber'] == "205":
                carbs = item['value']
                continue
            if item['nutrientNumber'] == "318":
                vitamin_a = item['value']
                continue
            if item['nutrientNumber'] == "401":
                vitamin_c = item['value']
                continue

        vitamins = float(vitamin_a) + float(vitamin_c)
        print(name)
        nutrition_data = nutrition_data.append(
            {
                'name': name,
                'calories': calorie,
                'protein': protein,
                'calcium': calcium / 1000,
                'fat': fat,
                'carbohydrates': carbs,
                'vitamins': vitamins / 1000
            },
            ignore_index=True)

    return nutrition_data


nutri = get_nutrition([
    'pizza', 'burger', 'samosa', 'icecream', 'soup', 'vadapav', 'cupcakes',
    'biryani', 'dosa', 'idli', 'paratha'
])

print(nutri)
#nutrition101 = nutri.reset_index(drop=True)

nutri.to_json('static/nutri1.json')
'''
nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('static/nutri.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]
print(nutrition_table);
'''