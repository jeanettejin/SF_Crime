import requests
import pandas as pd

# URL
url = 'http://localhost:5000/api'

test = pd.read_csv('Data/test.csv')

# Change the value of experience that you want to test
values = test[0:2].to_json()

r = requests.post(url, json=values)
print(r.json())
