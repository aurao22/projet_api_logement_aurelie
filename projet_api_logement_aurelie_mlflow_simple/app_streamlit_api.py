import streamlit as st
import pandas as pd
from PIL import Image
import sys
sys.path.append("C:\\Users\\User\\WORK\\workspace-ia\\PROJETS\\")
from projet_api_logement.api_logement_commons import get_img_path
import mlflow
    # Predict on a Pandas DataFrame.
import pandas as pd
import requests

st.set_page_config(
     layout="wide",
     page_title="Californie",
     page_icon = "🌴"
 )

labels = {
        'MedInc':['Revenu médian dans le secteur', '(en 10K $ dollars)'],
        'HouseAge':['Age médian des logements dans le secteur', 'ans'],
        'AveRooms':['Nombre moyen de pièces','pièces'],
        'AveBedrms':['Nombre moyen de chambres', 'chambres'],
        'Population':['Population dans le secteur','habitants'],
        'AveOccup':['Occupation moyenne des logements', 'habitants / logement'],
        'Latitude':['Latitude', '° degré'],
        'Longitude':['Longitude', '° degré'],
        }

default_values = {
        'MedInc':[9.87],
        'HouseAge':[150],
        'AveRooms':[7],
        'AveBedrms':[3],
        'Population':[1425],
        'AveOccup':[3],
        'Latitude':[35],
        'Longitude':[-119],
        }

st.title('Estimation du prix des logements en Californie')
col1, col2 = st.columns([1, 1])

img_path = r'C:\Users\User\WORK\workspace-ia\PROJETS\projet_api_logement\static\illustration-california-color2.png'
image = Image.open(img_path)
col1.image(image, caption='California')

col2.subheader("Merci de renseigner les valeurs pour la prédiction")

for key, las in labels.items():
    step = 1.0
    if isinstance(default_values.get(key, [0])[0], int):
        step = 1
    # création dynamique des variables qui contiendront la valeur du champs
    vars()[key] = col2.number_input(las[0],value=default_values.get(key, [0])[0], step=step, format=None, key=key, help=las[1], on_change=None)

if col2.button("Prédire le prix moyen", key="button_submit", help='Cliquez sur le bouton pour estimer le prix du logement.'):
  
        # "MedInc	HouseAge	AveRooms	AveBedrms	Population	AveOccup	Latitude	Longitude"
        input_datas = {
            "MedInc":[vars()["MedInc"]],
            "HouseAge":[vars()["HouseAge"]],
            "AveRooms":[vars()["AveRooms"]],
            "AveBedrms":[vars()["AveBedrms"]],
            "Population":[vars()["Population"]],
            "AveOccup":[vars()["AveOccup"]],
            "Latitude":[vars()["Latitude"]],
            "Longitude":[vars()["Longitude"]]
        }
        X_te = pd.DataFrame(input_datas)
        
        headers = {'Content-Type': 'application/json'}
        url = 'http://127.0.0.1:5000/invocations'
        data = X_te.to_json(orient='split')
        response = requests.post(url,headers=headers, data=data)
        responseJson = response.json()
        pred = round(responseJson[0])
        col1.title(f'Le prix du logement est estimé à {round(pred,2)} (10 k$ Dollars)' )

