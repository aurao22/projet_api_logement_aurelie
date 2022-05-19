# projet_api_logement_aurelie WITH ML FLOW

`mlflow --help`

# Version simple

## 1. En python
### 1.1. Créer le model en python, et le sauvegarder dans ML Flow 

Créer la signature :

```python
from mlflow.models.signature import infer_signature
signature = infer_signature(X_train, y_train)

```

### 1.2. Ajout du modèle dans ML Flow
```python
import mlflow.sklearn
nom_model = 'mlfow_model'
mlflow.sklearn.save_model(pipeline, nom_model, signature=signature)
print(f"mlflow models serve -m {nom_model}/") # avec conda
print(f"mlflow models serve -m {nom_model}/ --env-manager=local") # Sans conda
```

## 2. Lancer le serveur ML Flow

Se positionner dans le répertoire du projet

```python
# avec conda
print(f"mlflow models serve -m {nom_model}/") 
# Sans conda
print(f"mlflow models serve -m {nom_model}/ --env-manager=local") 
```

## 3. Lancer le serveur avec l'UI

Se positionner dans le répertoire du projet

```python
print('streamlit run myfile.py')
```

# ML Flow avec suivi des environnements et version de modèles

```python
TODO
```
