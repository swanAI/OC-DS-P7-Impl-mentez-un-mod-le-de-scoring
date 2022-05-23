# 1. Library imports
import pandas as pd 
import uvicorn
from fastapi import FastAPI
import joblib


# Creer l'objet app
app = FastAPI(title="App pret à dépenser",
    description="""scoring crédit pour calculer la probabilité qu’un client rembourse son crédit"""
    )


@app.get("/")
async def root():
    return {"message": "Hello World"}



#importer dataframe des données clients tests

df_test_prod = pd.read_csv('df_test_ok_prod_100_V7.csv', index_col=[0])
# supprimer target
df_test_prod.drop(columns=['TARGET'], inplace=True)
# mettre SK_ID_CURR en index 
df_test_prod_request  = df_test_prod.set_index('SK_ID_CURR')
# Création list des clients 
clients_id = df_test_prod["SK_ID_CURR"].tolist() 






# fonction predict
@app.get('/predict/{id}')
async def fonction_predict_LGBM(id: int):

    if id not in clients_id:
        raise HTTPException(status_code=404, detail="client's id not found")
    
    else:
        
        
        pipe_prod = joblib.load('LGBM_pipe_version7.pkl')
    
        values_id_client = df_test_prod_request.loc[[id]]
       
        # Définir le best threshold
        prob_preds = pipe_prod.predict_proba(values_id_client)
        
        #Fast_API_prob_preds
        threshold = 0.332# definir threshold ici
        y_test_prob = [1 if prob_preds[i][1]> threshold else 0 for i in range(len(prob_preds))]
        
       
        return {
            "prediction": y_test_prob[0],
            "probability_0" : prob_preds[0][0],
            "probability_1" : prob_preds[0][1],}
    

# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
#if __name__ == '__main__':
    #uvicorn.run("hello_world_fastapi:app")
    #uvicorn.run(app, host='127.0.0.1', port=8000)









