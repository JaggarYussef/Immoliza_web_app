import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor # NEW
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression # OLD
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import os 

def train():
    
    
    
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    
    # Obtenez le répertoire du script Python en cours d'exécution
    repertoire_script = os.path.dirname(os.path.abspath(__file__))

    
    # Spécifiez le nom du fichier que vous voulez ouvrir
    nom_fichier = 'data_input.csv'  # Remplacez 'input.csv' par le nom de votre fichier
       
    # Chemin complet vers le fichier
    chemin_fichier = os.path.join(repertoire_script, nom_fichier)

    # Imprimer le chemin complet utilisé pour ouvrir le fichier
    print(f"Chemin complet utilisé pour ouvrir le fichier : {chemin_fichier}")

    # Charger le fichier CSV avec le délimiteur correct
    data  = pd.read_csv(chemin_fichier, delimiter=';', encoding='ISO-8859-1', index_col=0)
       
    

        
    # Imprimer les noms de colonnes du DataFrame
    print("Noms de colonnes du DataFrame:")
    print(data.columns.tolist())  # Convertir les noms de colonnes en une liste et les imprimer
    
    print(data.index)
    print("ok................................................")
    
    # Define features to use
    num_features = [
        'total_area_sqm',
        'nbr_bedrooms',
        'zip_code', 
        'latitude', 
        'longitude', 
        'primary_energy_consumption_sqm',
        'surface_land_sqm',
        'density',
        'construction_year',
        'garden_sqm', 'terrace_sqm',
        #'prix_m2',
        'prix_m2_y',
        'cadastral_income',
        'nbr_frontages',
        
        ]
    fl_features = [
        'fl_garden', 
        'fl_terrace',
        'fl_swimming_pool',   # suppr change rien
        'fl_floodzone',       # suppr aumente le train et diminue le test 
        
        'fl_furnished',      # suppr aumente le test et diminue le train
        'fl_open_fire',
        #'fl_double_glazing'
            
        
        ]
    cat_features = [
        'property_type', 
        'subproperty_type',
        'region', 
        'province', 
        'locality',
        'equipped_kitchen',
        'state_building',
        
        #'heating_type', 
        ##'epc'
        ]
    

    

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model using HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
    train()