
# Importations des Bibliothèques
from dfm import DynamicFactorModel
from pcfm import PrincipalComponentsFactorModel
from kfs import KalmanFilterSmoother
from forecasting import ForecastingWithDFM
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
#### Récupération des données###################################
"""
observation_date : Date d'observation.
CPI : Indice des prix à la consommation.
INDPRO : Indice de production industrielle.
RPI : Indice des prix de détail.
UNRATE : Taux de chômage.
"""

# Chargement des données du fichier Excel
data = pd.read_excel('data/macro_datas.xlsx')

# Afficher les premières lignes des données pour comprendre leur structure
data.head(), data.columns
######Vérification de la stationnarité des données ########################

# Fonction pour tester la stationnarité d'une série temporelle avec le test Dickey-Fuller augmenté
def test_stationarity(series, column_name):
    result = adfuller(series, autolag='AIC')  # Test Dickey-Fuller
    print(f'Résultats du test de Dickey-Fuller pour {column_name}:')
    print(f'Statistique de test : {result[0]}')
    print(f'P-valeur : {result[1]}')
    print(f'Valeurs Critiques :')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    print(f'{"Stationnaire" if result[1] < 0.05 else "Non stationnaire"}\n')

# Appliquer le test de stationnarité à chaque série temporelle
for column in ['CPI', 'INDPRO', 'RPI', 'UNRATE']:
    test_stationarity(data[column], column)

# Appliquer la première différence aux séries non stationnaires et tester à nouveau leur stationnarité
data_diff = data[['CPI', 'INDPRO', 'RPI']].diff().dropna()  # Calculer la première différence et supprimer la première ligne NaN

# Tester la stationnarité des séries différenciées
for column in data_diff.columns:
    test_stationarity(data_diff[column], column)


# Ajouter la série stationnaire UNRATE déjà existante dans le dataframe original
data_diff['UNRATE'] = data['UNRATE'].iloc[1:].values#.reset_index(drop=True)

# Renommer les colonnes pour indiquer qu'elles sont les premières différences
data_diff.columns = [f"{col}_diff" for col in data_diff.columns]

# Réajuster le dataframe pour utiliser les dates d'observation comme index
data_diff.index = data['observation_date'].iloc[1:].reset_index(drop=True)

# Afficher le dataframe mis à jour
data_diff.tail(), data_diff.index
##############################################################

# Génération de données simulées pour tester le modèle
T, N, r = data_diff.shape[0], data_diff.shape[1], 3  # Nombre de périodes, variables observées, facteurs

##### DFM############
# Exemple d'initialisation et d'utilisation de la classe
dfm_example = DynamicFactorModel(n_factors=N, n_variables=r)
print(dfm_example)  # Afficher les informations du modèle

Y_simulated =np.array(data_diff) #np.random.randn(T, N)

# Adapter le modèle aux données simulées
dfm_example.fit(Y_simulated)

# Transformation des données observées en scores de facteur
factor_scores = dfm_example.transform(Y_simulated)

# Prédire les observations à partir des scores de facteur
Y_predicted = dfm_example.predict(factor_scores)

# Affichage des résultats
print("\nChargements de facteurs estimés (Lambda):")
print(dfm_example.factor_loadings)
print("\nVariances idiosyncratiques estimées (diagonal de Sigma_e):")
print(dfm_example.idiosyncratic_variances)

#####Principal Components############

pc_model = PrincipalComponentsFactorModel(n_factors=r, n_variables=N)
pc_model.fit(Y_simulated)  # Adapter le modèle
factor_scores = pc_model.transform(Y_simulated)  # Transformer les données

# Afficher les résultats
print("\nChargements des facteurs (Lambda):")
print(pc_model.get_factor_loadings())
print("\nValeurs propres:")
print(pc_model.get_eigenvalues())

#####KFS######################
# Supposons que nous avons les matrices suivantes pour notre modèle de facteurs dynamiques
# La matrice de transition d'état A doit être r x r
A = np.eye(r)

# Si vous avez 4 variables observées (pour CPI_diff, INDPRO_diff, RPI_diff, UNRATE_diff),
# la matrice d'observation C doit être 4 x r
C = np.random.rand(N, r)  # initialisée aléatoirement, ajustez selon votre modèle

# La matrice de covariance du bruit du processus Q doit être r x r
Q = np.eye(r) * 0.01

# La matrice de covariance du bruit d'observation R doit être 4 x 4
R = np.eye(N) * 0.02

# L'état initial estimé du système doit être de taille r
initial_state = np.zeros(r)

# La covariance initiale estimée doit être r x r
initial_covariance = np.eye(r) * 0.1
# Création d'observations factices pour le test
observations = np.random.rand(T, N)  

# Instanciation de la classe KalmanFilterSmoother avec les paramètres du modèle
kfs = KalmanFilterSmoother(A, C, Q, R, initial_state, initial_covariance)

# Exécuter l'algorithme EM sur les facteurs (scores de PCA)
kfs.em_algorithm(observations, iterations=10)


# Après exécution, les matrices A, C, Q et R ainsi que les états sont mis à jour
print("Matrice A après EM:", kfs.A)
print("Matrice C après EM:", kfs.C)
print("Matrice Q après EM:", kfs.Q)
print("Matrice R après EM:", kfs.R)
print("Estimations des états:", kfs.smoothed_states)
################Forecating with DFM#############################

y = np.random.randn(100, 1)  # Simuler une série temporelle
factors = np.random.randn(100, 3)  # Simuler des facteurs estimés
print(data_diff.shape)
print(factor_scores.shape)

# Pour les prévisions avec DFM, initialiser le modèle avec le nombre correct de retards et de facteurs
forecasting_model = ForecastingWithDFM(num_factors=r, lags_y=1, lags_f=1)

# Ajuster le modèle sur les données
forecasting_model.fit(data_diff.values, factor_scores)
# Faire des prévisions avec le modèle ajusté
predictions = forecasting_model.predict(data_diff.values, factor_scores)
# Afficher les prédictions
print(predictions)



"""
forecasting_model = ForecastingWithDFM(num_factors=r, lags_y=1, lags_f=1)
forecasting_model.fit(y, factors)
predictions = forecasting_model.predict(y, factors)
print(predictions)
"""