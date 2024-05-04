# Importation des bibliothèques nécessaires
import numpy as np
from numpy.linalg import pinv
from sklearn.linear_model import LinearRegression

class ForecastingWithDFM:
    """
    Classe pour réaliser des prévisions avec un modèle de facteurs dynamiques (DFM).
    """
    def __init__(self, num_factors, lags_y, lags_f):
        """
        Initialisation de la classe avec le nombre de facteurs, le nombre de retards pour y et F.
        
        :param num_factors: Nombre de facteurs latents à utiliser dans la régression.
        :param lags_y: Nombre de retards de la variable y à utiliser dans la régression.
        :param lags_f: Nombre de retards des facteurs à utiliser dans la régression.
        """
        self.num_factors = num_factors
        self.lags_y = lags_y
        self.lags_f = lags_f
        self.reg_f = LinearRegression()
        self.reg_ar = LinearRegression()

    def fit(self, y, factors):
        """
        Ajustement des modèles pour les prévisions.
        
        :param y: Variable cible pour les prévisions.
        :param factors: Facteurs estimés pour être utilisés dans les prévisions.
        """
        # Assurer que y est un tableau numpy unidimensionnel
        y = np.asarray(y).flatten()

        # Calculer le décalage maximal utilisé dans le modèle
        max_lag = max(self.lags_y, self.lags_f)

        # Vérifier si le nombre d'observations est suffisant pour le nombre maximal de retards
        if max_lag >= len(y):
            raise ValueError("Le nombre maximal de retards est supérieur ou égal au nombre d'observations.")

        # Préparation des données explicatives (avec retards)
        X_f = self.prepare_data(y, factors)

        # Ajustement de y pour correspondre à la taille de X_f
        # Cela suppose que `y` a été initialement plus longue en raison de l'absence de retards lors de sa création
        y_adjusted = y[-X_f.shape[0]:]
        #y_adjusted = y[max_lag:]  # Ajuster y pour correspondre aux dimensions de X_f après application des retards

        # Ajustement des modèles de régression
        self.reg_f.fit(X_f, y_adjusted)
        self.reg_ar.fit(X_f[:, :self.lags_y], y_adjusted)  # Supposer que les premières colonnes de X_f correspondent aux retards de y


    def fita(self, y, factors):
        """
        Ajustement des modèles pour les prévisions.
        
        :param y: Variable cible pour les prévisions.
        :param factors: Facteurs estimés pour être utilisés dans les prévisions.
        """

        y = np.asarray(y).flatten()
        max_lag = max(self.lags_y, self.lags_f)
        if max_lag >= len(y):
            raise ValueError("Le nombre maximal de retards est supérieur ou égal au nombre d'observations.")

        # La taille de y doit être ajustée pour correspondre à la taille des données après la création des retards
        y_adjusted = y[max_lag:]
        
        # Préparation des données pour la régression factorielle
        X_f = self.prepare_data(y, factors)
        self.reg_f.fit(X_f, y_adjusted)

        # Préparation des données pour la régression autorégressive
        X_ar = self.prepare_data(y, None)
        self.reg_ar.fit(X_ar, y_adjusted)

    def prepare_data(self, y, factors):
        # Assurez-vous que y est une matrice avec des lignes comme observations
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples = y.shape[0]

        # Initialiser la liste pour les retards de y
        lagged_ys = []
        # Créer des retards pour y
        for t in range(1, self.lags_y + 1):
            lagged_ys.append(y[:-t])  # ajuster les indices pour aligner correctement

        X = np.hstack(lagged_ys) if lagged_ys else y[:-self.lags_y]

        # Si des facteurs sont fournis, les ajouter aux retards de y
        if factors is not None:
            factors = np.asarray(factors)
            lagged_factors = []
            for t in range(1, self.lags_f + 1):
                lagged_factors.append(factors[:-t])  # ajuster les indices ici aussi

            # Assurez-vous que toutes les matrices ont le même nombre de lignes
            min_length = min(len(x) for x in [X] + lagged_factors)
            X = np.hstack([X[-min_length:]] + [f[-min_length:] for f in lagged_factors])

        return X


    def prepare_datat(self, y, factors):
        """
        Prépare les données pour la régression en incorporant les retards de y et, si fourni, les facteurs.
        
        :param y: Variable cible pour les prévisions.
        :param factors: Facteurs estimés pour être utilisés dans les prévisions (peut être None).
        :return: Matrice de conception pour la régression.
        """
        # S'assurer que y est une matrice avec des lignes comme observations et une colonne pour les valeurs
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_samples = y.shape[0]

        # Initialiser la liste pour les retards de y
        lagged_ys = []
        for t in range(1, self.lags_y + 1):
            lagged_ys.append(y[t:n_samples - self.lags_y + t])

        # Création des retards pour y en tant que matrice de conception
        X = np.hstack(lagged_ys) if lagged_ys else y[:-self.lags_y]

        # Si des facteurs sont fournis, les ajouter aux retards de y
        if factors is not None:
            factors = np.asarray(factors)
            lagged_factors = []
            for t in range(1, self.lags_f + 1):
                lagged_factors.append(factors[t:n_samples - self.lags_f + t])

            # Ajouter les retards des facteurs à la matrice de conception
            X = np.hstack([X] + lagged_factors) if lagged_factors else X

        # Retourner la matrice de conception pour la régression
        return X

    def predict(self, y, factors):
        """
        Réalise des prévisions en utilisant le modèle ajusté.
        
        :param y: Variable cible pour les prévisions.
        :param factors: Facteurs estimés pour être utilisés dans les prévisions.
        :return: Prévisions de la variable cible.
        """
        X_f = self.prepare_data(y, factors)
        print(X_f)
        prediction_f = self.reg_f.predict(X_f)
        prediction_ar = self.reg_ar.predict(X_f[:, :self.lags_y])
        return prediction_f, prediction_ar

