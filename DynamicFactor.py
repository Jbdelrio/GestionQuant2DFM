import numpy as np

class DynamicFactorModel:
    """
    Classe pour implémenter un modèle de facteur dynamique (Dynamic Factor Model - DFM).
    """

    def __init__(self, n_factors, n_variables):
        """
        Initialiser le modèle avec un nombre spécifié de facteurs et de variables.

        :param n_factors: le nombre de facteurs latents (r).
        :param n_variables: le nombre de variables observées (N).
        """
        self.n_factors = n_factors  # Nombre de facteurs latents
        self.n_variables = n_variables  # Nombre de variables observées
        self.factor_loadings = np.random.randn(n_variables, n_factors)  # Matrice de chargement des facteurs (Lambda)
        self.idiosyncratic_variances = np.random.randn(n_variables)  # Variances idiosyncratiques des erreurs (diagonal de Sigma_e)

    def fit(self, Y):
        """
        Adapter le modèle aux données observées. Cette méthode doit être développée pour inclure
        des algorithmes spécifiques pour estimer les chargements des facteurs et les variances.

        :param Y: matrice des observations (T x N).
        """
        T = Y.shape[0]

        # Initialiser la matrice des facteurs (F) avec des valeurs aléatoires
        F = np.random.randn(T, self.n_factors)

        # Estimer les chargements des facteurs (Lambda) par la méthode des moindres carrés.
        self.factor_loadings = np.linalg.pinv(F) @ Y  # Pseudo-inverse de F multiplié par Y

        # Estimer les variances idiosyncratiques (la diagonale de Sigma_e).
        residuals = Y - F @ self.factor_loadings.T
        self.idiosyncratic_variances = np.var(residuals, axis=0)

    def transform(self, Y):
        """
        Transformer les données observées en scores de facteur. Cette méthode nécessiterait
        une estimation des facteurs donnée la matrice de chargement des facteurs.

        :param Y: matrice des observations (T x N).
        :return: les scores des facteurs estimés (T x r).
        """
        return(np.linalg.pinv(self.factor_loadings) @ Y.T).T#np.linalg.pinv(self.factor_loadings) @ Y.T  # Pseudo-inverse de Lambda et multiplication par Y

    def predict(self, F_new):
        """
        Prédire les observations en utilisant de nouveaux scores de facteur.

        :param F_new: nouveaux scores des facteurs (T x r).
        :return: les observations prédites (T x N).
        """
        if self.factor_loadings is None:
            raise ValueError("Le modèle doit être adapté avant de faire des prédictions.")
        # Y = Lambda * F + erreur, ici nous ignorons l'erreur pour la prédiction
        return F_new @ self.factor_loadings.T

    def __str__(self):
        return f"Modèle de Facteur Dynamique avec {self.n_factors} facteurs et {self.n_variables} variables."


