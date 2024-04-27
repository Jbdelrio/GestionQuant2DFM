import numpy as np

class PrincipalComponentsFactorModel:
    """
    Classe pour l'extraction de facteurs par la méthode des composantes principales (PC).
    """

    def __init__(self, n_factors, n_variables):
        """
        Initialiser le modèle avec un nombre spécifié de facteurs et de variables.

        :param n_factors: Nombre de facteurs latents (r).
        :param n_variables: Nombre de variables observées (N).
        """
        self.n_factors = n_factors
        self.n_variables = n_variables
        self.factor_loadings = None  # Chargements des facteurs (Lambda)
        self.factors = None  # Facteurs (F)
        self.eigenvalues = None  # Valeurs propres pour le tri des chargements des facteurs

    def fit(self, Y):
        """
        Adapter le modèle aux données observées en utilisant la méthode des composantes principales.

        :param Y: Matrice des observations (T x N).
        """
        # Centrer les données
        Y_centered = Y - np.mean(Y, axis=0)
        
        # Calculer la matrice de covariance
        covariance_matrix = np.cov(Y_centered, rowvar=False)
        
        # Effectuer la décomposition en valeurs propres
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        # Trier les vecteurs propres en fonction des valeurs propres décroissantes
        idx = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, idx]
        
        # Garder les r principaux vecteurs propres comme chargements des facteurs
        self.factor_loadings = sorted_eigenvectors[:, :self.n_factors]
        self.eigenvalues = eigenvalues[idx][:self.n_factors]
        
        # Calculer les scores des facteurs
        self.factors = Y_centered @ self.factor_loadings

    def transform(self, Y):
        """
        Transformer les données observées en scores de facteur en utilisant les chargements des facteurs estimés.

        :param Y: Matrice des observations (T x N).
        :return: Les scores des facteurs estimés (T x r).
        """
        if self.factor_loadings is None:
            raise ValueError("Le modèle doit être adapté avant la transformation.")
        Y_centered = Y - np.mean(Y, axis=0)
        return Y_centered @ self.factor_loadings

    def get_factor_loadings(self):
        """
        Obtenir les chargements des facteurs estimés.

        :return: Matrice des chargements des facteurs (N x r).
        """
        if self.factor_loadings is None:
            raise ValueError("Le modèle doit être adapté pour obtenir les chargements des facteurs.")
        return self.factor_loadings

    def get_eigenvalues(self):
        """
        Obtenir les valeurs propres utilisées pour trier les chargements des facteurs.

        :return: Liste des valeurs propres (r).
        """
        if self.eigenvalues is None:
            raise ValueError("Le modèle doit être adapté pour obtenir les valeurs propres.")
        return self.eigenvalues

    def __str__(self):
        return f"Modèle des Composantes Principales avec {self.n_factors} facteurs et {self.n_variables} variables."

