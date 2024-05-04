import numpy as np
from KFS import KalmanFilterSmoother
from PrincipalComponents import PrincipalComponentsFactorModel

class DynamicFactorModel:
    """
    Implémentation d'un modèle de facteur dynamique qui supporte différentes méthodes d'extraction de facteurs.
    
    Attributes:
        n_factors (int): Nombre de facteurs latents (r).
        n_variables (int): Nombre de variables observées (N).
        extraction_method (str): Méthode utilisée pour l'extraction des facteurs ('PC' pour PCA, 'KFS' pour Kalman Filter Smoother).
        params (list): Paramètres nécessaires pour l'initialisation des modèles d'extraction.
    """
    def __init__(self, n_factors, n_variables,extraction_method="PC", params=None,iterations=10):

        self.n_factors = n_factors  # Nombre de facteurs latents
        self.n_variables = n_variables  # Nombre de variables observées
        self.factor_loadings = np.zeros((n_variables, n_factors))  # Matrice de chargement des facteurs (Lambda)
        self.idiosyncratic_variances = np.ones(n_variables)  # Variances idiosyncratiques des erreurs (diagonal de Sigma_e)
        self.params = params
        self.extraction_method=extraction_method
        self.iterations=iterations

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
        # Choix du modèle d'extraction
        if self.extraction_method == "PC":
            self.factor_model = PrincipalComponentsFactorModel(self.n_factors,self.n_variables)
            self.factor_model.fit(Y)
            print("\nChargements des facteurs (Lambda):")
            print(self.factor_model.get_factor_loadings())
            print("\nValeurs propres:")
            print(self.factor_model.get_eigenvalues())
        elif self.extraction_method == "KFS":
            if self.params is None:
                raise ValueError("Parameters for Kalman Filter must be provided.")
            self.factor_model = KalmanFilterSmoother(self.params[0],self.params[1],self.params[2],self.params[3],self.params[4],self.params[5],self.iterations)
            # Exécuter l'algorithme EM sur les facteurs (scores de PCA)
            self.factor_model.em_algorithm(Y)
            # Après exécution, les matrices A, C, Q et R ainsi que les états sont mis à jour
            print("Matrice A après EM:", self.factor_model.A)
            print("Matrice C après EM:", self.factor_model.C)
            print("Matrice Q après EM:", self.factor_model.Q)
            print("Matrice R après EM:", self.factor_model.R)
            #Le nombre d'états lissés trop importants pour être affichés 
            #print("Estimations des états:", self.factor_model.smoothed_states)
        else: 
            raise ValueError("Wrong extraction method, choose PC or KFS.")
        
    
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


