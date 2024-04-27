import numpy as np

class KalmanFilterSmoother:
    def __init__(self, A, C, Q, R, initial_state, initial_covariance):
        """
        Initialise le filtre de Kalman.
        
        :param A: Matrice de transition d'état.
        :param C: Matrice d'observation.
        :param Q: Matrice de covariance du bruit du processus.
        :param R: Matrice de covariance du bruit d'observation.
        :param initial_state: État initial estimé du système.
        :param initial_covariance: Covariance initiale estimée.
        """

        self.A = A  # Matrice de transition d'état
        self.C = C  # Matrice d'observation
        self.Q = Q  # Bruit du processus
        self.R = R  # Bruit d'observation
        # S'assurer que l'état initial est un vecteur colonne
        if initial_state.ndim == 1:
            initial_state = initial_state.reshape(-1, 1)
        self.current_state_estimate = initial_state
        self.current_covariance_estimate = initial_covariance
        self.states = []  # Stocker les états estimés pour le lissage
        self.covariances = []  # Stocker les covariances estimées pour le lissage

    def kalman_filter(self, observation):
        """
        Applique le filtre de Kalman pour une seule observation.
        
        :param observation: Observation à un instant t.
        """
        # Prédiction
        predicted_state = self.A @ self.current_state_estimate
        predicted_covariance = self.A @ self.current_covariance_estimate @ self.A.T + self.Q
        
        # Mise à jour
        observation = np.atleast_2d(observation).T  # Assurer le format vecteur colonne
        innovation = observation - self.C @ predicted_state
        innovation_covariance = self.C @ predicted_covariance @ self.C.T + self.R
        kalman_gain = predicted_covariance @ self.C.T @ np.linalg.inv(innovation_covariance)
        self.current_state_estimate = predicted_state + kalman_gain @ innovation
        self.current_covariance_estimate = (np.eye(self.A.shape[0]) - kalman_gain @ self.C) @ predicted_covariance
        
        # Stockage des états pour le lissage
        self.states.append(self.current_state_estimate.copy())
        self.covariances.append(self.current_covariance_estimate.copy())

    def kalman_smoother(self):
        """
        Applique le lisseur de Kalman pour estimer les états passés.
        """
        smoothed_states = self.states[::-1]  # Inversion de la séquence des états pour le lissage
        smoothed_covariances = self.covariances[::-1]  # Inversion de la séquence des covariances pour le lissage
        # Commence avec l'estimation finale du filtre de Kalman
        previous_smoothed_state = smoothed_states[0]
        previous_smoothed_covariance = smoothed_covariances[0]
        
        for t in range(1, len(smoothed_states)):
            # Extraction des estimations à l'instant t
            current_state = self.states[-t-1]
            current_covariance = self.covariances[-t-1]
            next_covariance = self.covariances[-t]
            
            # Calcul du gain de lissage
            smoothing_gain = current_covariance @ self.A.T @ np.linalg.inv(next_covariance)
            smoothed_state = current_state + smoothing_gain @ (previous_smoothed_state - self.A @ current_state)
            smoothed_covariance = current_covariance + smoothing_gain @ (previous_smoothed_covariance - next_covariance) @ smoothing_gain.T
            
            # Mise à jour des états lissés
            previous_smoothed_state = smoothed_state
            previous_smoothed_covariance = smoothed_covariance
            smoothed_states[t] = smoothed_state
            smoothed_covariances[t] = smoothed_covariance

        # Inversion des séquences lissées pour correspondre à l'ordre de temps original
        self.smoothed_states = smoothed_states[::-1]
        self.smoothed_covariances = smoothed_covariances[::-1]
    def e_step(self):
        """
        Calcule les attentes pour l'étape E de l'algorithme EM.
        """
         
        # Vérifie que les états lissés sont une matrice 2D
        smoothed_states_matrix = np.atleast_2d([state for state in self.smoothed_states])
        self.expected_states = np.mean(smoothed_states_matrix, axis=0)
        self.expected_covariances = np.mean(self.smoothed_covariances, axis=0)

    def m_step(self):
        """
        Étape de Maximisation (M) de l'algorithme EM.
        Met à jour les paramètres du modèle en maximisant la fonction de vraisemblance attendue.
        """
        # Assurons-nous que les états attendus sont une matrice avec les périodes en lignes et les états en colonnes
        expected_states_2d = np.atleast_2d(self.expected_states).T
        
        # Vérifions si nous avons suffisamment de données pour procéder
        if expected_states_2d.shape[0] > 1:
            # Calculons la pseudo-inverse des états attendus
            pinv_expected_states = np.linalg.pinv(expected_states_2d[:-1])
            # Mettons à jour la matrice de transition A
            self.A = np.dot(expected_states_2d[1:], pinv_expected_states)
        else:
            # Gérer le cas avec des données insuffisantes
            print("Pas assez de données pour mettre à jour A.")


    def em_algorithm(self, observations, iterations=10):
        """
        Applique l'algorithme EM pour ajuster le modèle de facteurs dynamiques.
        """
         
        self.observations = observations
        
        for _ in range(iterations):
            for observation in observations:
                self.kalman_filter(observation)
            self.kalman_smoother()
            self.e_step()
            self.m_step()
