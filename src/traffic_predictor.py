"""
Traffic Flow State Modeling Using Markov Chains

This module implements a traffic flow prediction system using Markov chain models
and Monte Carlo simulations to analyze and predict traffic states in urban areas.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from datetime import datetime
import random
from typing import Dict, List, Tuple, Optional


class TrafficPredictor:
    """
    A traffic flow prediction system using Markov chains and Monte Carlo simulations.
    
    This class implements a stochastic traffic flow model that predicts traffic states
    based on various environmental factors including weather conditions, time of day,
    and taxi density.
    """
    
    def __init__(self):
        """Initialize the TrafficPredictor with base transition matrices and modifiers."""
        # Traffic states
        self.traffic_states = ['Free Flow', 'Congested Flow', 'Heavy Congestion']
        
        # Base transition matrices for different conditions
        self.base_matrix = {
            'Free Flow': {'Free Flow': 0.7, 'Congested Flow': 0.25, 'Heavy Congestion': 0.05},
            'Congested Flow': {'Free Flow': 0.2, 'Congested Flow': 0.6, 'Heavy Congestion': 0.2},
            'Heavy Congestion': {'Free Flow': 0.1, 'Congested Flow': 0.3, 'Heavy Congestion': 0.6}
        }
        
        # Modifiers for different conditions
        self.weather_mod = {'Clear': 1.0, 'Rain': 0.8, 'Snow': 0.6}
        self.time_mod = {
            'Night': 1.2,    # Better flow at night
            'Morning': 0.7,  # Worse in morning rush
            'Afternoon': 0.9,
            'Evening': 0.75  # Worse in evening rush
        }
        self.density_mod = {
            'Low': 1.2,
            'Medium': 1.0,
            'High': 0.8
        }
    
    def get_time_of_day(self) -> str:
        """Categorize current time into time bins."""
        hour = datetime.now().hour
        if 5 <= hour < 10:
            return 'Morning'
        elif 10 <= hour < 15:
            return 'Afternoon'
        elif 15 <= hour < 20:
            return 'Evening'
        else:
            return 'Night'
    
    def get_taxi_density(self, df: pd.DataFrame) -> str:
        """Estimate density based on number of taxis in region."""
        num_taxis = len(df)
        if num_taxis < 10:
            return 'Low'
        elif num_taxis < 20:
            return 'Medium'
        else:
            return 'High'
    
    def build_transition_matrix(self, weather: str, time_of_day: str, density: str) -> Dict:
        """Build dynamic transition matrix based on conditions."""
        matrix = {}
        weather_factor = self.weather_mod.get(weather, 1.0)
        time_factor = self.time_mod.get(time_of_day, 1.0)
        density_factor = self.density_mod.get(density, 1.0)
        combined_factor = weather_factor * time_factor * density_factor
        
        for from_state in self.traffic_states:
            adjusted = {}
            total = 0
            
            for to_state in self.traffic_states:
                base_prob = self.base_matrix[from_state][to_state]
                
                # Apply modifiers based on state transitions
                if from_state == 'Free Flow' and to_state != 'Free Flow':
                    adjusted_prob = base_prob * (1/combined_factor)
                elif from_state != 'Heavy Congestion' and to_state == 'Heavy Congestion':
                    adjusted_prob = base_prob * (1/combined_factor)
                else:
                    adjusted_prob = base_prob * combined_factor
                
                adjusted[to_state] = adjusted_prob
                total += adjusted_prob
            
            # Normalize probabilities
            matrix[from_state] = {k: v/total for k, v in adjusted.items()}
        
        return matrix
    
    def simulate_traffic_states(self, initial_state: str, transition_matrix: Dict, 
                              num_steps: int) -> List[str]:
        """
        Simulate traffic flow states using the Markov chain model.
        
        Parameters:
            initial_state (str): The initial traffic state
            transition_matrix (dict): The transition probability matrix
            num_steps (int): The number of time steps to simulate
        
        Returns:
            list: A list of traffic states over time
        """
        states = [initial_state]
        for _ in range(num_steps):
            current_state = states[-1]
            next_state = np.random.choice(
                list(transition_matrix[current_state].keys()),
                p=list(transition_matrix[current_state].values())
            )
            states.append(next_state)
        return states
    
    def estimate_steady_state(self, transition_matrix: Dict, tolerance: float = 1e-6, 
                            max_iterations: int = 1000) -> Dict:
        """
        Estimate the steady-state probabilities of the Markov chain.
        
        Parameters:
            transition_matrix (dict): The transition probability matrix
            tolerance (float): Convergence tolerance
            max_iterations (int): Maximum number of iterations
        
        Returns:
            dict: Steady-state probabilities for each state
        """
        states = list(transition_matrix.keys())
        n_states = len(states)
        
        # Convert transition matrix to a numpy array
        P = np.zeros((n_states, n_states))
        for i, state in enumerate(states):
            for j, next_state in enumerate(states):
                P[i, j] = transition_matrix[state].get(next_state, 0)
        
        # Initialize steady-state vector
        pi = np.ones(n_states) / n_states
        
        # Iterate until convergence
        for _ in range(max_iterations):
            pi_new = pi @ P
            if np.linalg.norm(pi_new - pi) < tolerance:
                break
            pi = pi_new
        
        # Return steady-state probabilities as a dictionary
        steady_state = {state: pi[i] for i, state in enumerate(states)}
        return steady_state
    
    def monte_carlo_simulation(self, current_state: str, weather: str, holiday: bool,
                             steps: int = 10, num_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulation of traffic state transitions."""
        results = []
        transition_matrix = self.build_transition_matrix(weather, self.get_time_of_day(), 'Medium')
        
        # Adjust for holiday effects
        if holiday:
            for from_state in transition_matrix:
                for to_state in transition_matrix[from_state]:
                    if to_state == 'Free Flow':
                        transition_matrix[from_state][to_state] *= 0.9
                    else:
                        transition_matrix[from_state][to_state] *= 1.1
                # Normalize
                total = sum(transition_matrix[from_state].values())
                transition_matrix[from_state] = {k: v/total for k, v in transition_matrix[from_state].items()}
        
        for _ in range(num_simulations):
            state_sequence = [current_state]
            for _ in range(steps):
                current = state_sequence[-1]
                next_state = np.random.choice(
                    list(transition_matrix[current].keys()),
                    p=list(transition_matrix[current].values())
                )
                state_sequence.append(next_state)
            results.append(state_sequence)
        
        # Calculate state probabilities at each time step
        state_counts = defaultdict(lambda: [0] * (steps + 1))
        for seq in results:
            for i, state in enumerate(seq):
                state_counts[state][i] += 1
        
        state_probs = {
            state: [count / num_simulations for count in counts]
            for state, counts in state_counts.items()
        }
        
        # Ensure all states are represented even if they never occurred
        for state in self.traffic_states:
            if state not in state_probs:
                state_probs[state] = [0.0] * (steps + 1)
        
        return state_probs
    
    def predict(self, df: pd.DataFrame, steps: int = 24, num_simulations: int = 1000,
               weather: Optional[str] = None, holiday: Optional[bool] = None) -> Dict:
        """Predict traffic flow using Markov chain simulation."""
        if weather is None:
            weather = random.choice(['Clear', 'Rain', 'Snow'])
        if holiday is None:
            holiday = random.choice([True, False])
            
        time_of_day = self.get_time_of_day()
        density = self.get_taxi_density(df)
        
        print(f"\nPrediction Conditions:")
        print(f"Weather: {weather}, Time: {time_of_day}, Taxi Density: {density}")
        print(f"Holiday: {holiday}")
        
        transition_matrix = self.build_transition_matrix(weather, time_of_day, density)
        current_state = 'Free Flow'  # Default starting state
        
        # Run Markov chain simulation
        state_counts = {state: 0 for state in self.traffic_states}
        current = current_state
        
        for _ in range(steps):
            next_state = np.random.choice(
                list(transition_matrix[current].keys()),
                p=list(transition_matrix[current].values())
            )
            state_counts[next_state] += 1
            current = next_state
        
        # Convert counts to probabilities
        total = sum(state_counts.values())
        state_probs = {k: v/total for k, v in state_counts.items()}
        
        # Visualization
        self.plot_results(state_probs, steps, weather, time_of_day, density, holiday)
        
        return state_probs
    
    def plot_results(self, probs: Dict, steps: int, weather: str, time_of_day: str, 
                    density: str, holiday: bool):
        """Visualize the prediction results."""
        plt.figure(figsize=(10, 6))
        
        # Bar plot of state probabilities
        states = list(probs.keys())
        probabilities = list(probs.values())
        
        colors = ['green', 'orange', 'red']
        bars = plt.bar(states, probabilities, color=colors)
        
        # Add probability labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom')
        
        plt.title('Traffic State Probabilities\n' +
                 f'Conditions: {weather}, {time_of_day}, {density} Density, Holiday: {holiday}')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def calculate_region_traffic(self, df: pd.DataFrame, weather: str, holiday: bool,
                               steps: int = 10, num_simulations: int = 500) -> Dict:
        """Calculate aggregated traffic probabilities for all stands in the region."""
        all_probs = {state: np.zeros(steps + 1) for state in self.traffic_states}
        
        for _, row in df.iterrows():
            stand_probs = self.monte_carlo_simulation(
                current_state='Free Flow',
                weather=weather,
                holiday=holiday,
                steps=steps,
                num_simulations=num_simulations
            )
            
            for state in self.traffic_states:
                all_probs[state] += np.array(stand_probs[state])
        
        # Average the probabilities across all stands
        num_stands = len(df)
        for state in self.traffic_states:
            all_probs[state] /= num_stands
        
        return all_probs


def main():
    """Main function to demonstrate the TrafficPredictor."""
    # Load data from CSV file
    try:
        df = pd.read_csv("data/taxi_stands_data.csv")
    except FileNotFoundError:
        print("Error: CSV file not found. Using sample data instead.")
        data = {
            'ID': range(1, 21),  # Simulating 20 taxis
            'Descricao': [f"Stand_{i}" for i in range(1, 21)],
            'Latitude': np.random.uniform(41.14, 41.19, 20),
            'Longitude': np.random.uniform(-8.67, -8.56, 20)
        }
        df = pd.DataFrame(data)
    
    # Initialize and run predictor
    predictor = TrafficPredictor()
    results = predictor.predict(df)
    
    print("\nPredicted Traffic State Probabilities:")
    for state, prob in results.items():
        print(f"{state}: {prob:.1%}")


if __name__ == "__main__":
    main()
