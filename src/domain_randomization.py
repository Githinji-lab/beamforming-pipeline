import numpy as np
import sys
import os

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_path)

from simulators import BeamformingSimulatorV4

class DomainRandomizer:
    """Apply domain randomization to simulator for robust training."""
    
    def __init__(self, base_simulator):
        self.base_sim = base_simulator
        self.randomization_params = {
            'fc_range': (2.5e9, 4.5e9),  # Frequency 2.5-4.5 GHz
            'distance_range': (50, 500),  # Distance 50-500 m
            'scenario_list': ['UMa_LoS', 'UMa_NLoS', 'RMa_LoS'],
            'N_tx_options': [4, 8, 16],  # Variable antenna count
            'K_options': [2, 4, 8],  # Variable user count
            'pathloss_std_range': (4, 12),  # Shadow fading std 4-12 dB
            'rician_k_range': (0, 10),  # K-factor 0-10 dB
        }
    
    def randomize_simulator(self):
        """Create randomized simulator instance."""
        fc = np.random.uniform(*self.randomization_params['fc_range'])
        distance = np.random.uniform(*self.randomization_params['distance_range'])
        scenario = np.random.choice(self.randomization_params['scenario_list'])
        
        sim = BeamformingSimulatorV4(
            N_tx=self.base_sim.N_tx,
            K=self.base_sim.K,
            fc=fc,
            distance=distance,
            scenario=scenario
        )
        
        return sim
    
    def generate_randomized_channel(self):
        """Generate channel with randomization."""
        # Occasionally randomize frequency and distance
        if np.random.rand() < 0.3:
            sim_rand = self.randomize_simulator()
            return sim_rand.generate_channel_matrix_v4()
        else:
            return self.base_sim.generate_channel_matrix_v4()


def create_augmented_training_data(base_simulator, num_samples=5000, 
                                   augmentation_strength=0.5):
    """Create training data with domain randomization."""
    
    randomizer = DomainRandomizer(base_simulator)
    
    data = {
        'H': [],
        'scenario': [],
        'snr': [],
    }
    
    print(f"Generating {num_samples} augmented training samples...")
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{num_samples}")
        
        if np.random.rand() < augmentation_strength:
            # Randomized channel
            H = randomizer.generate_randomized_channel()
            scenario = "randomized"
        else:
            # Standard channel
            H = base_simulator.generate_channel_matrix_v4()
            scenario = base_simulator.scenario
        
        snr = np.random.choice(base_simulator.snr_db_list)
        
        data['H'].append(H)
        data['scenario'].append(scenario)
        data['snr'].append(snr)
    
    # Convert to numpy arrays
    for key in data:
        if key != 'scenario':
            data[key] = np.array(data[key])
    
    return data


def adversarial_evaluation(agent, num_test_variants=10, samples_per_variant=100):
    """Test agent on diverse simulator variants."""
    
    base_sim = BeamformingSimulatorV4()
    randomizer = DomainRandomizer(base_sim)
    
    results = []
    
    print(f"\nAdversarial evaluation on {num_test_variants} simulator variants...")
    
    for variant_idx in range(num_test_variants):
        if (variant_idx + 1) % 2 == 0:
            print(f"  Variant {variant_idx + 1}/{num_test_variants}")
        
        sim_variant = randomizer.randomize_simulator()
        variant_capacities = []
        
        for _ in range(samples_per_variant):
            H = sim_variant.generate_channel_matrix_v4()
            snr = sim_variant.snr_db_list[len(sim_variant.snr_db_list) // 2]
            
            # Use agent to select beam (implementation depends on agent type)
            # This is a placeholder
            from preprocessing import preprocess_channel
            state = preprocess_channel(H, snr, 8, 4)
            
            variant_capacities.append(np.mean([1, 2, 3]))  # Placeholder
        
        avg_cap = np.mean(variant_capacities)
        results.append({
            'variant_idx': variant_idx,
            'fc': sim_variant.fc,
            'distance': sim_variant.distance,
            'scenario': sim_variant.scenario,
            'avg_capacity': avg_cap
        })
    
    print("Adversarial evaluation complete.")
    return results
