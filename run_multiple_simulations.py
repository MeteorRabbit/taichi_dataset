import os
import sys
import subprocess

def main():
    # Define the materials to test
    materials = ['elasticity', 'newtonian', 'non_newtonian', 'plasticine', 'sand']
    
    # Base output directory
    base_output_dir = os.path.join(os.path.dirname(__file__), 'output_npy')
    os.makedirs(base_output_dir, exist_ok=True)
    
    script_path = os.path.join(os.path.dirname(__file__), 'simulate_solid_ground.py')

    # Run simulation for each material
    for material in materials:
        output_dir = os.path.join(base_output_dir, f'output_{material}')
        print(f"Running simulation for material: {material}")
        print(f"Output directory: {output_dir}")
        
        # Call the simulation script as a separate process to clean up GPU memory between runs
        try:
            cmd = [sys.executable, script_path, '--output_dir', output_dir, '--material', material]
            subprocess.check_call(cmd)
            print(f"Completed simulation for {material}\n")
        except subprocess.CalledProcessError as e:
            print(f"Error running simulation for {material}: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()