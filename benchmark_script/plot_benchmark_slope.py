import matplotlib.pyplot as plt
import csv
import argparse
import os

def plot_benchmark_slope(output_dir, output_filename):
    mu_list = [0.0, 0.1, 0.3, 2.0]
    colors = ['b', 'g', 'r', 'm']
    
    plt.figure(figsize=(10, 6))
    
    # Track if we have plot anything to show legend/save
    has_data = False

    for mu, color in zip(mu_list, colors):
        filename = f"benchmark_slope_mu_{mu:.1f}.csv"
        path = os.path.join(output_dir, filename)
        
        if os.path.exists(path):
            times = []
            vel_mag = []
            theory_v = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    times.append(float(row['time']))
                    vel_mag.append(float(row['vel_mag']))
                    theory_v.append(float(row['theory_v']))
            
            label_sim = f'Sim: $\mu={mu}$'
            label_theory = f'Theory: $\mu={mu}$'
            
            if mu == 0.0: 
                label_sim = 'Sim: Frictionless'
                label_theory = 'Theory: Frictionless'
            elif mu >= 0.6:
                label_sim = f'Sim: Static ($\mu={mu}$)'
                label_theory = f'Theory: Static (Expected 0)'
                
            plt.plot(times, vel_mag, color=color, linestyle='-', label=label_sim)
            plt.plot(times, theory_v, color=color, linestyle='--', alpha=0.5, label=label_theory)
            has_data = True
            
            # For reference line, just use the first/frictionless file's time
            if mu == 0.0:
                 g = 9.8
                 t_ref = times
                 v_freefall = [g * t for t in t_ref]
                 plt.plot(t_ref, v_freefall, 'k:', alpha=0.3, label='Ref: Free Fall ($a=g$)')

    if not has_data:
        print("No benchmark CSV files found.")
        return

    plt.title('Inclined Plane: Sliding Velocity vs Friction (Angle=$30^{\circ}$)')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    
    out_path = os.path.join(output_dir, output_filename)
    plt.savefig(out_path)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default="/root/workspace/taichi_dataset/benchmark_output", help="Output directory")
    args = parser.parse_args()
    
    plot_benchmark_slope(args.output_dir, "benchmark_slope_result.png")
