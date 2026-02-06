import matplotlib.pyplot as plt
import csv
import argparse
import os

def plot_benchmark(csv_path, output_path):
    times = []
    com_y = []
    vel_y = []
    theory_y = []
    theory_v = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['time']))
            com_y.append(float(row['com_y']))
            vel_y.append(float(row['vel_y']))
            theory_y.append(float(row['theory_y']))
            theory_v.append(float(row['theory_v']))
            
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot Height
    ax1.plot(times, com_y, 'b-', label='Simulated (MPM)')
    ax1.plot(times, theory_y, 'r--', label='Analytical (Exact)')
    ax1.set_title('Free Fall: Center of Mass vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Height (m)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Velocity
    ax2.plot(times, vel_y, 'b-', label='Simulated (MPM)')
    ax2.plot(times, theory_v, 'r--', label='Analytical (Exact)')
    ax2.set_title('Free Fall: Velocity vs Time')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path', type=str, help="Path to input CSV")
    parser.add_argument('output_path', type=str, help="Path to output PNG")
    args = parser.parse_args()
    
    plot_benchmark(args.csv_path, args.output_path)
