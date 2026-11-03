import csv
import math
import random
import os

def generate_realistic_telemetry(filename, num_ticks, tick_rate=1.5):
    filepath = os.path.join(os.path.dirname(__file__), "simulator", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["speed_kmh", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z", "tilt_roll", "tilt_pitch", "label"])
        
        speed = 35.0
        for i in range(num_ticks):
            # Baseline cruising behavior
            behavior = "SAFE"
            accel_x = random.gauss(0, 0.2)
            accel_y = random.gauss(0, 0.3)
            accel_z = 9.8 + random.gauss(0, 0.15)
            gyro_x, gyro_y, gyro_z = random.gauss(0, 1.5), random.gauss(0, 1.5), random.gauss(0, 3)
            tilt_roll = max(-8, min(8, random.gauss(0, 2)))
            tilt_pitch = max(-6, min(6, random.gauss(0, 1.5)))
            
            # Inject a harsh braking event for testing (e.g., around tick 50)
            if 50 <= i <= 53:
                behavior = "AGGRESSIVE"
                accel_y = random.uniform(-8.0, -5.5)
                speed = max(10, speed - 15)
            # Inject a sudden acceleration event
            elif 80 <= i <= 83:
                behavior = "AGGRESSIVE"
                accel_y = random.uniform(4.5, 7.0)
                speed += 12
            else:
                speed = max(15, min(70, speed + random.uniform(-2, 2)))

            writer.writerow([
                round(speed, 1), round(accel_x, 2), round(accel_y, 2), round(accel_z, 2),
                round(gyro_x, 1), round(gyro_y, 1), round(gyro_z, 1), 
                round(tilt_roll, 1), round(tilt_pitch, 1), behavior
            ])
    print(f"Generated {filepath}")

if __name__ == "__main__":
    # Generate 150 ticks (~3.5 minutes) of unseen raw ride data
    generate_realistic_telemetry("raw_unseen_ride.csv", 150)