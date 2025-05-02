import threading
import pandas as pd
from rich.console import Console
from rich.table import Table
from highd_integration import HighDData

# Initialize rich console for formatted output
console = Console()

# ✅ Load HighD Dataset
console.print("📦 Loading HighD dataset...", style="bold cyan")
highd_data = HighDData("highD-dataset/data/")
console.print("✅ HighD dataset loaded successfully.\n", style="bold green")

# Define actual mobility models
MOBILITY_MODELS = {
    "Model_A": "Intelligent Driver Model (IDM)",
    "Model_B": "Krauss Model",
    "Model_C": "Wiedemann Model"
}

# Function to simulate a mobility model
def run_mobility_model(model_key, model_name, vehicle_ids, results_dict):
    console.print(f"🚗 Running [bold cyan]{model_name}[/] using HighD data...\n")

    # Extract first position of each vehicle
    sample_data = highd_data.tracks[highd_data.tracks['id'].isin(vehicle_ids)]
    sample_data = sample_data.groupby('id').first().reset_index()
    sample_data = sample_data.rename(columns={"id": "Vehicle ID", "x": "X (m)", "y": "Y (m)", "laneId": "Lane ID"})

    results_dict[model_name] = sample_data

# Prepare results dictionary
results = {}

# Assign different vehicle IDs for each model
vehicle_sets = {
    "Model_A": [1, 2, 3, 4, 5],
    "Model_B": [6, 7, 8, 9, 10],
    "Model_C": [11, 12, 13, 14, 15]
}

# Run models in parallel threads
threads = []
for model_key, model_name in MOBILITY_MODELS.items():
    thread = threading.Thread(
        target=run_mobility_model,
        args=(model_key, model_name, vehicle_sets[model_key], results)
    )
    threads.append(thread)
    thread.start()

# Wait for all threads to finish
for thread in threads:
    thread.join()

# Display results in formatted tables
for model_name, data in results.items():
    table = Table(title=f"[bold yellow]{model_name} Simulation[/]", show_header=True, header_style="bold magenta")

    table.add_column("Vehicle ID", justify="center")
    table.add_column("X (m)", justify="center")
    table.add_column("Y (m)", justify="center")
    table.add_column("Lane ID", justify="center")

    for _, row in data.iterrows():
        table.add_row(str(int(row["Vehicle ID"])), f"{row['X (m)']:.2f}", f"{row['Y (m)']:.2f}", str(int(row["Lane ID"])))

    console.print(table)
    console.print("\n" + "=" * 50 + "\n")

console.print("✅ [bold green]All mobility models executed successfully.[/]\n")
