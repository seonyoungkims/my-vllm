import re
import matplotlib.pyplot as plt
import ast
import os

def draw_plots_from_file(file_path):
    # Lists and dictionaries for data storage
    iterations = []
    batch_sizes = []
    request_lifetimes = {}  # {id: [start_iter, end_iter]}

    # Regex for data extraction from logs
    # Format: Iteration: [num], Batch Size: [num], IDs: [list]
    pattern = r"Iteration: (\d+), Batch Size: (\d+), IDs: (\[.*?\])"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                it = int(match.group(1))
                size = int(match.group(2))
                # Convert IDs string into an actual list
                try:
                    ids = ast.literal_eval(match.group(3))
                except Exception:
                    continue

                iterations.append(it)
                batch_sizes.append(size)

                for req_id in ids:
                    if req_id not in request_lifetimes:
                        request_lifetimes[req_id] = [it, it]
                    else:
                        request_lifetimes[req_id][1] = it

    # --- Visualization Settings ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4)

    # Determine X-axis limits with padding on the left
    x_min = -20
    x_max = max(iterations) + 5 if iterations else 100

    # Plot 1: Batch size over iteration
    ax1.step(iterations, batch_sizes, where='post', color='#1f77b4', linewidth=2)
    ax1.set_title("Plot 1: Batch Size over Iteration Steps", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Iteration Step", fontsize=12)
    ax1.set_ylabel("Batch Size", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(0, max(batch_sizes) + 1 if batch_sizes else 6)
    # Apply left-side padding
    ax1.set_xlim(left=x_min, right=x_max)

    # Plot 2: Life of each request (Span)
    # Sort IDs numerically for a cleaner Y-axis layout
    sorted_ids = sorted(request_lifetimes.keys(), key=lambda x: int(x))
    
    for i, req_id in enumerate(sorted_ids):
        start, end = request_lifetimes[req_id]
        # barh(y, width, left)
        ax2.barh(req_id, end - start, left=start, color='#ff7f0e', edgecolor='white', height=0.6)

    ax2.set_title("Plot 2: Life of Each Request (Span)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Iteration Step", fontsize=12)
    ax2.set_ylabel("Request ID", fontsize=12)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.6)
    ax2.tick_params(axis='y', labelsize=8)

    # Apply left-side padding (Synced with ax1)
    ax2.set_xlim(left=x_min, right=x_max)

    # Save and display results
    output_filename = "/content/drive/MyDrive/CON_batch_analysis_result.png"
    plt.savefig(output_filename, dpi=300)
    print(f"Plot has been saved to: {output_filename}")
    plt.show()

# --- Execution ---
# Replace with your actual log file path
draw_plots_from_file("/content/drive/MyDrive/continuous.txt")