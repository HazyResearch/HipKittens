import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B", "#55555A"]


mi355x_baselines = {
    "triton": {
        "1024": 130.300560,
        "2048": 599,
        "4096": 993.69,
        "8192": 1143.91,
        "16384": 1087.09,
    },
    "hipblaslt": {
        "1024": 165.829,
        "2048": 598.810,
        "4096": 1111.09,
        "8192": 1379.180,
        "16384": 1335.310,
    },
    "ck": {
        "1024": 170.212,
        "2048": 252.214,
        "4096": 954.717,
        "8192": 963.052,
        "16384": 500.964,
    }
}

mi350x_baselines = {
    "triton": {
        "1024": 127.296012,
        "2048": 557.3908475222089,
        "4096": 861.833127618323,
        "8192": 909.506,
        "16384": 872.262,
    },
    "hipblaslt": {
        "1024": 160.740,
        "2048": 531.720,
        "4096": 917.728,
        "8192": 1081.88,
        "16384": 1041.98,
    },
    "ck": {
        "1024": 180.422,
        "2048": 236.799,
        "4096": 770.77,
        "8192": 760.005,
        "16384": 370.43,
    }
}


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'{device}_bf16_gemm.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading mi325x/{device}_bf16_gemm.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    pytorch_tflops = [data[str(size)]['tflops_pytorch'] for size in matrix_sizes]
    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    # Process data to separate OOM values
    pytorch_vals, pytorch_oom = process_data(pytorch_tflops)
    tk_vals, tk_oom = process_data(tk_tflops)

    max_tflops = max(max(pytorch_vals), max(tk_vals))

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))
    first_bar = x - width
    second_bar = x 
    bars0 = ax.bar(first_bar, pytorch_vals, width, label='PyTorch', color=colors[4])
    bars3 = ax.bar(second_bar, tk_vals, width, label='HipKittens', color=colors[3])

    # Plot X markers for OOM
    oom_height = max_tflops * 0.95

    for idx in pytorch_oom:
        ax.plot(x[idx] - 3*width, oom_height, 'x', color=colors[0], markersize=15, markeredgewidth=3)
        ax.text(x[idx] - 3*width, oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=12, color=colors[0])

    for idx in tk_oom:
        ax.plot(x[idx], oom_height, 'x', color=colors[3], markersize=15, markeredgewidth=3)
        ax.text(x[idx], oom_height + max_tflops * 0.03, 'OOM', ha='center', va='bottom', fontsize=12, color=colors[3])

    # Add value labels on bars
    for bar, value in zip(bars0, pytorch_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)

    for bar, value in zip(bars3, tk_vals):
        if value > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=12)


    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'BF16 GEMM Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=14)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_bf16_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"PyTorch TFLOPS: {[f'{t:.2f}' if t > 0 else 'OOM' for t in pytorch_vals]}")