import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract data for plotting
    matrix_sizes = sorted([int(size) for size in data.keys()])
    pytorch_tflops = [data[str(size)]['tflops_pytorch'] for size in matrix_sizes]
    try:
        aiter_tflops = [data[str(size)]['tflops_aiter'] for size in matrix_sizes]
    except KeyError:
        aiter_tflops = None

    try:
        hipblaslt_tflops = [data[str(size)]['tflops_hipblaslt'] for size in matrix_sizes]
    except KeyError:
        hipblaslt_tflops = None

    tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

    # Create bar chart
    x = np.arange(len(matrix_sizes))
    width = 0.23

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x - width, pytorch_tflops, width, label='PyTorch', color=colors[0])
     
    if aiter_tflops is not None:
        bars1 = ax.bar(x, aiter_tflops, width, label='AITER (AMD)', color=colors[1])
    else:
        bars1 = None

    if hipblaslt_tflops is not None:
        bars2 = ax.bar(x + width, hipblaslt_tflops, width, label='HipblasLT', color=colors[2])
    else:
        bars2 = None

    if aiter_tflops is not None and hipblaslt_tflops is not None:
        bars3 = ax.bar(x + 2 * width, tk_tflops, width, label='ThunderKittens', color=colors[3])
    elif aiter_tflops is not None:
        bars3 = ax.bar(x + width, tk_tflops, width, label='ThunderKittens', color=colors[3])
    else:
        bars3 = ax.bar(x, tk_tflops, width, label='ThunderKittens', color=colors[3])

    if aiter_tflops is not None:
        max_tflops = max(max(pytorch_tflops), max(aiter_tflops), max(tk_tflops))
        if hipblaslt_tflops is not None:
            max_tflops = max(max_tflops, max(hipblaslt_tflops))
    else:
        max_tflops = max(max(pytorch_tflops), max(tk_tflops))
        if hipblaslt_tflops is not None:
            max_tflops = max(max_tflops, max(hipblaslt_tflops))

    # Add value labels on bars
    for bar, value in zip(bars0, pytorch_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    if bars1 is not None:
        for bar, value in zip(bars1, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    if hipblaslt_tflops is not None:
        for bar, value in zip(bars2, hipblaslt_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars3, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Matrix Size (N×N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'BF16 GEMM Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(matrix_sizes, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_bf16_gemm_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Matrix sizes tested: {matrix_sizes}")
    print(f"PyTorch TFLOPS: {[f'{t:.2f}' for t in pytorch_tflops]}")
    if aiter_tflops is not None:
        print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")
    print(f"TK TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")

