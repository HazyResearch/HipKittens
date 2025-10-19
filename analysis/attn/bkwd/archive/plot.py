import json
import matplotlib.pyplot as plt
import numpy as np

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]


for device in ['mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract data for plotting
    tk_4warps_tflops = [data['MHA_bkwd_4warps_4096']['tflops_tk']]
    tk_asm_interleaved_tflops = [data['MHA_bkwd_asm_interleaved_4096']['tflops_tk']]
    aiter_tflops = [data['MHA_bkwd_asm_interleaved_4096']['tflops_ref']]

    # Create bar chart
    x = np.arange(3)
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 6))
    bars0 = ax.bar(x[0], tk_4warps_tflops, width, label='HipKittens', color=colors[3])
    bars1 = ax.bar(x[1], tk_asm_interleaved_tflops, width, label='HipKittens ASM', color=colors[0])
    bars2 = ax.bar(x[2], aiter_tflops, width, label='AITER (AMD)', color=colors[1])

    max_tflops = max(max(aiter_tflops), max(tk_4warps_tflops), max(tk_asm_interleaved_tflops))

    # Add value labels on bars
    for bar, value in zip(bars0, tk_4warps_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars1, tk_asm_interleaved_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars2, aiter_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.25)
    ax.set_xlabel('Kernel', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title('MHA Backwards Performance Comparison MI350X (N=4096)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(['HK 4-warps', 'HK ASM 4-warps', 'AITER (ASM)'], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16, loc='upper left')
    # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_attn_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"HipKittens (4 warps) TFLOPS: {[f'{t:.2f}' for t in tk_4warps_tflops]}")
    print(f"HipKittens (ASM Interleaved) TFLOPS: {[f'{t:.2f}' for t in tk_asm_interleaved_tflops]}")
    print(f"AITER (AMD) TFLOPS: {[f'{t:.2f}' for t in aiter_tflops]}")

