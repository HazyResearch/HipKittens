import json
import matplotlib.pyplot as plt
import numpy as np



mi355x_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 449.97,
        "2048": 707.80,
        "4096": 829.07,
        "8192": 894.56,
        "16384": 991.03,
    },
    "torch": {
        "1024": 34,
        "2048": 36,
        "4096": 37,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi350x_baselines_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 420.40,
        "2048": 661.89,
        "4096": 754.86,
        "8192": 829.72,
        "16384": 905.81,
    },
    "torch": {
        "1024": 31.600038,
        "2048": 34.332461,
        "4096": 35.813629,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 137.76,
        "2048": 164.12,
        "4096": 187.95,
        "8192": 257.95,
        "16384": 313.90,
    },
    "hk": {
        "1024": 401.01,
        "2048": 483.26,
        "4096": 622.12,
        "8192": 737.69,
        "16384": 780.95, # 15 batch size
    }
}


mi355x_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 352.54,
        "2048": 418.05,
        "4096": 438.59,
        "8192": 458.98,
        "16384": 462.01,
    },
    "torch": {
        "1024": 72,
        "2048": 79,
        "4096": 83,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi350x_baselines_non_causal = {
    # triton not available for gqa bwd attn
    "ck": {
        "1024": 334.65,
        "2048": 391.00,
        "4096": 416.55,
        "8192": 431.70,
        "16384": 437.35,
    },
    "torch": {
        "1024": 69.704672,
        "2048": 76.491021,
        "4096": 80.135474,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 267.64,
        "2048": 320.77,
        "4096": 349.91,
        "8192": 363.39,
        "16384": 372.18,
    },
    "hk": {
        "1024": 730.33,
        "2048": 774.57,
        "4096": 865.53,
        "8192": 899.39,
        "16384": 901.42, # 15 batch size
    }
}



colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC"]

for device in ['mi350x']:

    # Read data
    try:
        with open(f'{device}_data_to_log.json', 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading {device}_data_to_log.json: {e}")
        continue

    # Extract GQA_bkwd_asm_interleaved data
    gqa_data = {}
    for key, value in data.items():
        if key.startswith('GQA_bkwd_asm_interleaved_') and not key.endswith('_b15'):
            n_value = value['N']
            gqa_data[n_value] = {
                'tflops_tk': value['tflops_tk'],
                'tflops_ref': value['tflops_ref']
            }

    # Sort by N value
    n_values = sorted(gqa_data.keys())
    tk_tflops = [gqa_data[n]['tflops_tk'] for n in n_values]
    ref_tflops = [gqa_data[n]['tflops_ref'] for n in n_values]

    # Create grouped bar chart
    x = np.arange(len(n_values))
    width = 0.3

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width, ref_tflops, width, label='AITER (AMD)', color=colors[1])
    bars2 = ax.bar(x, tk_tflops, width, label='HipKittens', color=colors[3])

    max_tflops = max(max(tk_tflops), max(ref_tflops))

    # Add value labels on bars
    for bar, value in zip(bars1, ref_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    for bar, value in zip(bars2, tk_tflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                f'{value:.0f}', ha='center', va='bottom', fontsize=14)

    # add some padding to the top of the y-axis to prevent label overlap
    ax.set_ylim(0, max_tflops * 1.15)
    ax.set_xlabel('Sequence Length (N)', fontsize=16)
    ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
    ax.set_title(f'GQA Backwards Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_gqa_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Sequence lengths tested: {n_values}")
    print(f"AITER (CK) TFLOPS: {[f'{t:.2f}' for t in ref_tflops]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")
