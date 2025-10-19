import json
import matplotlib.pyplot as plt
import numpy as np



mi355x_baselines_causal = {
    "triton": {
        "1024": 147,
        "2048": 179,
        "4096": 215,
        "8192": 237,
        "16384": 251,
    },
    "ck": {
        "1024": 372.36,
        "2048": 622.46,
        "4096": 800.12,
        "8192": 870.58,
        "16384": 986.59,
    },
    "torch": {
        "1024": 34,
        "2048": 36,
        "4096": 38,
        "8192": 37,
        "16384": "OOM",
    }
}

mi350x_baselines_causal = {
    "triton": {
        "1024": 139.080476,
        "2048": 163.599388,
        "4096": 188.696534,
        "8192": 200.207510,
        "16384": 206.924580,
    },
    "ck": {
        "1024": 350.65,
        "2048": 579.85,
        "4096": 728.33,
        "8192": 813.47,
        "16384": 886.17,
    },
    "torch": {
        "1024": 31.582108,
        "2048": 33.775558,
        "4096": 35.817826,
        "8192": 34.693869,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 416.30,
        "2048": 587.63,
        "4096": 735.77,
        "8192": 799.96,
        "16384": 848.04,
    },
    "hk": {
        "1024": 349.09,
        "2048": 520.36,
        "4096": 698.59,
        "8192": 767.11,
        "16384": 814.23,
    }
}


mi355x_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 348.60,
        "2048": 415.37,
        "4096": 437.66,
        "8192": 465.44,
        "16384": 465.83,
    },
    "torch": {
        "1024": 71,
        "2048": 78,
        "4096": 84,
        "8192": 81,
        "16384": "OOM",
    }
}

mi350x_baselines_non_causal = { 
    # triton not available for non-causal bwd attn
    "ck": {
        "1024": 331.73,
        "2048": 391.71,
        "4096": 412.22,
        "8192": 434.40,
        "16384": 443.33,
    },
    "torch": {
        "1024": 70.188240,
        "2048": 76.365602,
        "4096": 81.948592,
        "8192": 79.742189,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 596.54,
        "2048": 776.44,
        "4096": 824.74,
        "8192": 892.54,
        "16384": 919.44,
    },
    "hk": {
        "1024": 611.45,
        "2048": 776.34,
        "4096": 814.93,
        "8192": 850.11,
        "16384": 877.71,
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

    # Extract MHA_bkwd_asm_interleaved data
    mha_data = {}
    for key, value in data.items():
        if key.startswith('MHA_bkwd_asm_interleaved_'):
            n_value = value['N']
            mha_data[n_value] = {
                'tflops_tk': value['tflops_tk'],
                'tflops_ref': value['tflops_ref']
            }

    # Sort by N value
    n_values = sorted(mha_data.keys())
    tk_tflops = [mha_data[n]['tflops_tk'] for n in n_values]
    ref_tflops = [mha_data[n]['tflops_ref'] for n in n_values]

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
    ax.set_title(f'MHA Backwards Performance Comparison {device.upper()}', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values], fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=16)

    plt.tight_layout()
    plt.show()

    output_file = f'{device}_mha_bkwd_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

    # Print summary
    print(f"Sequence lengths tested: {n_values}")
    print(f"AITER (ASM) TFLOPS: {[f'{t:.2f}' for t in ref_tflops]}")
    print(f"HipKittens TFLOPS: {[f'{t:.2f}' for t in tk_tflops]}")
