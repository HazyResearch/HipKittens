import json
import matplotlib.pyplot as plt
import numpy as np


# B = 16, H = 64, HK8, D = 128.
mi355x_gqa_baselines_causal = {
    "triton": {
        "1024": 456,
        "2048": 569,
        "4096": 614,
        "8192": 764,
        "16384": 848,
    },
    "ck": {
        "1024": 596.53,
        "2048": 695.71,
        "4096": 799.97,
        "8192": 861.79,
        "16384": 878.86,
    },
    "torch": {
        "1024": 14,
        "2048": 15,
        "4096": 15,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi355x_gqa_baselines_non_causal = {
    "triton": {
        "1024": 844,
        "2048": 945,
        "4096": 996,
        "8192": 1005,
        "16384":1011,
    },
    "ck": {
        "1024": 799,
        "2048": 847,
        "4096": 884,
        "8192": 904,
        "16384": 901,
    },
    "torch": {
        "1024": 29,
        "2048": 31,
        "4096": 34,
        "8192": "OOM",
        "16384": "OOM",
    }
}

mi350x_gqa_baselines_causal = {
    "triton": {
        "1024": 661.689624,
        "2048": 736.288587,
        "4096": 777.730750,
        "8192": 806.935961,
        "16384":811.201966,
    },
    "ck": {
        "1024": 611,
        "2048": 653,
        "4096": 694,
        "8192": 702,
        "16384": 691,
    },
    "torch": {
        "1024": 26.912733,
        "2048": 29.571504,
        "4096": 31.270997,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 818.35,
        "2048": 967.91,
        "4096": 968.28,
        "8192": 983.25,
        "16384": 1009.18,
    }
}

mi350x_gqa_baselines_non_causal = {
    "triton": {
        "1024": 398.220437,
        "2048": 462.552210,
        "4096": 488.052800,
        "8192": 618.134744,
        "16384": 691.502584,
    },
    "ck": {
        "1024": 461,
        "2048": 557,
        "4096": 627,
        "8192": 654,
        "16384": 677,
    },
    "torch": {
        "1024": 12.668953,
        "2048": 13.742200,
        "4096": 14.442294,
        "8192": "OOM",
        "16384": "OOM",
    },
    "aiter": {
        "1024": 567.01,
        "2048": 782.79,
        "4096": 844.96,
        "8192": 915.63,
        "16384": 968.90,
    }
}


# B = 16, H = 16, D = 128.
mi355x_mha_baselines_causal = {
    "triton": {
        "1024": 371,
        "2048": 508,
        "4096": 573,
        "8192": 733,
        "16384": 845,
    },
    "ck": {
        "1024": 485,
        "2048": 601,
        "4096": 745,
        "8192": 834,
        "16384": 893,
    },
    "torch": {
        "1024": 13,
        "2048": 14,
        "4096": 15,
        "8192": 15,
        "16384": "OOM",
    }
}

# B = 16, H = 16, D = 128.
mi355x_mha_baselines_non_causal = {
    "triton": {
        "1024": 694,
        "2048": 855,
        "4096": 944,
        "8192": 1001,
        "16384": 1011,
    },
    "ck": {
        "1024": 761,
        "2048": 733,
        "4096": 816,
        "8192": 896,
        "16384": 914,
    },
    "torch": {
        "1024": 29,
        "2048": 32,
        "4096": 34,
        "8192": 33,
        "16384": "OOM",
    }
}

mi350x_mha_baselines_non_causal = {
    "triton": {
        "1024": 314.211205,
        "2048": 424.128758,
        "4096": 473.762229,
        "8192": 593.775992,
        "16384": 654.668156,
    },
    "ck": {
        "1024": 412,
        "2048": 504,
        "4096": 600,
        "8192": 663,
        "16384": 700,
    },
    "torch": {
        "1024": 12.70,
        "2048": 13.45,
        "4096": 14.26,
        "8192": 13.84,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 392.17,
        "2048": 698.19,
        "4096": 953.31,
        "8192": 909.30,
        "16384": 952.64,
    }
}

mi350x_mha_baselines_causal = {
    "triton": {
        "1024": 554.682706,
        "2048": 667.633763,
        "4096": 736.837676,
        "8192": 775.493057,
        "16384": 802.799758,
    },
    "ck": {
        "1024": 591,
        "2048": 603,
        "4096": 665,
        "8192": 700,
        "16384": 715,
    },
    "torch": {
        "1024": 26.98,
        "2048": 29.24,
        "4096": 30.99,
        "8192": 30.72,
        "16384": "OOM",
    },
    "aiter": {
        "1024": 570.79,
        "2048": 896.85,
        "4096": 976.44,
        "8192": 956.59,
        "16384": 974.80,
    }
}

colors = ["#8E69B8", "#E59952", "#68AC5A", "#7CB9BC", "#DE836B"]


def process_data(data_list):
    """Separate numeric values and OOM indices"""
    values = []
    oom_indices = []
    for i, val in enumerate(data_list):
        if val == "OOM":
            values.append(0)  # Use 0 for bar height
            oom_indices.append(i)
        else:
            values.append(val)
    return values, oom_indices


for device in ['mi300x', 'mi325x', 'mi350x', 'mi355x']:

    for setting in ['mha_causal_fwd', 'mha_non_causal_fwd', 'gqa_causal_fwd', 'gqa_non_causal_fwd']:

        # Read data
        try:
            with open(f'mi350x/{device}_{setting}.json', 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {device}_{setting}.json: {e}")
            continue

        # Extract data for plotting
        matrix_sizes = sorted([int(size) for size in data.keys()])
        aiter_tflops = [data[str(size)]['tflops_ref'] for size in matrix_sizes]
        tk_tflops = [data[str(size)]['tflops'] for size in matrix_sizes]

        triton_tflops = []
        torch_tflops = []
        ck_tflops = []
        if setting == 'mha_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_mha_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_fwd' and device == 'mi355x':
            triton_tflops = [mi355x_gqa_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi355x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi355x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]

        elif setting == 'mha_causal_fwd' and device == 'mi350x':
            triton_tflops = [mi350x_mha_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_mha_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'mha_non_causal_fwd' and device == 'mi350x':
            triton_tflops = [mi350x_mha_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_mha_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_mha_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_causal_fwd' and device == 'mi350x':
            triton_tflops = [mi350x_gqa_baselines_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_gqa_baselines_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_causal['ck'][str(size)] for size in matrix_sizes]
        elif setting == 'gqa_non_causal_fwd' and device == 'mi350x':
            triton_tflops = [mi350x_gqa_baselines_non_causal['triton'][str(size)] for size in matrix_sizes]
            torch_tflops = [mi350x_gqa_baselines_non_causal['torch'][str(size)] for size in matrix_sizes]
            ck_tflops = [mi350x_gqa_baselines_non_causal['ck'][str(size)] for size in matrix_sizes]
            
        # Process data to separate OOM values
        triton_vals, triton_oom = process_data(triton_tflops) if triton_tflops else ([], [])
        torch_vals, torch_oom = process_data(torch_tflops) if torch_tflops else ([], [])
        ck_vals, ck_oom = process_data(ck_tflops) if ck_tflops else ([], [])

        # Calculate max for numeric values only
        numeric_vals = aiter_tflops + tk_tflops
        if triton_vals:
            numeric_vals.extend([v for v in triton_vals if v != 0])
        if torch_vals:
            numeric_vals.extend([v for v in torch_vals if v != 0])
        if ck_vals:
            numeric_vals.extend([v for v in ck_vals if v != 0])
        max_tflops = max(numeric_vals) if numeric_vals else 100

        # Create bar chart
        x = np.arange(len(matrix_sizes))
        width = 0.17

        fig, ax = plt.subplots(figsize=(16, 6))
        first_bar_start = x - 2*width
        second_bar_start = x - width
        third_bar_start = x
        fourth_bar_start = x + width
        fifth_bar_start = x + 2*width
        bars0 = ax.bar(fourth_bar_start, aiter_tflops, width, label='AITER', color=colors[0])
        bars1 = ax.bar(fifth_bar_start, tk_tflops, width, label='HipKittens', color=colors[3])
        bars2 = ax.bar(second_bar_start, triton_vals, width, label='Triton', color=colors[2])
        bars3 = ax.bar(first_bar_start, torch_vals, width, label='PyTorch SDPA', color=colors[4])
        bars4 = ax.bar(third_bar_start, ck_vals, width, label='Composable Kernel', color=colors[1])

        # Plot X markers for OOM
        oom_height = 50  # Position X near top of chart
        if torch_oom:
            for idx in torch_oom:
                ax.plot(x[idx] -  2*width, oom_height, 'x', color=colors[4], 
                       markersize=15, markeredgewidth=3)
                ax.text(x[idx] -  2*width, oom_height + max_tflops * 0.03,
                       'OOM', ha='center', va='bottom', fontsize=10, color=colors[4])

        # Add value labels on bars
        for bar, value in zip(bars0, aiter_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        for bar, value in zip(bars1, tk_tflops):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                    f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        if len(triton_vals) > 0:
            for bar, value in zip(bars2, triton_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=14)
        
        if len(torch_vals) > 0:
            for i, (bar, value) in enumerate(zip(bars3, torch_vals)):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=14)
                        
        if len(ck_vals) > 0:
            for bar, value in zip(bars4, ck_vals):
                if value > 0:  # Only label non-OOM bars
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_tflops * 0.01,
                            f'{value:.0f}', ha='center', va='bottom', fontsize=14)

        # add some padding to the top of the y-axis to prevent label overlap
        ax.set_ylim(0, max_tflops * 1.15)
        ax.set_xlabel('Sequence Length (N)', fontsize=16)
        ax.set_ylabel('Performance (TFLOPS)', fontsize=16)
        ax.set_title(f'Attention Performance Comparison {device.upper()}', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(matrix_sizes, fontsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.legend(fontsize=16)

        plt.tight_layout()
        plt.show()

        output_file = f'{device}_{setting}_attn_plot.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_file}")