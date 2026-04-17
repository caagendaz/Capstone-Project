import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

LEGACY_DIR = Path("results/figures/legacy")
LEGACY_DIR.mkdir(parents=True, exist_ok=True)

def create_table_figure(df, title, filename, col_widths=None):
    """
    Creates a table figure and saves it as a PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 4))  # Adjust size as needed
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)  # Scale width and height
    
    # Bold the headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif row > 0:
             if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
             else:
                cell.set_facecolor('white')

    # Specific highlighting for XGBoost AUC in Table 1 if present
    if "AUC" in df.columns:
        # Find the cell with the highest AUC and make it bold/highlighted
        # Assuming XGBoost is the last row (index 2 in 0-based data) and AUC is the last column
        # In this specific case I know the location
        pass

    if col_widths:
        for i, width in enumerate(col_widths):
            for row in range(len(df) + 1):
                table[row, i].set_width(width)


    plt.title(title, fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved {filename}")
    plt.close()

def main():
    # Table 1: Comparative Model Performance
    data1 = {
        'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
        'Accuracy': ['76.3%', '76.3%', '76.9%'],
        'Precision': ['68.2%', '68.2%', '71.4%'],
        'Recall': ['33.3%', '33.3%', '33.3%'],
        'F1 Score': ['0.448', '0.448', '0.455'],
        'AUC': ['0.657', '0.668', '0.672']
    }
    df1 = pd.DataFrame(data1)
    create_table_figure(df1, 'Table 1: Comparative Model Performance', 
                        str(LEGACY_DIR / 'table1_performance.png'))

    # Table 2: Performance at Optimized Decision Threshold (0.45)
    data2 = {
        'Metric': ['Recall', 'Specificity', 'Precision'],
        'Value': ['100%', '~26%', '31.0%'],
        'Implication': [
            'Zero resistant cases missed.',
            'High rate of false positives (susceptible flagged as resistant).',
            'Positive predictions require confirmatory testing.'
        ]
    }
    df2 = pd.DataFrame(data2)
    # Adjust column widths for Table 2 because "Implication" text is long
    create_table_figure(df2, 'Table 2: Performance at Optimized Decision Threshold (0.45)', 
                        str(LEGACY_DIR / 'table2_threshold.png'),
                        col_widths=[0.15, 0.15, 0.7])

if __name__ == "__main__":
    main()
