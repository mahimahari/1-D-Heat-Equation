import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_normalized_losses(csv_path='results/loss_history.csv', save_path=None):
    """
    Plot normalized loss components from training history including Initial Condition Loss.
    
    Args:
        csv_path: Path to loss_history.csv file
        save_path: Optional path to save the plot (if None, just shows plot)
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Normalize each loss component to [0,1] range
    df_normalized = df.copy()
    for col in [ 'pde', 'bc', 'ic']:
        if col in df.columns:
            df_normalized[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each component
    #plt.plot(df_normalized['total'], label='Total Loss', linewidth=2)
    plt.plot(df_normalized['pde'], label='PDE Loss', linestyle='--')
    plt.plot(df_normalized['bc'], label='BC Loss', linestyle=':')
    plt.plot(df_normalized['ic'], label='IC Loss', linestyle='-.')  # Added line
    
    # Set y-axis scale and labels
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.ylim(0, 1)
    plt.ylabel("Normalized Loss Value")
    
    # Set x-axis
    plt.xlabel("Epoch")
    plt.xlim(0, len(df)-1)
    
    # Add title and legend
    plt.title("Normalized Training Loss Components (PDE, BC, IC)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Save or show the plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    # Example usage:
    plot_normalized_losses(
        csv_path='results/loss_history.csv',
        save_path='results/normalized_loss_plot.png'  # Set to None to just show plot
    )
