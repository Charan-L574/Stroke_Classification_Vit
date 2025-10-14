import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_and_save_charts(history_csv_path, save_dir='.', chart_prefix='training_charts'):
    """
    Reads training history from a CSV file and plots accuracy and loss charts.

    Args:
        history_csv_path (str): Path to the training history CSV file.
        save_dir (str): Directory where the charts will be saved.
        chart_prefix (str): Prefix for the saved chart filenames.
    """
    if not os.path.exists(history_csv_path):
        print(f"Error: History file not found at {history_csv_path}")
        return

    history_df = pd.read_csv(history_csv_path)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['train_acc'], label='Train Accuracy')
    plt.plot(history_df['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    accuracy_chart_path = os.path.join(save_dir, f"{chart_prefix}_accuracy.png")
    plt.savefig(accuracy_chart_path)
    plt.close()
    print(f"Accuracy chart saved to {accuracy_chart_path}")

    # --- Plot Loss ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_df['train_loss'], label='Train Loss')
    plt.plot(history_df['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    loss_chart_path = os.path.join(save_dir, f"{chart_prefix}_loss.png")
    plt.savefig(loss_chart_path)
    plt.close()
    print(f"Loss chart saved to {loss_chart_path}")

if __name__ == '__main__':
    # This part allows running the script directly if needed,
    # for example, by providing the path to the history file.
    import argparse
    parser = argparse.ArgumentParser(description="Plot training charts from a history CSV file.")
    parser.add_argument("history_file", type=str, help="Path to the training history CSV file.")
    parser.add_argument("--save_dir", type=str, default=".", help="Directory to save the charts.")
    parser.add_argument("--prefix", type=str, default="training_charts", help="Prefix for chart filenames.")
    
    args = parser.parse_args()
    
    plot_and_save_charts(args.history_file, args.save_dir, args.prefix)
