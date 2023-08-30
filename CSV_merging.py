import os
import pandas as pd

def merge_feature_csv(root_folder):
    combined_data = pd.DataFrame()

    for root, dirs, files in os.walk(root_folder):
        for dir in dirs:
            if dir.startswith("ALLE "):
                dir_path = os.path.join(root, dir)
                for file in os.listdir(dir_path):
                    if file.endswith("_features.csv"):
                        csv_file = os.path.join(dir_path, file)
                        df = pd.read_csv(csv_file)
                        combined_data = pd.concat([combined_data, df], ignore_index=True)

    output_csv = os.path.join(root_folder, "dataset.csv")
    combined_data.to_csv(output_csv, index=False)
    print(f"Merged dataset created: {output_csv}")

if __name__ == "__main__":
    root_folder = os.path.dirname(os.path.abspath(__file__))  # Pfad zum Verzeichnis des Python-Skripts
    merge_feature_csv(root_folder)
