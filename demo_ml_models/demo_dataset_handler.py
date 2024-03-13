import pandas as pd
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DatasetHandler:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder
        self.dataset_path = None
        self.data = None

    def list_csv_files(self):
        csv_files = [f for f in os.listdir(self.dataset_folder) if f.endswith('.csv')]
        for i, file in enumerate(csv_files):
            print(f"{i + 1}: {file}")
        return csv_files

    def select_csv_file(self, csv_files):
        choice = int(input("Select a CSV file by entering its number: ")) - 1
        selected_file = csv_files[choice]
        self.dataset_path = os.path.join(self.dataset_folder, selected_file)
        print(f"Selected dataset: {selected_file}")

    def load_and_show_sample(self):
        # Load the data, attempting to infer data types more accurately and handle memory more efficiently
        self.data = pd.read_csv(self.dataset_path, low_memory=False)

        # Drop any 'Unnamed' columns that may have been created due to extra comma separators
        self.data = self.data.loc[:, ~self.data.columns.str.contains('^Unnamed')]

        print("Showing a sample of the dataset:")
        print(self.data.sample(20))  # Adjusted to show a sample of 20 as per your request

    def show_label_distribution(self):
        print("Label distribution:")
        self.data['Label'].value_counts().plot.bar(color='skyblue')  # Specify the color here
        plt.show()

    def preprocess_and_split_data(self):
        X_train_dev, X_test, y_train_dev, y_test = train_test_split(self.data.drop('Label', axis=1), self.data['Label'],
                                                                    test_size=0.2, random_state=42)
        X_train, X_dev, y_train, y_dev = train_test_split(X_train_dev, y_train_dev, test_size=0.25, random_state=42)

        print("Label names:", self.data['Label'].unique())
        print("Training label shape:", y_train.shape)
        print("Development label shape:", y_dev.shape)
        print("Test label shape:", y_test.shape)

        # Visualize the split in a pie chart
        sizes = [len(y_train), len(y_dev), len(y_test)]
        labels = ['Training', 'Development', 'Test']
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title('Dataset Split')
        plt.show()

def main():
    dataset_folder = 'demo_datasets'  # Update with the correct path to your datasets folder
    handler = DatasetHandler(dataset_folder)

    csv_files = handler.list_csv_files()
    handler.select_csv_file(csv_files)
    handler.load_and_show_sample()
    handler.show_label_distribution()
    handler.preprocess_and_split_data()

if __name__ == "__main__":
    main()
