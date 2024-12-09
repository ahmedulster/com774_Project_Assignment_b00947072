#!/opt/anaconda3/bin/python3

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def explore_activity_data(data_path):
    """
    Comprehensive data exploration for Human Activity Recognition dataset
    Parameters:
        data_path: Path to the dataset CSV file
    """
    print("Starting Data Exploration...")
    
    # Step 1: Load and Initial Data Inspection
    #-----------------------------------------
    print("\n=== Step 1: Loading and Initial Inspection ===")
    data = pd.read_csv(data_path)
    print(f"Dataset loaded with shape: {data.shape}")
    print("\nFirst few rows of the dataset:")
    print(data.head())
    
    # Step 2: Data Quality Checks
    #---------------------------
    print("\n=== Step 2: Data Quality Analysis ===")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    print("\nMissing Values Summary:")
    print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")
    
    # Check for duplicates
    duplicates = data.duplicated().sum()
    print(f"\nNumber of duplicate rows: {duplicates}")
    
    # Basic information about the dataset
    print("\nDataset Information:")
    data.info()
    
    # Step 3: Statistical Analysis
    #----------------------------
    print("\n=== Step 3: Statistical Analysis ===")
    
    # Summary statistics for numerical columns
    print("\nSummary Statistics:")
    print(data.describe())
    
    # Step 4: Activity Distribution Analysis (if applicable)
    #----------------------------------------------------
    if 'Activity' in data.columns:
        print("\n=== Step 4: Activity Distribution ===")
        activity_distribution = data['Activity'].value_counts()
        print("\nClass Distribution:")
        print(activity_distribution)
        
        # Visualize activity distribution
        plt.figure(figsize=(12, 6))
        sns.countplot(x='Activity', data=data)
        plt.title("Distribution of Activities")
        plt.xlabel("Activity Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    # Step 5: Feature Analysis
    #------------------------
    print("\n=== Step 5: Feature Analysis ===")
    
    # Select numerical columns for correlation analysis
    numeric_data = data.select_dtypes(include=['number'])
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_data.corr(), cmap="coolwarm", annot=False, cbar=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    
    # Step 6: Feature Distribution
    #-----------------------------
    print("\n=== Step 6: Feature Distributions ===")
    
    # Plot distributions for first few numerical features
    num_features = min(5, len(numeric_data.columns))
    plt.figure(figsize=(15, 3*num_features))
    for i, column in enumerate(numeric_data.columns[:num_features], 1):
        plt.subplot(num_features, 1, i)
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.show()
    
    return data

# Main execution
if __name__ == "__main__":
    # Define the dataset path
    data_path = "/Users/ayesha/Desktop/Mini_Project/data/dataset.csv"
    
    # Run the exploration
    data = explore_activity_data(data_path)