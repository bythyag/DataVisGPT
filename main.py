import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
import json

# Optional: Advanced AI Integration
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class DataInsightsAgent:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Data Insights Agent
        
        :param api_key: Optional OpenAI API key for advanced insights
        """
        self.api_key = api_key
        if api_key and OpenAI:
            self.client = OpenAI(api_key=api_key)
        
        # Logging and configuration
        self.logs = []
        self.config = {
            'outlier_threshold': 3,
            'missing_value_threshold': 0.2
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats
        
        :param file_path: Path to the data file
        :return: Loaded DataFrame
        """
        try:
            # Support multiple file formats
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            self.log(f"Successfully loaded data from {file_path}")
            return df
        except Exception as e:
            self.log(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive data preprocessing
        
        :param df: Input DataFrame
        :return: Preprocessed DataFrame
        """
        if df.empty:
            return df
        
        # Step 1: Handle missing values
        missing_report = self._analyze_missing_values(df)
        df = self._handle_missing_values(df, missing_report)
        
        # Step 2: Remove outliers
        df = self._remove_outliers(df)
        
        # Step 3: Data type conversion
        df = self._convert_data_types(df)
        
        # Optional: AI-driven preprocessing if API key is available
        if self.api_key and OpenAI:
            df = self._ai_enhanced_preprocessing(df)
        
        return df
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze missing values in the DataFrame
        
        :param df: Input DataFrame
        :return: Missing value report
        """
        missing_report = df.isnull().mean()
        self.log("Missing Value Analysis:")
        for col, pct in missing_report.items():
            if pct > 0:
                self.log(f"{col}: {pct*100:.2f}% missing")
        
        return missing_report.to_dict()
    
    def _handle_missing_values(self, df: pd.DataFrame, missing_report: Dict[str, float]) -> pd.DataFrame:
        """
        Strategically handle missing values
        
        :param df: Input DataFrame
        :param missing_report: Missing value percentages
        :return: DataFrame with handled missing values
        """
        for col, pct in missing_report.items():
            if pct > self.config['missing_value_threshold']:
                # Drop column if too many missing values
                df = df.drop(columns=[col])
                self.log(f"Dropped column {col} due to high missing value percentage")
            else:
                # Fill missing values based on column type
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0])
        
        return df.dropna()
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers using Z-score method
        
        :param df: Input DataFrame
        :return: DataFrame with outliers removed
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < self.config['outlier_threshold']]
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert data types for analysis
        
        :param df: Input DataFrame
        :return: DataFrame with converted types
        """
        # Attempt to convert object columns to numeric if possible
        for col in df.select_dtypes(include=['object']):
            try:
                df[col] = pd.to_numeric(df[col], errors='raise')
            except:
                try:
                    df[col] = pd.to_datetime(df[col], errors='raise')
                except:
                    pass  # Keep as is if conversion fails
        
        return df
    
    def _ai_enhanced_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optional AI-driven preprocessing using OpenAI
        
        :param df: Input DataFrame
        :return: Enhanced DataFrame
        """
        if not OpenAI:
            return df
        
        try:
            # Generate preprocessing instructions
            prompt = f"""
            Analyze this dataset description and provide advanced preprocessing recommendations:
            
            Columns: {list(df.columns)}
            Column Types: {df.dtypes.to_dict()}
            Total Rows: {len(df)}
            
            Suggest:
            1. Feature engineering ideas
            2. Potential column transformations
            3. Normalization or scaling recommendations
            
            Respond in strict JSON format.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data preprocessing expert."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Placeholder for AI-driven enhancements
            return df
        
        except Exception as e:
            self.log(f"AI preprocessing failed: {e}")
            return df
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data insights
        
        :param df: Preprocessed DataFrame
        :return: Insights dictionary
        """
        insights = {
            'summary_statistics': df.describe().to_dict(),
            'correlations': self._compute_correlations(df),
            'categorical_insights': self._analyze_categorical_columns(df)
        }
        
        return insights
    
    def _compute_correlations(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Compute correlation matrix
        
        :param df: Input DataFrame
        :return: Correlation insights
        """
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        correlations = df[numeric_cols].corr().to_dict()
        
        return correlations
    
    def _analyze_categorical_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze categorical columns
        
        :param df: Input DataFrame
        :return: Categorical column insights
        """
        categorical_insights = {}
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            categorical_insights[col] = {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts(normalize=True).head().to_dict()
            }
        
        return categorical_insights
    
    def visualize_data(self, df: pd.DataFrame, output_dir: str = 'visualizations') -> Dict[str, plt.Figure]:
        """
        Generate and save comprehensive visualizations
        
        :param df: Preprocessed DataFrame
        :param output_dir: Directory to save visualizations
        :return: Dictionary of visualizations
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        visualizations = {}
        
        # Numeric column distributions
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}')
            # Save the plot
            fig_path = os.path.join(output_dir, f'{col}_distribution.png')
            fig.savefig(fig_path, bbox_inches='tight', dpi=300)
            visualizations[f'{col}_distribution'] = fig
            plt.close(fig)
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('Correlation Heatmap')
            # Save the plot
            fig_path = os.path.join(output_dir, 'correlation_heatmap.png')
            fig.savefig(fig_path, bbox_inches='tight', dpi=300)
            visualizations['correlation_heatmap'] = fig
            plt.close(fig)
        
        self.log(f"Visualizations saved to {output_dir}/")
        return visualizations
    
    def log(self, message: str):
        """
        Log messages for tracking
        
        :param message: Log message
        """
        print(message)
        self.logs.append(message)
    
    def generate_report(self, insights: Dict[str, Any]) -> str:
        """
        Generate a comprehensive report
        
        :param insights: Data insights
        :return: Formatted report
        """
        report = "# Data Insights Report\n\n"
        
        report += "## Summary Statistics\n"
        for stat, values in insights['summary_statistics'].items():
            report += f"### {stat.capitalize()}\n"
            for col, val in values.items():
                report += f"- {col}: {val:.2f}\n"
        
        report += "\n## Correlations\n"
        for col, correlations in insights['correlations'].items():
            report += f"### {col} Correlations\n"
            for corr_col, corr_val in correlations.items():
                report += f"- {corr_col}: {corr_val:.2f}\n"
        
        return report

def main():
    # Configuration
    CSV_PATH = 'filepath.csv'
    OPENAI_API_KEY = 'open_ai_key'
    OUTPUT_DIR = 'visualizations'
    
    # Initialize Agent
    agent = DataInsightsAgent(api_key=OPENAI_API_KEY)
    
    try:
        # Workflow
        print("Loading Data...")
        df = agent.load_data(CSV_PATH)
        
        print("Preprocessing Data...")
        preprocessed_df = agent.preprocess_data(df)
        
        print("Analyzing Data...")
        insights = agent.analyze_data(preprocessed_df)
        
        print("Generating and Saving Visualizations...")
        visualizations = agent.visualize_data(preprocessed_df, output_dir=OUTPUT_DIR)
        
        print("Generating Report...")
        report = agent.generate_report(insights)
        
        # Save report
        with open('data_insights_report.md', 'w') as f:
            f.write(report)
        
        print("Process Complete! Check data_insights_report.md and the visualizations folder")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
