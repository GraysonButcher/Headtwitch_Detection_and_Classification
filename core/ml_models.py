"""
Machine Learning models for HTR classification.
Handles training, prediction, and model evaluation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import glob


class HTRClassifier:
    """HTR classifier with XGBoost backend."""
    
    def __init__(self, model_type: str = 'xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.training_history = {}
        
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2,
              hyperparameter_tuning: bool = True, random_state: int = 42) -> Dict[str, Any]:
        """Train the HTR classifier."""
        print(f"Training {self.model_type} classifier...")
        print(f"Training data shape: {X.shape}")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        
        # Train model
        if hyperparameter_tuning:
            print("Performing hyperparameter tuning...")
            self.model, best_params = self._train_with_tuning(X_train, y_train, random_state)
        else:
            print("Training with default parameters...")
            self.model, best_params = self._train_default(X_train, y_train, random_state)
        
        # Evaluate on validation set
        val_results = self.evaluate(X_val, y_val)
        
        # Store training history
        self.training_history = {
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'feature_count': len(self.feature_names),
            'best_params': best_params,
            'validation_results': val_results,
            'training_date': datetime.now().isoformat()
        }
        
        print("Training completed successfully!")
        return self.training_history
    
    def _train_with_tuning(self, X_train: pd.DataFrame, y_train: pd.Series, 
                          random_state: int) -> Tuple[Any, Dict]:
        """Train with hyperparameter tuning."""
        if self.model_type == 'xgboost':
            # Calculate scale_pos_weight for class imbalance
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            base_model = XGBClassifier(
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            )
            
            param_grid = {
                'n_estimators': [100, 300],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.7, 1.0],
                'colsample_bytree': [0.7, 1.0]
            }
        
        else:  # Random Forest
            base_model = RandomForestClassifier(random_state=random_state)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            n_jobs=-1,
            scoring='f1',
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def _train_default(self, X_train: pd.DataFrame, y_train: pd.Series, 
                      random_state: int) -> Tuple[Any, Dict]:
        """Train with default parameters."""
        if self.model_type == 'xgboost':
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            model = XGBClassifier(
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight
            )
        else:
            model = RandomForestClassifier(random_state=random_state)
        
        model.fit(X_train, y_train)
        return model, model.get_params()
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on data."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # For legacy models, try to get feature names from the model itself
        if not self.feature_names and hasattr(self.model, 'feature_names_in_'):
            self.feature_names = list(self.model.feature_names_in_)  # Convert to list
            print(f"  Retrieved {len(self.feature_names)} feature names from model")
            print(f"  Feature names type: {type(self.feature_names)}")
            print(f"  Available columns: {len(X.columns)}")
            print(f"  Columns type: {type(X.columns)}")
        
        # Filter to only feature columns (exclude metadata columns)
        if self.feature_names is not None and len(self.feature_names) > 0:
            # Use only the feature columns that were used in training
            print(f"  Checking for missing columns...")
            try:
                missing_cols = [col for col in self.feature_names if col not in X.columns]
                print(f"  Missing columns check complete: {len(missing_cols)} missing")
                if len(missing_cols) > 0:
                    raise ValueError(f"Missing required feature columns: {missing_cols}")
                print(f"  Creating filtered dataframe...")
                X_filtered = X[list(self.feature_names)]  # Ensure it's a list
                print(f"  Filtered to {len(self.feature_names)} feature columns")
            except Exception as e:
                print(f"  Error during column filtering: {e}")
                print(f"  Feature names: {self.feature_names[:5]}... (showing first 5)")
                print(f"  DataFrame columns: {list(X.columns)[:5]}... (showing first 5)")
                raise
        else:
            # Fallback: exclude known metadata columns
            metadata_cols = ['ground_truth', 'rat_id', 'dose', 'start_frame', 'end_frame', 'prediction', 'prediction_confidence']
            feature_cols = [col for col in X.columns if col not in metadata_cols]
            X_filtered = X[feature_cols]
            print(f"  Using {len(feature_cols)} columns (excluded metadata columns)")
        
        predictions = self.model.predict(X_filtered)
        probabilities = self.model.predict_proba(X_filtered)
        
        return predictions, probabilities
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        predictions, probabilities = self.predict(X)
        
        # Calculate metrics
        report = classification_report(y, predictions, output_dict=True)
        cm = confusion_matrix(y, predictions)
        
        # ROC AUC if we have probabilities for positive class
        try:
            roc_auc = roc_auc_score(y, probabilities[:, 1])
        except (IndexError, ValueError):
            roc_auc = None
        
        results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'accuracy': report['accuracy'],
            'f1_score': report['macro avg']['f1-score'],
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall']
        }
        
        return results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model."""
        try:
            model_data = {
                'model': self.model,
                'model_type': self.model_type,
                'feature_names': self.feature_names,
                'training_history': self.training_history
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            model_data = joblib.load(filepath)
            
            # Handle both dictionary format (new) and raw model format (legacy)
            if isinstance(model_data, dict):
                # New format: dictionary with metadata
                self.model = model_data['model']
                self.model_type = model_data.get('model_type', 'unknown')
                self.feature_names = model_data.get('feature_names', None)
                self.training_history = model_data.get('training_history', {})
            else:
                # Legacy format: raw model object
                self.model = model_data
                self.model_type = 'unknown'
                self.feature_names = None
                self.training_history = {}
                print("Loaded legacy model format - some metadata may be missing")
            
            print(f"Model loaded successfully: type={self.model_type}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


class BatchPredictor:
    """Handles batch prediction on multiple feature files."""
    
    def __init__(self, model: HTRClassifier):
        self.model = model
    
    def predict_folder(self, input_folder: str, output_folder: str, 
                      file_pattern: str = "*.csv") -> Dict[str, Any]:
        """Predict HTRs for all CSV files in a folder."""
        print("Starting batch prediction...")
        
        # Find all CSV files
        search_path = os.path.join(input_folder, file_pattern)
        csv_files = glob.glob(search_path)
        
        if not csv_files:
            return {
                'success': False,
                'error': f'No CSV files found in {input_folder}',
                'files_processed': 0
            }
        
        print(f"Found {len(csv_files)} files to process")
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        processed_files = 0
        failed_files = []
        
        for i, file_path in enumerate(csv_files):
            print(f"Processing {i+1}/{len(csv_files)}: {os.path.basename(file_path)}")
            
            try:
                # Load and predict
                df = pd.read_csv(file_path)
                
                # Ensure required feature columns exist
                if self.model.feature_names:
                    print(f"  Checking feature columns. Expected: {len(self.model.feature_names)}")
                    missing_cols = [col for col in self.model.feature_names if col not in df.columns]
                    if missing_cols:
                        print(f"  Warning: Missing columns {missing_cols}, skipping")
                        failed_files.append((file_path, f"Missing columns: {missing_cols}"))
                        continue
                else:
                    print(f"  No feature names stored in model, using all columns")
                
                print(f"  Input data shape: {df.shape}")
                
                # Make predictions
                print(f"  Making predictions...")
                predictions, probabilities = self.model.predict(df)
                print(f"  Predictions shape: {predictions.shape}, probabilities shape: {probabilities.shape}")
                confidence_scores = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
                
                # Add results to dataframe
                df['prediction'] = predictions.astype(int)
                df['prediction_confidence'] = confidence_scores
                
                # Parse metadata from filename
                metadata = self._parse_filename(file_path)
                for key, value in metadata.items():
                    df[key] = value
                
                # Save results
                output_filename = os.path.basename(file_path).replace('.csv', '_predicted.csv')
                output_path = os.path.join(output_folder, output_filename)
                df.to_csv(output_path, index=False)
                
                processed_files += 1
                
            except Exception as e:
                print(f"  Error: {e}")
                failed_files.append((file_path, str(e)))
        
        results = {
            'success': True,
            'files_processed': processed_files,
            'files_failed': len(failed_files),
            'failed_files': failed_files,
            'output_folder': output_folder
        }
        
        print(f"\nBatch prediction complete: {processed_files} files processed, {len(failed_files)} failed")
        return results
    
    def _parse_filename(self, filepath: str) -> Dict[str, str]:
        """Parse metadata from filename."""
        import re

        filename = os.path.basename(filepath)

        try:
            # Remove common file extensions and suffixes
            base_name = filename.replace('.csv', '').replace('.h5', '').replace('_htr_features', '').replace('_predicted', '')

            # Extract rat ID as the first sequence of digits in the filename
            rat_id_match = re.search(r'(\d+)', base_name)
            if rat_id_match:
                rat_id = rat_id_match.group(1)
            else:
                rat_id = 'unknown'

            # Don't try to parse dose - just set to unknown
            dose = 'unknown'

            return {'rat_id': rat_id, 'dose': dose}

        except Exception:
            return {'rat_id': 'unknown', 'dose': 'unknown'}


class ResultsAggregator:
    """Aggregates prediction results into final reports."""
    
    def __init__(self, fps: float = 120.0):
        self.fps = fps
    
    def compile_results(self, predictions_folder: str, output_excel: str) -> Dict[str, Any]:
        """Compile all prediction results into an Excel report."""
        print("Compiling prediction results...")
        
        # Find all prediction files
        search_path = os.path.join(predictions_folder, '*_predicted.csv')
        prediction_files = glob.glob(search_path)
        
        if not prediction_files:
            return {
                'success': False,
                'error': f'No *_predicted.csv files found in {predictions_folder}'
            }
        
        print(f"Found {len(prediction_files)} prediction files")
        
        # Load and combine all files
        all_dfs = []
        for file_path in prediction_files:
            try:
                df = pd.read_csv(file_path)
                all_dfs.append(df)
            except Exception as e:
                print(f"Warning: Could not read {os.path.basename(file_path)}: {e}")
        
        if not all_dfs:
            return {
                'success': False,
                'error': 'No valid prediction files could be loaded'
            }
        
        # Combine all data
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df['rat_id'] = master_df['rat_id'].astype(str)
        
        print(f"Combined {len(master_df)} total events")
        
        # Filter for positive predictions (HTRs)
        htr_df = master_df[master_df['prediction'] == 1].copy()
        print(f"Identified {len(htr_df)} HTR events")
        
        if htr_df.empty:
            print("No HTRs identified, creating empty report")
            # Create empty Excel with message
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                empty_df = pd.DataFrame({'Message': ['No HTRs were identified in the analyzed data']})
                empty_df.to_excel(writer, sheet_name='Summary', index=False)
            
            return {
                'success': True,
                'total_events': len(master_df),
                'htr_events': 0,
                'output_file': output_excel
            }
        
        # Create detailed and summary sheets
        detailed_sheet = self._create_detailed_sheet(htr_df)
        summary_sheet = self._create_summary_sheet(htr_df)
        
        # Write to Excel
        try:
            with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
                detailed_sheet.to_excel(writer, sheet_name='All Identified HTRs', index=False)
                summary_sheet.to_excel(writer, sheet_name='HTR Summary by Rat')
            
            print(f"Results saved to: {output_excel}")
            
            return {
                'success': True,
                'total_events': len(master_df),
                'htr_events': len(htr_df),
                'unique_rats': len(htr_df['rat_id'].unique()),
                'output_file': output_excel
            }
            
        except ImportError:
            return {
                'success': False,
                'error': 'openpyxl library required for Excel export. Please install with: pip install openpyxl'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error writing Excel file: {e}'
            }
    
    def _create_detailed_sheet(self, htr_df: pd.DataFrame) -> pd.DataFrame:
        """Create detailed sheet with all HTR events."""
        detailed = htr_df.copy()
        
        # Add timestamp column
        detailed['timestamp'] = detailed['start_frame'].apply(
            lambda x: self._frames_to_timestamp(x)
        )
        
        # Select and order columns for the detailed view
        columns = ['rat_id', 'dose', 'start_frame', 'end_frame', 'timestamp', 
                  'duration_frames', 'prediction_confidence']
        
        # Add any additional columns that exist
        for col in detailed.columns:
            if col not in columns and col not in ['prediction']:
                columns.append(col)
        
        return detailed[columns].sort_values(['rat_id', 'start_frame']).reset_index(drop=True)
    
    def _create_summary_sheet(self, htr_df: pd.DataFrame) -> pd.DataFrame:
        """Create wide summary sheet organized by rat."""
        # Add timestamp for summary
        htr_df['timestamp'] = htr_df['start_frame'].apply(self._frames_to_timestamp)
        
        # Get unique rats and sort
        rat_ids = sorted(htr_df['rat_id'].unique())
        
        # Create summary data for each rat
        rat_summaries = []
        column_headers = []
        
        max_events = 0
        for rat in rat_ids:
            rat_data = htr_df[htr_df['rat_id'] == rat].sort_values('start_frame').reset_index(drop=True)
            htr_count = len(rat_data)
            max_events = max(max_events, htr_count)
            
            dose = rat_data['dose'].iloc[0] if not rat_data.empty else 'unknown'
            
            rat_summaries.append({
                'rat_id': rat,
                'dose': dose,
                'htr_count': htr_count,
                'data': rat_data[['start_frame', 'timestamp']].values.tolist()
            })
        
        # Create wide format DataFrame
        summary_data = []
        
        for i in range(max_events):
            row = {}
            for rat_summary in rat_summaries:
                rat = rat_summary['rat_id']
                dose = rat_summary['dose']
                count = rat_summary['htr_count']
                
                col_base = f"{rat} ({dose}) - Count: {count}"
                
                if i < len(rat_summary['data']):
                    frame, timestamp = rat_summary['data'][i]
                    row[f'{col_base}_Frame'] = frame
                    row[f'{col_base}_Timestamp'] = timestamp
                else:
                    row[f'{col_base}_Frame'] = None
                    row[f'{col_base}_Timestamp'] = None
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)
    
    def _frames_to_timestamp(self, frame_number: float) -> str:
        """Convert frame number to timestamp string."""
        if pd.isna(frame_number):
            return ""
        
        total_seconds = int(frame_number / self.fps)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ModelEvaluator:
    """Generates evaluation plots and reports for trained models."""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create and optionally save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Predicted No HTR (0)', 'Predicted HTR (1)'],
                   yticklabels=['Actual No HTR (0)', 'Actual HTR (1)'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    @staticmethod
    def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20,
                              save_path: Optional[str] = None) -> plt.Figure:
        """Create and optionally save feature importance plot."""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=top_features, x='importance', y='feature', palette='viridis', ax=ax)
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


class HTRPredictor:
    """Unified predictor interface for GUI compatibility.
    
    Provides a simple interface that combines HTRClassifier, BatchPredictor, 
    and ResultsAggregator functionality.
    """
    
    def __init__(self):
        self.classifier = None
        self.batch_predictor = None
        self.results_aggregator = ResultsAggregator()
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model from file."""
        try:
            print(f"Loading model from: {model_path}")
            self.classifier = HTRClassifier()
            if self.classifier.load_model(model_path):
                self.batch_predictor = BatchPredictor(self.classifier)
                print(f"HTRPredictor model loaded successfully")
                return True
            else:
                print(f"HTRClassifier.load_model() returned False")
                return False
        except Exception as e:
            print(f"Error in HTRPredictor.load_model: {e}")
            return False
    
    def predict_folder(self, features_folder: str, predictions_folder: str) -> Dict[str, Any]:
        """Predict HTRs for all CSV files in features folder."""
        if not self.batch_predictor:
            return {
                'success': False,
                'error': 'No model loaded. Call load_model() first.',
                'files_processed': 0
            }
        
        # Ensure output folder exists
        os.makedirs(predictions_folder, exist_ok=True)
        
        return self.batch_predictor.predict_folder(features_folder, predictions_folder)
    
    def compile_results(self, predictions_folder: str, output_excel: str) -> Dict[str, Any]:
        """Compile prediction results into Excel report."""
        return self.results_aggregator.compile_results(predictions_folder, output_excel)


class ModelTrainer:
    """Unified training interface for GUI compatibility."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
    
    def train_model(self, input_folder: str, ground_truth_csv: str, model_output_path: str) -> Dict[str, Any]:
        """Train a new HTR model from H5 files and ground truth labels.
        
        Args:
            input_folder: Folder containing H5 files
            ground_truth_csv: CSV file with labeled training data
            model_output_path: Where to save the trained model
            
        Returns:
            Dict with training results: {'success': bool, 'accuracy': float, 'error': str}
        """
        try:
            import pandas as pd
            from sklearn.metrics import accuracy_score
            
            # Load ground truth data
            if not os.path.exists(ground_truth_csv):
                return {
                    'success': False,
                    'error': f'Ground truth file not found: {ground_truth_csv}'
                }
            
            print(f"Loading ground truth data from: {ground_truth_csv}")
            ground_truth_df = pd.read_csv(ground_truth_csv)
            
            # Check required columns
            required_cols = ['ground_truth']
            missing_cols = [col for col in required_cols if col not in ground_truth_df.columns]
            if missing_cols:
                return {
                    'success': False,
                    'error': f'Missing required columns in ground truth CSV: {missing_cols}'
                }
            
            # Split features and labels
            feature_cols = [col for col in ground_truth_df.columns if col not in ['ground_truth', 'rat_id', 'start_frame', 'end_frame']]
            
            if not feature_cols:
                return {
                    'success': False,
                    'error': 'No feature columns found in ground truth CSV'
                }
            
            X = ground_truth_df[feature_cols]
            y = ground_truth_df['ground_truth']
            
            print(f"Training data shape: {X.shape}")
            print(f"Feature columns: {len(feature_cols)}")
            print(f"Class distribution: {y.value_counts().to_dict()}")
            
            # Train classifier
            classifier = HTRClassifier()
            training_results = classifier.train(X, y, hyperparameter_tuning=True)
            
            # Save model
            os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
            if classifier.save_model(model_output_path):
                print(f"Model saved to: {model_output_path}")
                
                return {
                    'success': True,
                    'accuracy': training_results.get('accuracy', 0.0),
                    'model_path': model_output_path,
                    'training_details': training_results
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to save trained model'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f'Training failed: {str(e)}'
            }