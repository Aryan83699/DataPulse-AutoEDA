import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta
import json


def read_dataset(file_path):
    """Smart file reader that detects actual format"""
    # Try excel first if extension says so
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        try:
            return pd.read_excel(file_path)
        except Exception:
            pass
    
    # Try CSV with different encodings
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    
    # Last resort
    return pd.read_csv(file_path, encoding='utf-8', errors='replace')






from pycaret.classification import (
    setup as clf_setup, 
    compare_models as clf_compare, 
    tune_model as clf_tune,
    get_config as clf_get_config, 
    pull as clf_pull, 
    save_model as clf_save, 
    predict_model as clf_predict, 
    plot_model as clf_plot
)
from pycaret.regression import (
    setup as reg_setup, 
    compare_models as reg_compare,
    tune_model as reg_tune, 
    get_config as reg_get_config, 
    pull as reg_pull, 
    save_model as reg_save, 
    predict_model as reg_predict, 
    plot_model as reg_plot
)
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def detect_problem_type(df, target):
    """
    Detect if it's classification or regression
    """
    unique_values = df[target].nunique()
    total_values = len(df[target])
    
    if df[target].dtype == 'object':
        return "Classification"
    
    if unique_values < 10 and unique_values / total_values < 0.05:
        return "Classification"
    
    return "Regression"

def detect_datetime_column(df):
    """
    Detect if dataset has datetime column suitable for time series
    Returns: (has_datetime, datetime_column_name)
    """
    for col in df.columns:
        try:
            # Try to convert to datetime
            pd.to_datetime(df[col])
            # Check if it's actually datetime-like
            if df[col].dtype == 'object' or 'date' in col.lower() or 'time' in col.lower():
                return True, col
        except:
            continue
    return False, None


def perform_time_series_forecast(dataset_path, target, date_column, forecast_periods=30):
    """
    Perform time series forecasting using Prophet
    
    Parameters:
    - dataset_path: Path to CSV file
    - target: Target variable to forecast
    - date_column: Name of the datetime column
    - forecast_periods: Number of periods to forecast (default 30)
    
    Returns:
    - forecast_df: DataFrame with predictions
    - forecast_plot: Path to forecast plot
    - model: Trained Prophet model
    """
    try:
        # Load dataset
        df=read_dataset(dataset_path)
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = pd.to_datetime(df[date_column])
        prophet_df['y'] = df[target]
        
        # Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        model.fit(prophet_df)
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=forecast_periods)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Generate forecast plot
        plot_path = generate_forecast_plot(model, forecast, prophet_df)
        
        # Return forecast results
        forecast_results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_periods)
        
        return forecast_results, plot_path, model
        
    except Exception as e:
        print(f"Error in time series forecasting: {e}")
        return None, None, None




def generate_forecast_plot(model, forecast, actual_data, save_path="static/plots"):
    """
    Generate time series forecast visualization
    """
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot actual data
        plt.plot(actual_data['ds'], actual_data['y'], 
                'ko-', markersize=3, label='Actual Data', linewidth=1)
        
        # Plot forecast
        plt.plot(forecast['ds'], forecast['yhat'], 
                'b-', label='Forecast', linewidth=2)
        
        # Plot confidence interval
        plt.fill_between(forecast['ds'], 
                        forecast['yhat_lower'], 
                        forecast['yhat_upper'],
                        alpha=0.2, color='blue', label='Confidence Interval')
        
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Value', fontsize=11)
        plt.title('Time Series Forecast', fontsize=14, fontweight='bold')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, 'forecast_plot.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return 'forecast_plot.png'
        
    except Exception as e:
        print(f"Error generating forecast plot: {e}")
        return None





def generate_confusion_matrix(model, test_data, target, save_path="static/plots"):
    """
    Generate confusion matrix plot for classification (MEDIUM SIZE)
    """
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Make predictions
        predictions = clf_predict(model, data=test_data)
        
        # Get actual and predicted values
        y_true = predictions[target]
        y_pred = predictions['prediction_label']
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot with MEDIUM size (reduced from 10x8 to 6x5)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=sorted(y_true.unique()),
                    yticklabels=sorted(y_true.unique()),
                    cbar_kws={'shrink': 0.8})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=11)
        plt.xlabel('Predicted', fontsize=11)
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return 'confusion_matrix.png'
    
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")
        return None


def generate_actual_vs_predicted(model, test_data, target, save_path="static/plots"):
    """
    Generate actual vs predicted plot for regression (MEDIUM SIZE)
    """
    os.makedirs(save_path, exist_ok=True)
    
    try:
        # Make predictions
        predictions = reg_predict(model, data=test_data)
        
        # Get actual and predicted values
        y_true = predictions[target]
        y_pred = predictions['prediction_label']
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Plot with MEDIUM size (reduced from 10x8 to 6x5)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=40)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Values', fontsize=11)
        plt.ylabel('Predicted Values', fontsize=11)
        plt.title('Actual vs Predicted Values', fontsize=14, fontweight='bold')
        
        # Add metrics text
        textstr = f'R² Score: {r2:.4f}\nRMSE: {rmse:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, 
                fontsize=9, verticalalignment='top', bbox=props)
        
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(save_path, 'actual_vs_predicted.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()
        
        return 'actual_vs_predicted.png'
    
    except Exception as e:
        print(f"Error generating actual vs predicted plot: {e}")
        return None
    

def generate_prediction_distribution(model, test_data, target, problem_type, save_path="static/plots"):
    """
    Generate prediction distribution plot — works for both classification and regression
    """
    os.makedirs(save_path, exist_ok=True)

    try:
        if problem_type == "Classification":
            predictions = clf_predict(model, data=test_data)
        else:
            predictions = reg_predict(model, data=test_data)

        y_true = predictions[target]
        y_pred = predictions['prediction_label']

        plt.figure(figsize=(6, 5))

        if problem_type == "Classification":
            # Side by side bar chart of actual vs predicted class distribution
            import collections
            actual_counts = collections.Counter(y_true)
            pred_counts   = collections.Counter(y_pred)
            classes = sorted(set(list(actual_counts.keys()) + list(pred_counts.keys())))
            x = np.arange(len(classes))
            width = 0.35
            plt.bar(x - width/2, [actual_counts.get(c, 0) for c in classes],
                    width, label='Actual', color='steelblue', alpha=0.8)
            plt.bar(x + width/2, [pred_counts.get(c, 0) for c in classes],
                    width, label='Predicted', color='coral', alpha=0.8)
            plt.xticks(x, [str(c) for c in classes], rotation=45)
            plt.xlabel('Class', fontsize=11)
            plt.ylabel('Count', fontsize=11)
            plt.title('Actual vs Predicted Class Distribution', fontsize=13, fontweight='bold')
            plt.legend(fontsize=9)

        else:
            # Residual distribution plot
            residuals = y_true - y_pred
            plt.hist(residuals, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
            plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
            plt.xlabel('Residual (Actual - Predicted)', fontsize=11)
            plt.ylabel('Frequency', fontsize=11)
            plt.title('Residual Distribution', fontsize=13, fontweight='bold')
            plt.legend(fontsize=9)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(save_path, 'prediction_distribution.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=100)
        plt.close()

        return 'prediction_distribution.png'

    except Exception as e:
        print(f"Error generating distribution plot: {e}")
        return None




def run_automl_pipeline(dataset_path, target, 
                       train_size=0.8, 
                       normalize=False, 
                       feature_selection=False,
                       remove_outliers=False,
                       tune_best_model=False,
                       prediction_type='normal',  # NEW PARAMETER
                       date_column=None,          # NEW PARAMETER
                       forecast_periods=30):      # NEW PARAMETER
    """
    Enhanced AutoML pipeline with tunable parameters
    
    Parameters:
    - dataset_path: Path to CSV file
    - target: Target column name
    - train_size: Train/test split ratio (0.0 to 1.0)
    - normalize: Whether to normalize features
    - feature_selection: Whether to perform feature selection
    - remove_outliers: Whether to remove outliers
    - tune_best_model: Whether to tune hyperparameters of best model
    - prediction_type: 'normal' or 'timeseries'
    - date_column: Column containing dates (for time series)
    - forecast_periods: Number of periods to forecast
    """
    
    # Load dataset
    df=read_dataset(dataset_path)
    
    # ✅ CHECK PREDICTION TYPE FIRST - BEFORE DETECTING PROBLEM TYPE
    if prediction_type == 'timeseries':
        print("User selected: Time Series Forecasting")
        
        if date_column is None or date_column == '':
            # Auto-detect datetime column
            has_datetime, date_column = detect_datetime_column(df)
            if not has_datetime:
                raise ValueError("No datetime column found for time series forecasting")
        
        # Perform time series forecasting
        forecast_results, forecast_plot, ts_model = perform_time_series_forecast(
            dataset_path, target, date_column, forecast_periods
        )
        
        # Return time series results (different format)
        return (
            forecast_results,  # Forecast dataframe instead of model comparison
            ts_model,          # Prophet model
            "Time Series",     # Problem type
            None,              # No feature importance for time series
            None,              # No confusion matrix
            None,              # No actual vs predicted
            forecast_plot  ,
            None    # Forecast plot instead of model filename
        )
    
    # ✅ ONLY DETECT PROBLEM TYPE IF NOT TIME SERIES
    problem_type = detect_problem_type(df, target)
    print(f"Detected Problem Type: {problem_type}")

    # Setup based on problem type
    if problem_type == "Classification":
        
        # EXCLUDE LIGHTGBM FOR CLASSIFICATION TOO
        exclude_models = ['lightgbm']
        
# Feature selection via PyCaret uses LightGBM internally
# Use manual feature selection instead to avoid infinite loop
        if feature_selection:
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            X = df.drop(columns=[target])
            y = df[target]
            # Only use numeric columns
            X = X.select_dtypes(include=[np.number])
            rf.fit(X, y)
            importances = rf.feature_importances_
            # Keep top 70% most important features
            threshold = np.percentile(importances, 30)
            selected_features = X.columns[importances >= threshold].tolist()
            selected_features.append(target)
            df = df[selected_features]
            print(f"Selected features: {selected_features}")

        clf_setup(
            data=df,
            target=target,
            train_size=train_size,
            normalize=normalize,
            feature_selection=False,  # Always False — we handle it manually above
            remove_outliers=remove_outliers,
            session_id=42,
            verbose=False,
            html=False
        )
        
        # Compare models (EXCLUDE LIGHTGBM)
        try:
            best_model = clf_compare(
                n_select=1, 
                exclude=exclude_models,
                verbose=False
            )
            results = clf_pull()
        except Exception as e:
            print(f"Error during model comparison: {e}")
            # Fallback to logistic regression
            from sklearn.linear_model import LogisticRegression
            best_model = LogisticRegression(max_iter=1000)
            X_train = clf_get_config('X_train')
            y_train = clf_get_config('y_train')
            best_model.fit(X_train, y_train)
            
            # Create dummy results
            results = pd.DataFrame({
                'Model': ['Logistic Regression (Fallback)'],
                'Accuracy': [0.0],
                'AUC': [0.0],
                'Recall': [0.0],
                'Prec.': [0.0],
                'F1': [0.0],
                'Kappa': [0.0],
                'MCC': [0.0]
            })
        
        # Tune best model if requested
        if tune_best_model:
            print("Tuning best model hyperparameters...")
            try:
                best_model = clf_tune(best_model, optimize='Accuracy', verbose=False)
            except Exception as e:
                print(f"Tuning failed, using base model: {e}")
        
        # Get test data
        test_data = clf_get_config('X_test').copy()
        test_data[target] = clf_get_config('y_test').copy()
        
        # Generate confusion matrix
        confusion_plot = generate_confusion_matrix(best_model, test_data, target)
        actual_vs_pred_plot = None
        distribution_plot = generate_prediction_distribution(best_model, test_data, target, "Classification")
        
    else:  # Regression
        
        # EXCLUDE PROBLEMATIC MODELS TO AVOID INFINITE LOOP
        exclude_models = ['lightgbm']
        
        if feature_selection:
            from sklearn.ensemble import RandomForestRegressor
            
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            X = df.drop(columns=[target])
            y = df[target]
            X = X.select_dtypes(include=[np.number])
            rf.fit(X, y)
            importances = rf.feature_importances_
            threshold = np.percentile(importances, 30)
            selected_features = X.columns[importances >= threshold].tolist()
            selected_features.append(target)
            df = df[selected_features]
            print(f"Selected features: {selected_features}")

        reg_setup(
            data=df,
            target=target,
            train_size=train_size,
            normalize=normalize,
            feature_selection=False,  # Always False — handled manually above
            remove_outliers=remove_outliers,
            session_id=42,
            verbose=False,
            html=False
        )
        
        # Compare models (exclude problematic ones)
        try:
            best_model = reg_compare(
                n_select=1, 
                exclude=exclude_models,
                verbose=False
            )
            results = reg_pull()
        except Exception as e:
            print(f"Error during model comparison: {e}")
            # Fallback to linear regression
            from sklearn.linear_model import LinearRegression
            best_model = LinearRegression()
            X_train = reg_get_config('X_train')
            y_train = reg_get_config('y_train')
            best_model.fit(X_train, y_train)
            
            # Create dummy results
            results = pd.DataFrame({
                'Model': ['Linear Regression (Fallback)'],
                'MAE': [0.0],
                'MSE': [0.0],
                'RMSE': [0.0],
                'R2': [0.0],
                'RMSLE': [0.0],
                'MAPE': [0.0]
            })
        
        # Tune best model if requested
        if tune_best_model:
            print("Tuning best model hyperparameters...")
            try:
                best_model = reg_tune(best_model, optimize='R2', verbose=False)
            except Exception as e:
                print(f"Tuning failed, using base model: {e}")
        
        # Get test data
        test_data = reg_get_config('X_test').copy()
        test_data[target] = reg_get_config('y_test').copy()
        
        # Generate actual vs predicted plot
        actual_vs_pred_plot = generate_actual_vs_predicted(best_model, test_data, target)
        confusion_plot = None
        distribution_plot = generate_prediction_distribution(best_model, test_data, target, "Regression")
    
    # Extract feature importance
# Extract feature importance — unwrap PyCaret pipeline to get actual estimator
    feature_importance = None
    try:
        # Unwrap PyCaret pipeline
        actual_model = best_model
        if hasattr(best_model, 'steps'):
            # It's a sklearn Pipeline — get the last step
            actual_model = best_model.steps[-1][1]
        elif hasattr(best_model, 'named_steps'):
            actual_model = list(best_model.named_steps.values())[-1]

        feature_names = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()

        if hasattr(actual_model, 'feature_importances_'):
            importances = actual_model.feature_importances_
            # Match length in case feature selection reduced columns
            if len(importances) == len(feature_names):
                feature_importance = dict(zip(feature_names, importances))
            else:
                feature_importance = {f"feature_{i}": v for i, v in enumerate(importances)}
            feature_importance = dict(sorted(feature_importance.items(),
                                            key=lambda x: x[1], reverse=True)[:10])

        elif hasattr(actual_model, 'coef_'):
            coef = actual_model.coef_
            coefficients = np.abs(coef).flatten()[:len(feature_names)]
            feature_importance = dict(zip(feature_names[:len(coefficients)], coefficients))
            feature_importance = dict(sorted(feature_importance.items(),
                                            key=lambda x: x[1], reverse=True)[:10])



        # Fallback — permutation importance (works for ANY model)
        if not feature_importance:
            try:
                from sklearn.inspection import permutation_importance
                from sklearn.pipeline import Pipeline

                X_test = clf_get_config('X_test') if problem_type == "Classification" else reg_get_config('X_test')
                y_test = clf_get_config('y_test') if problem_type == "Classification" else reg_get_config('y_test')

                # Use the pipeline directly
                perm = permutation_importance(
                    best_model, X_test, y_test,
                    n_repeats=5,
                    random_state=42,
                    scoring='accuracy' if problem_type == "Classification" else 'r2'
                )

                feature_names = X_test.columns.tolist()
                importance_dict = dict(zip(feature_names, perm.importances_mean))
                feature_importance = dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10])

                print(f"✅ Permutation importance extracted: {list(feature_importance.keys())[:3]}")

            except Exception as e:
                print(f"⚠️ Permutation importance also failed: {e}")

        if feature_importance:
            print(f"✅ Feature importance extracted: {list(feature_importance.keys())[:3]}")
        else:
            print("⚠️ Model does not expose feature importance")

    except Exception as e:
        print(f"Could not extract feature importance: {e}")

    
    # Save the best model
    model_save_path = "static/models"
    os.makedirs(model_save_path, exist_ok=True)
    
    model_filename = f"best_model_{problem_type.lower()}"
    
    try:
        if problem_type == "Classification":
            clf_save(best_model, os.path.join(model_save_path, model_filename))
        else:
            reg_save(best_model, os.path.join(model_save_path, model_filename))
    except Exception as e:
        print(f"Error saving model: {e}")
        # Save using pickle as fallback
        import pickle
        with open(os.path.join(model_save_path, model_filename + '.pkl'), 'wb') as f:
            pickle.dump(best_model, f)
    
    return (
        results, 
        best_model, 
        problem_type, 
        feature_importance,
        confusion_plot,
        actual_vs_pred_plot,
        model_filename + '.pkl',
        distribution_plot
    )

def make_single_prediction(model_path, problem_type, input_data):
    """
    Make prediction on single input
    
    Parameters:
    - model_path: Path to saved model
    - problem_type: 'Classification' or 'Regression'
    - input_data: Dictionary of feature values
    
    Returns:
    - prediction result
    """
    from pycaret.classification import load_model as clf_load
    from pycaret.regression import load_model as reg_load
    
    # Load model
    if problem_type == "Classification":
        model = clf_load(model_path.replace('.pkl', ''))
        df_input = pd.DataFrame([input_data])
        prediction = clf_predict(model, data=df_input)
    else:
        model = reg_load(model_path.replace('.pkl', ''))
        df_input = pd.DataFrame([input_data])
        prediction = reg_predict(model, data=df_input)
    
    return prediction['prediction_label'].values[0]