import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

try:
    from dataset_gen.core.analyzers import TrendAnalyzer
except ImportError:
    # Fallback or local definition if imports fail in standalone script
    TrendAnalyzer = None

class StandardEvaluator:
    """
    The Single Source of Truth for TrendQA Evaluation.
    Handles:
    1. Result Formatting (ensure consistent schema between RL and Baselines)
    2. Metric Calculation (RMSE, MAE, Trend Accuracy)
    """

    @staticmethod
    def build_result_sample(original_sample: Dict[str, Any], 
                          filled_df: pd.DataFrame, 
                          method_name: str) -> Dict[str, Any]:
        """
        Constructs the standardized result dictionary.
        This should be called by both the RL Runner and Baseline scripts.
        """
        # 1. Create a shallow copy to preserve original metadata
        result = original_sample.copy()
        
        # 2. Attach the filled dataframe
        result['filled_sub_table_df'] = filled_df
        result['method_name'] = method_name
        
        # 3. Extract specific predictions for efficient metric calc
        predictions = []
        targets = original_sample.get('imputation_targets', [])
        
        for target in targets:
            row_idx = target.get('row_index')
            col_name = target.get('col_name')
            gt_val = target.get('gt')
            
            try:
                # Extract prediction from the filled dataframe
                pred_val = filled_df.iloc[row_idx][col_name]
                pred_val = float(pred_val) # Ensure numeric
            except (ValueError, TypeError, KeyError, IndexError):
                pred_val = np.nan
                
            predictions.append({
                'row_index': row_idx,
                'col_name': col_name,
                'gt': gt_val,
                'pred': pred_val
            })
            
        result['imputation_predictions'] = predictions
        
        # 4. Calculate Derived Trend (Downstream Task)
        # We run the analyzer on the *filled* table to see if the trend is correct.
        if TrendAnalyzer:
            # Assuming the target column for trend analysis is usually the one we filled
            # If multiple columns, we might need heuristics. 
            # For now, we try to infer the main numeric column or use metadata if available.
            
            # Determine Invert Logic: Only ranking (where smaller is better) needs True.
            # 'ranking' matches 'rank' and not 'point'. 
            # 'ranking_points' contains 'point', so it gets False.
            target_col = targets[0]['col_name'] if targets else None
            col_lower = target_col.lower() if target_col else ""
            use_invert = True if ("rank" in col_lower and "point" not in col_lower) else False
            
            # If we can't find a target col, we skip trend analysis for this sample
            if target_col and target_col in filled_df.columns:
                series = filled_df[target_col]
                # Re-use the project's standard trend logic
                derived_trend = TrendAnalyzer.analyze_trend(series, invert_logic=use_invert)
                
                result['derived_trend_analysis'] = derived_trend
            else:
                result['derived_trend_analysis'] = None
        else:
             result['derived_trend_analysis'] = "analyzer_not_loaded"

        return result

    @staticmethod
    def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Computes aggregate metrics over a list of result samples.
        """
        # --- Imputation Metrics ---
        y_true = []
        y_pred = []
        
        # --- Trend Metrics ---
        trend_hits = 0
        trend_total = 0
        
        # --- Coverage Metrics ---
        total_targets_count = 0
        valid_predictions_count = 0
        
        for res in results:
            # 1. Gather Imputation Pairs
            for p in res.get('imputation_predictions', []):
                gt = p['gt']
                pred = p['pred']
                
                # Count total expected targets (where GT exists)
                if pd.notna(gt):
                    total_targets_count += 1
                
                # Filter out NaNs (failed imputations) for RMSE calc
                if pd.notna(gt) and pd.notna(pred):
                    y_true.append(float(gt))
                    y_pred.append(float(pred))
                    valid_predictions_count += 1
            
            # 2. Gather Trend Correctness
            gt_trend = str(res.get('gt_analysis', '')).lower().strip()
            pred_trend = str(res.get('derived_trend_analysis', '')).lower().strip()
            
            # Normalize basic terms
            # Rise
            if gt_trend in ['rise', 'increasing', 'upward']: gt_trend = 'rise'
            if pred_trend in ['rise', 'increasing', 'upward']: pred_trend = 'rise'
            
            # Fall
            if gt_trend in ['fall', 'decreasing', 'downward']: gt_trend = 'fall'
            if pred_trend in ['fall', 'decreasing', 'downward']: pred_trend = 'fall'
            
            # Stable
            if gt_trend in ['stable', 'flat', 'flat (no trend)', 'no trend']: gt_trend = 'stable'
            if pred_trend in ['stable', 'flat', 'flat (no trend)', 'no trend']: pred_trend = 'stable'

            if gt_trend and pred_trend and pred_trend != 'none':
                trend_total += 1
                if gt_trend == pred_trend:
                    trend_hits += 1
        
        # Calculate Stats
        metrics = {}
        
        if y_true:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            metrics['rmse'] = float(np.sqrt(np.mean((y_true - y_pred)**2)))
            metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        else:
            metrics['rmse'] = np.nan
            metrics['mae'] = np.nan
            
        if trend_total > 0:
            metrics['trend_accuracy'] = trend_hits / trend_total
        else:
            metrics['trend_accuracy'] = 0.0
            
        metrics['total_samples'] = len(results)
        metrics['total_imputations'] = len(y_true)
        metrics['total_targets'] = total_targets_count # Exposed for logging
        metrics['coverage'] = valid_predictions_count / total_targets_count if total_targets_count > 0 else 0.0
        
        # --- Metadata & Methodology ---
        metrics['_meta'] = {
            "nan_handling": "Predictions that are NaN/None are EXCLUDED from RMSE/MAE calculations. They are reflected in the 'coverage' score.",
            "metrics_definition": {
                "rmse": "Root Mean Squared Error on valid (non-NaN) pairs.",
                "mae": "Mean Absolute Error on valid (non-NaN) pairs.",
                "coverage": "Ratio of valid predictions (non-NaN) to total ground truth targets.",
                "trend_accuracy": "Downstream task accuracy: derived trend from filled table vs ground truth."
            }
        }
        
        return metrics
