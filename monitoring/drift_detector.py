import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable
import logging
from src.config import PROCESSED_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    def __init__(self):
        self.reference_data = None
        self.drift_threshold = 0.2  # 20% drift threshold
    
    def set_reference_data(self, data):
        """Set the reference dataset (training data)"""
        self.reference_data = data
        logger.info("Reference data set for drift detection")
    
    def check_drift(self, current_data):
        """Check for data drift between current and reference data"""
        if self.reference_data is None:
            logger.error("No reference data set. Call set_reference_data() first.")
            return None
        
        try:
            # Create drift report
            drift_report = Report(metrics=[DataDriftTable()])
            drift_report.run(
                reference_data=self.reference_data,
                current_data=current_data
            )
            
            report_results = drift_report.as_dict()
            
            # Extract drift score
            drift_score = report_results['metrics'][0]['result']['dataset_drift']
            num_drifted_columns = report_results['metrics'][0]['result']['number_of_drifted_columns']
            
            logger.info(f"Data drift detected: {drift_score}")
            logger.info(f"Number of drifted columns: {num_drifted_columns}")
            
            return {
                'drift_detected': drift_score,
                'drift_score': drift_score,
                'drifted_columns': num_drifted_columns,
                'needs_retraining': drift_score > self.drift_threshold
            }
            
        except Exception as e:
            logger.error(f"Drift detection error: {e}")
            return None