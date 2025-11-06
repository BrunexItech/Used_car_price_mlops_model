import schedule
import time
import logging
import joblib
from datetime import datetime
from ..models.model_trainer import ModelTrainer
from src.features.__main__ import main as feature_pipeline
from monitoring.drift_detector import DriftDetector
from ..config import MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoTrainer:
    def __init__(self):
        self.retrain_interval_days = 30
        self.drift_threshold = 0.2
        self.performance_threshold = 0.05  # 5% performance drop
    
    def check_retraining_conditions(self):
        """Check if model needs retraining"""
        try:
            # Load current model and drift detector
            model_path = MODELS_DIR / "car_price_model.joblib"
            drift_detector_path = MODELS_DIR / "drift_detector.joblib"
            
            if not model_path.exists() or not drift_detector_path.exists():
                logger.warning("Model or drift detector not found. Retraining needed.")
                return True
            
            # Load drift detector
            drift_detector = joblib.load(drift_detector_path)
            
            # Get current data for drift check
            from src.data.__main__ import main as data_pipeline
            from src.features.feature_engineer import FeatureEngineer
            
            clean_data = data_pipeline()
            engineer = FeatureEngineer()
            current_encoded_data = engineer.encode_categorical(clean_data)
            
            # Check for data drift
            drift_result = drift_detector.check_drift(current_encoded_data)
            
            if drift_result and drift_result['needs_retraining']:
                logger.info("Data drift detected. Retraining triggered.")
                return True
            
            logger.info("No retraining conditions met.")
            return False
            
        except Exception as e:
            logger.error(f"Error checking retraining conditions: {e}")
            return False
    
    def auto_retrain(self):
        """Automated retraining pipeline"""
        if not self.check_retraining_conditions():
            logger.info("Skipping retraining - conditions not met")
            return
        
        logger.info("Starting automated retraining...")
        
        try:
            # Get latest data and features
            X_train_scaled, X_test_scaled, y_train, y_test, engineer = feature_pipeline()
            
            # Train new model
            trainer = ModelTrainer()
            models = trainer.train_models(X_train_scaled, y_train)
            final_model = trainer.fine_tune_best_model(X_train_scaled, y_train)
            
            # Evaluate new model
            new_r2, new_rmse = trainer.evaluate_final_model(X_test_scaled, y_test)
            
            # Load old model performance for comparison
            old_r2 = self._get_current_model_performance()
            
            # Deploy if performance is better or within threshold
            if old_r2 is None or new_r2 >= old_r2 - self.performance_threshold:
                trainer.save_model("car_price_model.joblib")
                
                # Update drift detector with new reference data
                from src.data.__main__ import main as data_pipeline
                clean_data = data_pipeline()
                encoded_data = engineer.encode_categorical(clean_data)
                
                drift_detector = DriftDetector()
                drift_detector.set_reference_data(encoded_data)
                joblib.dump(drift_detector, MODELS_DIR / "drift_detector.joblib")
                
                logger.info(f"New model deployed! R²: {new_r2:.4f}")
            else:
                logger.warning(f"New model performance worse. Old: {old_r2:.4f}, New: {new_r2:.4f}")
                
        except Exception as e:
            logger.error(f"Auto-retraining failed: {e}")
    
    def _get_current_model_performance(self):
        """Get current model performance from MLflow"""
        try:
            import mlflow
            # This would query MLflow for the latest model performance
            # For now, return None to always retrain
            return None
        except:
            return None
    
    def start_scheduler(self):
        """Start the automated retraining scheduler"""
        logger.info("Starting auto-retraining scheduler...")
        
        # Schedule retraining check every day
        schedule.every(1).days.do(self.auto_retrain)
        
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour

if __name__ == "__main__":
    auto_trainer = AutoTrainer()
    auto_trainer.start_scheduler()