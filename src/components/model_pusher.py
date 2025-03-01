import os
import sys
import shutil
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import ModelPusherConfig
from src.entity.artifact_entity import ModelEvaluationArtifact, ModelPusherArtifact

class ModelPusher:
    def __init__(
        self,
        model_pusher_config: ModelPusherConfig,
        model_evaluation_artifact: ModelEvaluationArtifact,
        data_transformation_artifact=None  # Add to pass transformer path
    ):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_pushing(self) -> ModelPusherArtifact:
        try:
            logging.info("Initiating model pushing")
            
            if not self.model_evaluation_artifact.is_model_accepted:
                logging.info("Model was not accepted by evaluation criteria")
                raise Exception("Model was not accepted by evaluation criteria")

            evaluated_model_path = self.model_evaluation_artifact.evaluated_model_path
            if not os.path.isfile(evaluated_model_path):
                raise Exception(f"Evaluated model path is not a valid file: {evaluated_model_path}")

            # Fixed directory and filenames
            api_assets_dir = "api_assets"
            model_file_name = "model.pkl"
            transformer_file_name = "transformer.pkl"
            
            # Save model to api_assets/
            push_model_path = os.path.join(api_assets_dir, model_file_name)
            os.makedirs(api_assets_dir, exist_ok=True)
            if os.path.exists(push_model_path):
                logging.warning(f"Overwriting existing model at {push_model_path}")
            shutil.copy2(evaluated_model_path, push_model_path)

            # Save transformer to api_assets/
            if self.data_transformation_artifact and hasattr(self.data_transformation_artifact, 'transformed_object_file_path'):
                transformer_source = self.data_transformation_artifact.transformed_object_file_path
                saved_transformer_path = os.path.join(api_assets_dir, transformer_file_name)
                if not os.path.isfile(transformer_source):
                    raise Exception(f"Transformer path is not a valid file: {transformer_source}")
                if os.path.exists(saved_transformer_path):
                    logging.warning(f"Overwriting existing transformer at {saved_transformer_path}")
                shutil.copy2(transformer_source, saved_transformer_path)
            else:
                logging.warning("No transformer artifact provided; skipping transformer push")
                saved_transformer_path = None

            model_version = "1.0.0"  # You can make this dynamic if needed

            model_pusher_artifact = ModelPusherArtifact(
                deployed_model_path=push_model_path,
                saved_model_path=saved_transformer_path if saved_transformer_path else push_model_path,  # Adjust if you want separate logic
                model_version=model_version
            )
            logging.info(f"Model pushing completed. Artifact: {model_pusher_artifact}")
            return model_pusher_artifact

        except Exception as e:
            raise CustomException(e, sys)