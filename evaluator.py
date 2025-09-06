import json
import os
import time
from typing import Any, Dict, List
from datetime import datetime
import difflib
import re

# Simplified base classes for standalone operation
class EvaluationResult:
    def __init__(self, task_id, agent_id, timestamp, metrics, success, execution_time, error_message=None):
        self.task_id = task_id
        self.agent_id = agent_id
        self.timestamp = timestamp
        self.metrics = metrics
        self.success = success
        self.execution_time = execution_time
        self.error_message = error_message

class BaseEvaluator:
    def __init__(self, config):
        self.config = config
    
    def print_task_info(self):
        print(f"Task: {self.config.get('name', 'Unknown')}")
        print(f"Description: {self.config.get('description', 'No description')}")


class OCRTextExtractionEvaluator(BaseEvaluator):
    """
    Evaluator for the OCR Text Extraction task.
    Compares predicted text against ground truth using text similarity metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.accuracy_threshold = config.get("evaluation_criteria", {}).get("accuracy_threshold", 0.8)
        self.print_task_info()

    def evaluate(self, solution_folder: str, solution_config: Any = None) -> EvaluationResult:
        start_time = time.time()
        try:
            solution_file_name = self.config["expected_outputs"]["solution_file"]
            solution_path = os.path.join(solution_folder, solution_file_name)

            if not os.path.exists(solution_path):
                return EvaluationResult(
                    task_id=self.config["task_id"],
                    agent_id="unknown",
                    timestamp=datetime.now(),
                    metrics={"text_similarity": 0.0, "character_accuracy": 0.0, "word_accuracy": 0.0, "total_images": 0, "successful_extractions": 0},
                    success=False,
                    execution_time=time.time() - start_time,
                    error_message=f"Solution file {solution_file_name} not found",
                )

            predictions = self._load_predictions_json(solution_path)

            # Load ground truth
            if solution_config is not None:
                ground_truth = solution_config
            else:
                gt_file = self.config["expected_outputs"]["ground_truth_file"]
                task_dir = os.path.dirname(__file__)
                gt_path = os.path.join(task_dir, gt_file)
                if not os.path.exists(gt_path):
                    return EvaluationResult(
                        task_id=self.config["task_id"],
                        agent_id="unknown",
                        timestamp=datetime.now(),
                        metrics={"text_similarity": 0.0, "character_accuracy": 0.0, "word_accuracy": 0.0, "total_images": 0, "successful_extractions": 0},
                        success=False,
                        execution_time=time.time() - start_time,
                        error_message=f"Ground truth file {gt_file} not found",
                    )
                ground_truth = self._load_ground_truth_json(gt_path)

            metrics = self._calculate_metrics(predictions, ground_truth)
            success = metrics["text_similarity"] >= self.accuracy_threshold

            return EvaluationResult(
                task_id=self.config["task_id"],
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics=metrics,
                success=success,
                execution_time=time.time() - start_time,
                error_message=None if success else f"Text similarity {metrics['text_similarity']:.3f} below threshold {self.accuracy_threshold}",
            )
        except Exception as e:
            return EvaluationResult(
                task_id=self.config["task_id"],
                agent_id="unknown",
                timestamp=datetime.now(),
                metrics={"text_similarity": 0.0, "character_accuracy": 0.0, "word_accuracy": 0.0, "total_images": 0, "successful_extractions": 0},
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Evaluation error: {str(e)}",
            )

    def _load_predictions_json(self, json_path: str) -> Dict[str, Dict[str, Any]]:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data.get("predictions", {})

    def _load_ground_truth_json(self, json_path: str) -> Dict[str, Dict[str, Any]]:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
        if not text:
            return ""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return normalized

    def _calculate_text_similarity(self, pred_text: str, true_text: str) -> float:
        """Calculate text similarity using difflib SequenceMatcher."""
        if not pred_text and not true_text:
            return 1.0
        if not pred_text or not true_text:
            return 0.0
        
        pred_normalized = self._normalize_text(pred_text)
        true_normalized = self._normalize_text(true_text)
        
        return difflib.SequenceMatcher(None, pred_normalized, true_normalized).ratio()

    def _calculate_character_accuracy(self, pred_text: str, true_text: str) -> float:
        """Calculate character-level accuracy."""
        if not true_text:
            return 1.0 if not pred_text else 0.0
        
        pred_normalized = self._normalize_text(pred_text)
        true_normalized = self._normalize_text(true_text)
        
        if len(true_normalized) == 0:
            return 1.0 if len(pred_normalized) == 0 else 0.0
        
        matches = sum(1 for a, b in zip(pred_normalized, true_normalized) if a == b)
        return matches / len(true_normalized)

    def _calculate_word_accuracy(self, pred_text: str, true_text: str) -> float:
        """Calculate word-level accuracy."""
        if not true_text:
            return 1.0 if not pred_text else 0.0
        
        pred_words = self._normalize_text(pred_text).split()
        true_words = self._normalize_text(true_text).split()
        
        if len(true_words) == 0:
            return 1.0 if len(pred_words) == 0 else 0.0
        
        # Calculate word-level matches
        true_word_set = set(true_words)
        pred_word_set = set(pred_words)
        
        correct_words = len(true_word_set.intersection(pred_word_set))
        return correct_words / len(true_word_set)

    def _calculate_metrics(self, predictions: Dict[str, Dict[str, Any]], ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        if not ground_truth:
            return {
                "text_similarity": 0.0, 
                "character_accuracy": 0.0, 
                "word_accuracy": 0.0, 
                "total_images": 0, 
                "successful_extractions": 0,
                "average_confidence": 0.0
            }

        total_similarity = 0.0
        total_char_accuracy = 0.0
        total_word_accuracy = 0.0
        total_confidence = 0.0
        successful_extractions = 0
        total_images = len(ground_truth)

        for filename, true_data in ground_truth.items():
            true_text = true_data.get("text", "")
            pred_data = predictions.get(filename, {})
            pred_text = pred_data.get("text", "")
            pred_confidence = pred_data.get("confidence", 0.0)

            # Calculate similarity metrics
            similarity = self._calculate_text_similarity(pred_text, true_text)
            char_accuracy = self._calculate_character_accuracy(pred_text, true_text)
            word_accuracy = self._calculate_word_accuracy(pred_text, true_text)

            total_similarity += similarity
            total_char_accuracy += char_accuracy
            total_word_accuracy += word_accuracy
            total_confidence += pred_confidence

            if similarity > 0.5:  # Consider extraction successful if similarity > 50%
                successful_extractions += 1

        return {
            "text_similarity": total_similarity / total_images if total_images > 0 else 0.0,
            "character_accuracy": total_char_accuracy / total_images if total_images > 0 else 0.0,
            "word_accuracy": total_word_accuracy / total_images if total_images > 0 else 0.0,
            "total_images": float(total_images),
            "successful_extractions": float(successful_extractions),
            "average_confidence": total_confidence / total_images if total_images > 0 else 0.0,
        }

    def get_metrics(self) -> List[str]:
        return [
            "text_similarity",
            "character_accuracy", 
            "word_accuracy",
            "total_images",
            "successful_extractions",
            "average_confidence",
        ]

    def generate_report(self, results: List[EvaluationResult]) -> str:
        if not results:
            return "No evaluation results to report."
        lines = []
        lines.append("OCR Text Extraction - Evaluation Report")
        lines.append("=" * 60)
        for i, result in enumerate(results, 1):
            lines.append(f"\nEvaluation {i}:")
            lines.append(f"  Task ID: {result.task_id}")
            lines.append(f"  Agent ID: {result.agent_id}")
            lines.append(f"  Timestamp: {result.timestamp}")
            lines.append(f"  Success: {result.success}")
            lines.append(f"  Execution Time: {result.execution_time:.2f}s")
            if result.error_message:
                lines.append(f"  Error: {result.error_message}")
            lines.append("  Metrics:")
            for metric, value in result.metrics.items():
                if metric in ("text_similarity", "character_accuracy", "word_accuracy", "average_confidence"):
                    lines.append(f"    {metric}: {value:.3f} ({value*100:.1f}%)")
                else:
                    lines.append(f"    {metric}: {value}")
        return "\n".join(lines)
