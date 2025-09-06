#!/usr/bin/env python3
"""
Agent Testing Framework for OCR Task
Simulates running the OCR task on 5 different AI models and evaluates their performance.
"""

import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List, Any
from evaluator import OCRTextExtractionEvaluator

# Model configurations
MODELS = {
    "AIDE_Claude_Sonnet_4": {
        "description": "AIDE (Claude Sonnet 4) - Advanced AI Development Environment",
        "strengths": ["code generation", "complex reasoning", "multimodal understanding"],
        "base_accuracy": 0.85,
        "confidence_variance": 0.1
    },
    "OpenHands_Claude_Sonnet_4": {
        "description": "OpenHands (Claude Sonnet 4) - Open-source hand tracking and manipulation",
        "strengths": ["computer vision", "spatial reasoning", "object detection"],
        "base_accuracy": 0.82,
        "confidence_variance": 0.12
    },
    "GoogleCLI_Gemini_2.5_Pro": {
        "description": "GoogleCLI (Gemini 2.5 Pro) - Google's command-line interface with Gemini",
        "strengths": ["multimodal processing", "document understanding", "API integration"],
        "base_accuracy": 0.88,
        "confidence_variance": 0.08
    },
    "Claude_Code_Claude_Sonnet_4": {
        "description": "Claude Code (Claude Sonnet 4) - Specialized for code generation and analysis",
        "strengths": ["code analysis", "text processing", "pattern recognition"],
        "base_accuracy": 0.80,
        "confidence_variance": 0.15
    },
    "Claude_Sonnet_4_General": {
        "description": "Claude Sonnet 4 (General) - General-purpose AI assistant",
        "strengths": ["general reasoning", "text understanding", "problem solving"],
        "base_accuracy": 0.83,
        "confidence_variance": 0.11
    }
}

class OCRAgentSimulator:
    """Simulates different AI agents performing OCR tasks."""
    
    def __init__(self, model_name: str, model_config: Dict[str, Any]):
        self.model_name = model_name
        self.config = model_config
        self.base_accuracy = model_config["base_accuracy"]
        self.confidence_variance = model_config["confidence_variance"]
        
    def simulate_ocr_extraction(self, ground_truth: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Simulate OCR text extraction with realistic variations."""
        predictions = {"predictions": {}}
        
        for filename, gt_data in ground_truth.items():
            true_text = gt_data["text"]
            difficulty = gt_data.get("difficulty", "medium")
            image_type = gt_data.get("type", "medium")
            
            # Adjust accuracy based on difficulty and image type
            accuracy_modifier = self._get_accuracy_modifier(difficulty, image_type)
            current_accuracy = self.base_accuracy * accuracy_modifier
            
            # Simulate text extraction with errors
            predicted_text = self._simulate_text_extraction(true_text, current_accuracy)
            
            # Generate realistic confidence score
            confidence = self._generate_confidence_score(current_accuracy, image_type)
            
            predictions["predictions"][filename] = {
                "text": predicted_text,
                "confidence": confidence
            }
        
        return predictions
    
    def _get_accuracy_modifier(self, difficulty: str, image_type: str) -> float:
        """Adjust accuracy based on task difficulty and image type."""
        modifiers = {
            "easy": 1.0,
            "medium": 0.9,
            "hard": 0.7
        }
        
        type_modifiers = {
            "easy": 1.0,
            "medium": 0.95,
            "hard": 0.85,
            "handwritten": 0.8,
            "document": 0.9,
            "rotated": 0.75,
            "noisy": 0.7
        }
        
        return modifiers.get(difficulty, 1.0) * type_modifiers.get(image_type, 1.0)
    
    def _simulate_text_extraction(self, true_text: str, accuracy: float) -> str:
        """Simulate OCR text extraction with realistic errors."""
        if not true_text:
            return ""
        
        # Simulate different types of OCR errors
        predicted_text = true_text
        
        # Character substitution errors (common in OCR)
        if random.random() > accuracy:
            predicted_text = self._introduce_character_errors(predicted_text)
        
        # Word boundary errors
        if random.random() > accuracy * 0.8:
            predicted_text = self._introduce_word_errors(predicted_text)
        
        # Case sensitivity issues
        if random.random() > accuracy * 0.9:
            predicted_text = self._introduce_case_errors(predicted_text)
        
        # Spacing issues
        if random.random() > accuracy * 0.85:
            predicted_text = self._introduce_spacing_errors(predicted_text)
        
        return predicted_text
    
    def _introduce_character_errors(self, text: str) -> str:
        """Introduce character substitution errors common in OCR."""
        char_substitutions = {
            '0': 'O', 'O': '0', '1': 'l', 'l': '1', 'I': '1', '1': 'I',
            '5': 'S', 'S': '5', '6': 'G', 'G': '6', '8': 'B', 'B': '8',
            'rn': 'm', 'm': 'rn', 'cl': 'd', 'd': 'cl'
        }
        
        result = text
        for wrong, right in char_substitutions.items():
            if random.random() < 0.1:  # 10% chance per substitution
                result = result.replace(right, wrong)
        return result
    
    def _introduce_word_errors(self, text: str) -> str:
        """Introduce word boundary and segmentation errors."""
        words = text.split()
        if len(words) <= 1:
            return text
        
        # Sometimes merge words
        if random.random() < 0.1 and len(words) > 1:
            merge_idx = random.randint(0, len(words) - 2)
            words[merge_idx] = words[merge_idx] + words[merge_idx + 1]
            words.pop(merge_idx + 1)
        
        # Sometimes split words
        if random.random() < 0.05:
            split_idx = random.randint(0, len(words) - 1)
            if len(words[split_idx]) > 3:
                mid = len(words[split_idx]) // 2
                words.insert(split_idx + 1, words[split_idx][mid:])
                words[split_idx] = words[split_idx][:mid]
        
        return ' '.join(words)
    
    def _introduce_case_errors(self, text: str) -> str:
        """Introduce case sensitivity errors."""
        if random.random() < 0.2:
            # Randomly change case
            return ''.join(c.upper() if random.random() < 0.3 else c.lower() for c in text)
        return text
    
    def _introduce_spacing_errors(self, text: str) -> str:
        """Introduce spacing errors."""
        if random.random() < 0.15:
            # Add extra spaces
            return text.replace(' ', '  ')
        elif random.random() < 0.1:
            # Remove some spaces
            return text.replace(' ', '')
        return text
    
    def _generate_confidence_score(self, accuracy: float, image_type: str) -> float:
        """Generate realistic confidence scores."""
        base_confidence = accuracy * 0.9  # Slightly lower than actual accuracy
        
        # Adjust based on image type
        type_adjustments = {
            "easy": 0.1,
            "medium": 0.0,
            "hard": -0.1,
            "handwritten": -0.15,
            "document": -0.05,
            "rotated": -0.2,
            "noisy": -0.25
        }
        
        adjustment = type_adjustments.get(image_type, 0.0)
        confidence = base_confidence + adjustment + random.uniform(-self.confidence_variance, self.confidence_variance)
        
        return max(0.0, min(1.0, confidence))

def run_agent_tests():
    """Run OCR task tests on all 5 models."""
    print("🤖 OCR Agent Testing Framework")
    print("=" * 50)
    
    # Load ground truth
    with open("ground_truth_text.json", "r") as f:
        ground_truth = json.load(f)
    
    # Load task config
    with open("config.json", "r") as f:
        task_config = json.load(f)
    
    evaluator = OCRTextExtractionEvaluator(task_config)
    results = []
    
    # Test each model
    for model_name, model_config in MODELS.items():
        print(f"\n🔍 Testing {model_name}...")
        print(f"   Description: {model_config['description']}")
        print(f"   Strengths: {', '.join(model_config['strengths'])}")
        
        # Create agent simulator
        agent = OCRAgentSimulator(model_name, model_config)
        
        # Simulate OCR extraction
        predictions = agent.simulate_ocr_extraction(ground_truth)
        
        # Save predictions to solution.json for evaluation
        with open("solution.json", "w") as f:
            json.dump(predictions, f, indent=2)
        
        # Evaluate performance
        result = evaluator.evaluate(".", ground_truth)
        result.agent_id = model_name
        
        # Debug: Print some predictions to verify format
        print(f"   🔍 Sample prediction: {list(predictions['predictions'].items())[0] if predictions['predictions'] else 'None'}")
        results.append(result)
        
        # Print results
        print(f"   ✅ Evaluation completed")
        print(f"   📊 Text Similarity: {result.metrics['text_similarity']:.3f}")
        print(f"   📊 Character Accuracy: {result.metrics['character_accuracy']:.3f}")
        print(f"   📊 Word Accuracy: {result.metrics['word_accuracy']:.3f}")
        print(f"   📊 Average Confidence: {result.metrics['average_confidence']:.3f}")
        print(f"   ⏱️  Execution Time: {result.execution_time:.2f}s")
        print(f"   🎯 Success: {'✅' if result.success else '❌'}")
        
        # Clean up solution file
        if os.path.exists("solution.json"):
            os.remove("solution.json")
    
    return results

def generate_comparison_report(results: List[Any]):
    """Generate a comprehensive comparison report."""
    print("\n" + "=" * 80)
    print("📊 OCR AGENT COMPARISON REPORT")
    print("=" * 80)
    
    # Sort results by text similarity
    sorted_results = sorted(results, key=lambda x: x.metrics['text_similarity'], reverse=True)
    
    print(f"\n🏆 RANKING BY TEXT SIMILARITY:")
    print("-" * 50)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result.agent_id}")
        print(f"   Text Similarity: {result.metrics['text_similarity']:.3f} ({result.metrics['text_similarity']*100:.1f}%)")
        print(f"   Character Accuracy: {result.metrics['character_accuracy']:.3f}")
        print(f"   Word Accuracy: {result.metrics['word_accuracy']:.3f}")
        print(f"   Average Confidence: {result.metrics['average_confidence']:.3f}")
        print(f"   Success: {'✅' if result.success else '❌'}")
        print()
    
    # Detailed metrics table
    print("📈 DETAILED METRICS COMPARISON:")
    print("-" * 80)
    print(f"{'Model':<30} {'Text Sim':<8} {'Char Acc':<8} {'Word Acc':<8} {'Conf':<8} {'Success':<8}")
    print("-" * 80)
    
    for result in sorted_results:
        model_short = result.agent_id.replace("_Claude_Sonnet_4", "").replace("_Gemini_2.5_Pro", "")
        print(f"{model_short:<30} {result.metrics['text_similarity']:<8.3f} {result.metrics['character_accuracy']:<8.3f} {result.metrics['word_accuracy']:<8.3f} {result.metrics['average_confidence']:<8.3f} {'✅' if result.success else '❌':<8}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    print("-" * 30)
    
    best_model = sorted_results[0]
    worst_model = sorted_results[-1]
    
    print(f"🥇 Best Performer: {best_model.agent_id}")
    print(f"   - Text Similarity: {best_model.metrics['text_similarity']:.3f}")
    print(f"   - Character Accuracy: {best_model.metrics['character_accuracy']:.3f}")
    
    print(f"\n🥉 Needs Improvement: {worst_model.agent_id}")
    print(f"   - Text Similarity: {worst_model.metrics['text_similarity']:.3f}")
    print(f"   - Character Accuracy: {worst_model.metrics['character_accuracy']:.3f}")
    
    # Success rate
    successful_models = sum(1 for r in results if r.success)
    print(f"\n📊 Overall Success Rate: {successful_models}/{len(results)} ({successful_models/len(results)*100:.1f}%)")
    
    # Average performance
    avg_text_sim = sum(r.metrics['text_similarity'] for r in results) / len(results)
    avg_char_acc = sum(r.metrics['character_accuracy'] for r in results) / len(results)
    avg_word_acc = sum(r.metrics['word_accuracy'] for r in results) / len(results)
    avg_conf = sum(r.metrics['average_confidence'] for r in results) / len(results)
    
    print(f"\n📈 Average Performance Across All Models:")
    print(f"   - Text Similarity: {avg_text_sim:.3f}")
    print(f"   - Character Accuracy: {avg_char_acc:.3f}")
    print(f"   - Word Accuracy: {avg_word_acc:.3f}")
    print(f"   - Average Confidence: {avg_conf:.3f}")

def main():
    """Main function to run all tests."""
    print("🚀 Starting OCR Agent Testing...")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run tests
        results = run_agent_tests()
        
        # Generate report
        generate_comparison_report(results)
        
        print(f"\n✅ Testing completed successfully!")
        print(f"⏰ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
