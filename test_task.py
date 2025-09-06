#!/usr/bin/env python3
"""
Test script for OCR task to verify structure and functionality.
"""

import json
import os
from evaluator import OCRTextExtractionEvaluator

def test_task_structure():
    """Test that all required files exist."""
    required_files = [
        "config.json",
        "prompt.md", 
        "evaluator.py",
        "generate_inputs.py",
        "requirements.txt",
        "ground_truth_text.json",
        "README.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_config_loading():
    """Test that config.json loads correctly."""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        required_keys = ["task_id", "name", "prompt", "description", "difficulty", 
                        "category", "input_dir", "expected_outputs", "evaluation_criteria"]
        
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            print(f"❌ Missing config keys: {missing_keys}")
            return False
        else:
            print("✅ Config file structure valid")
            return True
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return False

def test_evaluator():
    """Test that evaluator can be instantiated and run."""
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        
        evaluator = OCRTextExtractionEvaluator(config)
        print("✅ Evaluator instantiated successfully")
        
        # Test with sample predictions
        sample_predictions = {
            "predictions": {
                "image_1.png": {
                    "text": "Hello World",
                    "confidence": 0.95
                },
                "image_2.png": {
                    "text": "The quick brown fox jumps over the lazy dog",
                    "confidence": 0.87
                }
            }
        }
        
        # Create temporary solution file
        with open("test_solution.json", "w") as f:
            json.dump(sample_predictions, f)
        
        # Test evaluation
        result = evaluator.evaluate(".", None)
        print(f"✅ Evaluation completed - Success: {result.success}")
        print(f"   Text Similarity: {result.metrics.get('text_similarity', 0):.3f}")
        print(f"   Character Accuracy: {result.metrics.get('character_accuracy', 0):.3f}")
        
        # Clean up
        if os.path.exists("test_solution.json"):
            os.remove("test_solution.json")
        
        return True
    except Exception as e:
        print(f"❌ Error testing evaluator: {e}")
        return False

def test_ground_truth():
    """Test that ground truth file loads correctly."""
    try:
        with open("ground_truth_text.json", "r") as f:
            gt = json.load(f)
        
        if not isinstance(gt, dict):
            print("❌ Ground truth should be a dictionary")
            return False
        
        # Check structure of first few entries
        for filename, data in list(gt.items())[:3]:
            if not isinstance(data, dict):
                print(f"❌ Ground truth entry {filename} should be a dictionary")
                return False
            if "text" not in data:
                print(f"❌ Ground truth entry {filename} missing 'text' field")
                return False
        
        print(f"✅ Ground truth loaded - {len(gt)} images")
        return True
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing OCR Task Structure")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_task_structure),
        ("Config Loading", test_config_loading),
        ("Ground Truth", test_ground_truth),
        ("Evaluator", test_evaluator)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"   Test failed!")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! OCR task is ready for agent testing.")
    else:
        print("⚠️  Some tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()
