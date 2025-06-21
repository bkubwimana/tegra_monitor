"""
Tests for specialized evaluators (Budget and NoReasoning).

Tests configuration loading, inheritance behavior, and mode-specific features.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

from src.evaluators import BudgetEvaluator, NoReasoningEvaluator
from src.models import PredictionResult


class TestBudgetEvaluator(unittest.TestCase):
    """Test cases for BudgetEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'budget.yaml')
        
    def test_budget_evaluator_initialization(self):
        """Test budget evaluator initializes correctly."""
        evaluator = BudgetEvaluator(self.test_config_path)
        
        self.assertEqual(evaluator.config.name, "budget")
        self.assertEqual(evaluator.config.model['max_tokens'], 128)  # Updated to match actual config
        self.assertEqual(evaluator.early_termination_count, 0)
        self.assertEqual(evaluator.token_savings, 0)
        
    def test_text_compression(self):
        """Test text compression functionality."""
        evaluator = BudgetEvaluator(self.test_config_path)
        
        # Test short text (should not be compressed)
        short_text = "This is a short question."
        compressed = evaluator._compress_text(short_text, max_words=10)
        self.assertEqual(compressed, short_text)
        
        # Test long text (should be compressed)
        long_text = " ".join([f"word{i}" for i in range(60)])  # 60 words
        compressed = evaluator._compress_text(long_text, max_words=10)
        self.assertTrue(compressed.endswith("..."))
        self.assertLess(len(compressed.split()), 15)  # Should be much shorter
        
    def test_early_answer_detection(self):
        """Test early answer detection patterns."""
        evaluator = BudgetEvaluator(self.test_config_path)
        
        test_cases = [
            ("The answer is A", "A"),
            ("Answer: B", "B"),
            ("I think the answer is C", "C"),
            ("(D) is correct", "D"),
            ("The correct answer is B", "B"),
            ("No clear answer here", None)
        ]
        
        for text, expected in test_cases:
            result = evaluator._detect_early_answer(text)
            self.assertEqual(result, expected, f"Failed for: {text}")
            
    def test_prompt_compression(self):
        """Test prompt formatting with compression."""
        evaluator = BudgetEvaluator(self.test_config_path)
        
        # Create a mock question data object that matches expected structure
        class MockQuestionData:
            def __init__(self, question, choices):
                self.question = question
                self.choices = choices
        
        # Create a very long question that will definitely be compressed
        long_question = " ".join([f"word{i}" for i in range(60)])  # 60 words
        
        question_data = MockQuestionData(
            question=long_question,
            choices=[
                " ".join([f"choiceA{i}" for i in range(15)]),  # 15 words
                " ".join([f"choiceB{i}" for i in range(15)]),  # 15 words
                " ".join([f"choiceC{i}" for i in range(15)]),  # 15 words
                " ".join([f"choiceD{i}" for i in range(15)])   # 15 words
            ]
        )
        
        prompt = evaluator.format_prompt(question_data)
        
        # Should contain compressed content with ellipsis
        self.assertIn("...", prompt)
        # Should contain the beginning of the question
        self.assertIn("word0 word1", prompt)
        
    def test_efficiency_score_calculation(self):
        """Test efficiency score calculation."""
        evaluator = BudgetEvaluator(self.test_config_path)
        
        # Mock result
        mock_result = Mock()
        mock_result.accuracy = 0.8
        mock_result.avg_tokens_per_second = 50.0
        
        efficiency_score = evaluator._calculate_efficiency_score(mock_result)
        
        self.assertGreater(efficiency_score, 0.0)
        self.assertLessEqual(efficiency_score, 1.0)
        self.assertAlmostEqual(efficiency_score, 0.7 * 0.8 + 0.3 * 0.5, places=2)


class TestNoReasoningEvaluator(unittest.TestCase):
    """Test cases for NoReasoningEvaluator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'noreasoning.yaml')
        
    def test_noreasoning_evaluator_initialization(self):
        """Test no-reasoning evaluator initializes correctly."""
        evaluator = NoReasoningEvaluator(self.test_config_path)
        
        self.assertEqual(evaluator.config.name, "noreasoning")
        self.assertEqual(evaluator.config.model['max_tokens'], 4096)
        self.assertEqual(evaluator.quick_responses, 0)
        self.assertEqual(len(evaluator.confidence_scores), 0)
        
    def test_direct_answer_extraction(self):
        """Test direct answer extraction with confidence."""
        evaluator = NoReasoningEvaluator(self.test_config_path)
        
        test_cases = [
            ("The answer is A", ("A", 1.0)),
            ("B", ("B", 1.0)),
            ("Answer: C", ("C", 1.0)),
            ("I think A A A", ("A", 1.0)),  # High consistency
            ("A or maybe B or A", ("A", 0.67)),  # Medium consistency
            ("No clear answer", ("UNKNOWN", 0.0))
        ]
        
        for text, (expected_choice, expected_confidence) in test_cases:
            choice, confidence = evaluator._extract_direct_answer(text)
            self.assertEqual(choice, expected_choice, f"Failed choice for: {text}")
            if expected_confidence == 1.0:
                self.assertGreaterEqual(confidence, 0.8, f"Failed confidence for: {text}")
            elif expected_confidence == 0.0:
                self.assertEqual(confidence, 0.0, f"Failed confidence for: {text}")
                
    def test_quick_response_detection(self):
        """Test quick response detection."""
        evaluator = NoReasoningEvaluator(self.test_config_path)
        
        # Quick response (short and fast)
        self.assertTrue(evaluator._is_quick_response("A", 1000))
        self.assertTrue(evaluator._is_quick_response("The answer is B", 1500))
        
        # Not quick response (long or slow)
        self.assertFalse(evaluator._is_quick_response("A", 3000))  # Too slow
        long_text = " ".join(["word"] * 30)
        self.assertFalse(evaluator._is_quick_response(long_text, 1000))  # Too long
        
    def test_prompt_formatting(self):
        """Test direct prompt formatting."""
        evaluator = NoReasoningEvaluator(self.test_config_path)
        
        # Create a mock question data object that matches expected structure
        class MockQuestionData:
            def __init__(self, question, choices):
                self.question = question
                self.choices = choices
        
        question_data = MockQuestionData(
            question="What is 2+2?",
            choices=["3", "4", "5", "6"]
        )
        
        prompt = evaluator.format_prompt(question_data)
        
        # Should contain all elements without reasoning requests
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("A) 3", prompt)
        self.assertIn("B) 4", prompt)
        self.assertNotIn("reasoning", prompt.lower())
        self.assertNotIn("explain", prompt.lower())
        
    def test_directness_score_calculation(self):
        """Test directness score calculation."""
        evaluator = NoReasoningEvaluator(self.test_config_path)
        
        # Mock result with fast, short responses
        mock_result = Mock()
        mock_result.avg_time_per_question = 1000  # 1 second
        mock_result.question_results = [
            {'generated_text': "A"},
            {'generated_text': "The answer is B"},
            {}  # Last item (metrics)
        ]
        
        directness_score = evaluator._calculate_directness_score(mock_result)
        
        self.assertGreater(directness_score, 0.0)
        self.assertLessEqual(directness_score, 1.0)


class TestEvaluatorIntegration(unittest.TestCase):
    """Integration tests for evaluators working together."""
    
    def test_evaluator_inheritance(self):
        """Test that specialized evaluators properly inherit from base."""
        budget_eval = BudgetEvaluator()
        noreasoning_eval = NoReasoningEvaluator()
        
        # Should have base evaluator methods
        self.assertTrue(hasattr(budget_eval, 'setup_model'))
        self.assertTrue(hasattr(budget_eval, 'evaluate_subject'))
        self.assertTrue(hasattr(noreasoning_eval, 'setup_model'))
        self.assertTrue(hasattr(noreasoning_eval, 'evaluate_subject'))
        
        # Should have specialized features
        self.assertTrue(hasattr(budget_eval, '_compress_text'))
        self.assertTrue(hasattr(budget_eval, '_detect_early_answer'))
        self.assertTrue(hasattr(noreasoning_eval, '_extract_direct_answer'))
        self.assertTrue(hasattr(noreasoning_eval, '_is_quick_response'))
        
    def test_config_compatibility(self):
        """Test that evaluations work with their respective configs."""
        # This test ensures configs load properly
        try:
            budget_eval = BudgetEvaluator()
            noreasoning_eval = NoReasoningEvaluator()
            
            self.assertEqual(budget_eval.config.name, "budget")
            self.assertEqual(noreasoning_eval.config.name, "noreasoning")
            
        except Exception as e:
            self.fail(f"Config compatibility test failed: {e}")


if __name__ == '__main__':
    unittest.main()
