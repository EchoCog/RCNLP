#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: test_agentic_kernel.py
# Description: Test suite for the agentic cognitive kernel
# Author: RCNLP Agentic System
# Date: 2024

import sys
import os
import numpy as np
import unittest
from typing import List, Dict, Any

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from echo_ggml_kernel import (
    AgenticCognitiveKernel,
    EchoGGMLNetwork,
    HypergraphGrammar,
    ECANAttentionAllocator,
    HypergraphNode,
    GGMLTensor,
    generate_test_sequences,
    run_cognitive_benchmark
)

class TestGGMLTensor(unittest.TestCase):
    """Test GGML tensor operations"""
    
    def test_tensor_creation(self):
        """Test tensor creation and basic properties"""
        tensor = GGMLTensor((3, 4))
        self.assertEqual(tensor.shape, (3, 4))
        self.assertEqual(tensor.data.shape, (3, 4))
    
    def test_tensor_operations(self):
        """Test basic tensor operations"""
        t1 = GGMLTensor((2, 3), np.ones((2, 3)))
        t2 = GGMLTensor((2, 3), np.ones((2, 3)) * 2)
        
        # Test addition
        result = t1.add(t2)
        expected = np.ones((2, 3)) * 3
        np.testing.assert_array_equal(result.data, expected)
        
        # Test tanh
        result = t1.tanh()
        expected = np.tanh(np.ones((2, 3)))
        np.testing.assert_array_almost_equal(result.data, expected)
    
    def test_tensor_matmul(self):
        """Test matrix multiplication"""
        t1 = GGMLTensor((2, 3), np.ones((2, 3)))
        t2 = GGMLTensor((3, 2), np.ones((3, 2)) * 2)
        
        result = t1.matmul(t2)
        expected = np.ones((2, 2)) * 6  # 3 * 2 = 6
        np.testing.assert_array_equal(result.data, expected)

class TestEchoGGMLNetwork(unittest.TestCase):
    """Test Echo State Network with GGML tensors"""
    
    def setUp(self):
        """Set up test ESN"""
        self.esn = EchoGGMLNetwork(
            reservoir_size=10,
            input_dim=5,
            attention_dim=3
        )
    
    def test_esn_initialization(self):
        """Test ESN initialization"""
        self.assertEqual(self.esn.reservoir_size, 10)
        self.assertEqual(self.esn.input_dim, 5)
        self.assertEqual(self.esn.attention_dim, 3)
        self.assertEqual(self.esn.W.shape, (10, 10))
        self.assertEqual(self.esn.W_in.shape, (10, 5))
        self.assertEqual(self.esn.W_att.shape, (10, 3))
    
    def test_symbols_to_vector(self):
        """Test symbol-to-vector conversion"""
        symbols = ['start', 'process', 'end']
        vector = self.esn._symbols_to_vector(symbols)
        
        self.assertEqual(len(vector), self.esn.input_dim)
        self.assertEqual(vector[0], 1.0)  # 'start'
        self.assertEqual(vector[1], 0.5)  # 'process'
        self.assertEqual(vector[2], -1.0)  # 'end'
    
    def test_forward_pass(self):
        """Test forward pass through ESN"""
        symbols = ['start', 'process']
        esn_state = self.esn.forward(symbols)
        
        self.assertEqual(esn_state.reservoir_size, 10)
        self.assertEqual(esn_state.input_dim, 5)
        self.assertEqual(esn_state.attention_dim, 3)
        self.assertEqual(len(esn_state.activation), 10)
    
    def test_forward_with_attention(self):
        """Test forward pass with attention"""
        symbols = ['start', 'process']
        attention = np.array([0.1, 0.2, 0.3])
        
        esn_state = self.esn.forward(symbols, attention)
        self.assertEqual(len(esn_state.activation), 10)

class TestHypergraphGrammar(unittest.TestCase):
    """Test hypergraph grammar operations"""
    
    def setUp(self):
        """Set up test hypergraph"""
        initial_patterns = [
            HypergraphNode('node1', 'symbol', 1.0, [], {}),
            HypergraphNode('node2', 'tensor', 0.5, [], {}),
            HypergraphNode('node3', 'grammar', 0.8, [], {})
        ]
        self.hypergraph = HypergraphGrammar(initial_patterns)
    
    def test_hypergraph_initialization(self):
        """Test hypergraph initialization"""
        self.assertEqual(len(self.hypergraph.grammar.patterns), 3)
        self.assertEqual(self.hypergraph.grammar.depth, 1)
        self.assertEqual(self.hypergraph.grammar.complexity, 0.0)
    
    def test_update_from_esn_state(self):
        """Test hypergraph update from ESN state"""
        # Create mock ESN state
        from echo_ggml_kernel import ESNState
        esn_state = ESNState(
            reservoir_size=10,
            input_dim=5,
            attention_dim=3,
            weights=np.zeros((10, 10)),
            attention_weights=np.zeros((10, 3)),
            activation=np.array([0.5, 0.8, 0.2, 0.1, 0.9])
        )
        
        initial_complexity = self.hypergraph.grammar.complexity
        updated_grammar = self.hypergraph.update_from_esn_state(esn_state)
        
        # Check that complexity increased
        self.assertGreater(updated_grammar.complexity, initial_complexity)
        
        # Check that pattern weights were updated
        self.assertNotEqual(updated_grammar.patterns[0].weight, 1.0)

class TestECANAttentionAllocator(unittest.TestCase):
    """Test ECAN-style attention allocation"""
    
    def setUp(self):
        """Set up test attention allocator"""
        self.allocator = ECANAttentionAllocator(total_resources=1.0)
        
        # Create test patterns
        self.patterns = [
            HypergraphNode('node1', 'symbol', 1.0, ['node2'], {}),
            HypergraphNode('node2', 'tensor', 0.5, [], {}),
            HypergraphNode('node3', 'grammar', 0.8, ['node1', 'node2'], {})
        ]
    
    def test_attention_allocation(self):
        """Test attention allocation computation"""
        from echo_ggml_kernel import CognitiveGrammar, ESNState
        
        grammar = CognitiveGrammar(self.patterns, [], 1, 0.5)
        esn_state = ESNState(
            reservoir_size=10,
            input_dim=5,
            attention_dim=3,
            weights=np.zeros((10, 10)),
            attention_weights=np.zeros((10, 3)),
            activation=np.array([0.7, 0.3, 0.9])
        )
        
        attention = self.allocator.allocate_attention(grammar, esn_state)
        
        # Check that attention sums to total resources
        self.assertAlmostEqual(np.sum(attention), 1.0, places=5)
        
        # Check that attention is non-negative
        self.assertTrue(np.all(attention >= 0))
    
    def test_importance_scores(self):
        """Test importance score computation"""
        scores = self.allocator._compute_importance_scores(self.patterns)
        
        self.assertEqual(len(scores), 3)
        # Node3 should have highest importance (weight 0.8 + 2 connections)
        self.assertGreater(scores[2], scores[1])  # node3 > node2

class TestAgenticCognitiveKernel(unittest.TestCase):
    """Test the main agentic cognitive kernel"""
    
    def setUp(self):
        """Set up test kernel"""
        self.kernel = AgenticCognitiveKernel(
            reservoir_size=20,
            input_dim=10,
            attention_dim=5
        )
    
    def test_kernel_initialization(self):
        """Test kernel initialization"""
        self.assertEqual(self.kernel.reservoir_size, 20)
        self.assertEqual(self.kernel.input_dim, 10)
        self.assertEqual(self.kernel.attention_dim, 5)
        self.assertEqual(len(self.kernel.cognitive_log), 0)
    
    def test_process_symbols(self):
        """Test symbol processing"""
        symbols = ['start', 'process', 'transform', 'end']
        result = self.kernel.process_symbols(symbols)
        
        # Check result structure
        self.assertIn('esn_state', result)
        self.assertIn('grammar', result)
        self.assertIn('attention', result)
        self.assertIn('log_entry', result)
        
        # Check that log was updated
        self.assertEqual(len(self.kernel.cognitive_log), 1)
        
        # Check log entry content
        log_entry = result['log_entry']
        self.assertEqual(log_entry['input_symbols'], symbols)
        self.assertEqual(len(log_entry['esn_activation']), 20)
        self.assertGreater(log_entry['grammar_complexity'], 0)
    
    def test_cognitive_status(self):
        """Test cognitive status reporting"""
        # Process some symbols first
        self.kernel.process_symbols(['start', 'end'])
        
        status = self.kernel.get_cognitive_status()
        
        self.assertIn('log_entries', status)
        self.assertIn('grammar_complexity', status)
        self.assertIn('pattern_count', status)
        self.assertIn('reservoir_size', status)
        
        self.assertEqual(status['log_entries'], 1)
        self.assertEqual(status['reservoir_size'], 20)
    
    def test_kernel_reset(self):
        """Test kernel reset functionality"""
        # Process symbols and check state
        self.kernel.process_symbols(['start', 'process'])
        self.assertGreater(len(self.kernel.cognitive_log), 0)
        
        # Reset and check
        self.kernel.reset()
        self.assertEqual(len(self.kernel.cognitive_log), 0)
        self.assertEqual(self.kernel.hypergraph.grammar.complexity, 0.0)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_benchmark_execution(self):
        """Test benchmark execution"""
        kernel = AgenticCognitiveKernel(reservoir_size=10, input_dim=5, attention_dim=3)
        test_sequences = generate_test_sequences(5, 3)
        
        result = run_cognitive_benchmark(kernel, test_sequences)
        
        # Check benchmark result structure
        self.assertIn('execution_time', result)
        self.assertIn('sequences_processed', result)
        self.assertIn('complexity_evolution', result)
        self.assertIn('attention_stability', result)
        self.assertIn('final_status', result)
        
        # Check values
        self.assertEqual(result['sequences_processed'], 5)
        self.assertGreater(result['execution_time'], 0)
        self.assertEqual(len(result['complexity_evolution']), 5)
    
    def test_multiple_processing_cycles(self):
        """Test multiple symbol processing cycles"""
        kernel = AgenticCognitiveKernel(reservoir_size=15, input_dim=8, attention_dim=4)
        
        sequences = [
            ['start', 'symbol'],
            ['process', 'tensor'],
            ['transform', 'grammar'],
            ['end', 'attention']
        ]
        
        results = []
        for sequence in sequences:
            result = kernel.process_symbols(sequence)
            results.append(result)
        
        # Check that complexity generally increases (allow for small variations)
        complexities = [r['grammar'].complexity for r in results]
        self.assertGreaterEqual(complexities[-1], complexities[0])
        
        # Check that attention patterns evolve
        attentions = [r['attention'] for r in results]
        self.assertEqual(len(attentions), 4)
        
        # Check log accumulation
        self.assertEqual(len(kernel.cognitive_log), 4)

def run_tests():
    """Run all tests and display results"""
    print("Running Agentic Cognitive Kernel Test Suite")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestGGMLTensor,
        TestEchoGGMLNetwork,
        TestHypergraphGrammar,
        TestECANAttentionAllocator,
        TestAgenticCognitiveKernel,
        TestIntegration
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)