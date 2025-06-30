#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# File: integration_demo.py
# Description: Complete integration demonstration of the agentic cognitive kernel
# Author: RCNLP Agentic System
# Date: 2024

import sys
import subprocess
import numpy as np
from echo_ggml_kernel import (
    AgenticCognitiveKernel,
    run_cognitive_benchmark,
    generate_test_sequences
)

def run_scheme_demo():
    """Run the Scheme cognitive kernel demonstration"""
    print("=" * 60)
    print("SCHEME COGNITIVE KERNEL DEMONSTRATION")
    print("=" * 60)
    
    try:
        result = subprocess.run([
            'guile', 'agentic-cognitive-kernels-simple.scm'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print(result.stdout)
            return True
        else:
            print(f"Scheme execution failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Guile not available - skipping Scheme demonstration")
        return False

def run_python_demo():
    """Run the Python cognitive kernel demonstration"""
    print("=" * 60)
    print("PYTHON ECHO-GGML KERNEL DEMONSTRATION")
    print("=" * 60)
    
    # Initialize kernel with diverse parameters
    kernel = AgenticCognitiveKernel(
        reservoir_size=75,
        input_dim=30,
        attention_dim=15
    )
    
    # Test sequences representing different cognitive tasks
    cognitive_tasks = [
        ['start', 'symbol', 'process', 'end'],           # Symbol processing
        ['tensor', 'transform', 'attention', 'focus'],   # Tensor operations
        ['grammar', 'parse', 'generate', 'output'],      # Grammar processing
        ['memory', 'recall', 'associate', 'store'],      # Memory operations
        ['learn', 'adapt', 'evolve', 'optimize']         # Learning dynamics
    ]
    
    print(f"Processing {len(cognitive_tasks)} cognitive task sequences...")
    print()
    
    results = []
    for i, task in enumerate(cognitive_tasks):
        print(f"Task {i+1}: {' -> '.join(task)}")
        result = kernel.process_symbols(task)
        results.append(result)
        
        # Display key metrics
        esn_activation = result['esn_state'].activation
        grammar_complexity = result['grammar'].complexity
        attention_sum = np.sum(result['attention'])
        
        print(f"  ESN activation (mean): {np.mean(esn_activation):.3f}")
        print(f"  Grammar complexity: {grammar_complexity:.3f}")
        print(f"  Attention allocation sum: {attention_sum:.3f}")
        print()
    
    return results, kernel

def run_benchmark_demo(kernel):
    """Run comprehensive benchmark demonstration"""
    print("=" * 60)
    print("COGNITIVE SYNERGY BENCHMARK")
    print("=" * 60)
    
    # Generate diverse test sequences
    test_sequences = []
    
    # Add structured sequences
    structured_sequences = [
        ['start'] + ['process'] * i + ['end'] for i in range(1, 6)
    ]
    test_sequences.extend(structured_sequences)
    
    # Add random sequences
    random_sequences = generate_test_sequences(count=15, length=4)
    test_sequences.extend(random_sequences)
    
    print(f"Running benchmark on {len(test_sequences)} sequences...")
    
    # Run benchmark
    benchmark_result = run_cognitive_benchmark(kernel, test_sequences)
    
    # Display comprehensive results
    print(f"\nBenchmark Results:")
    print(f"  Total sequences: {benchmark_result['sequences_processed']}")
    print(f"  Execution time: {benchmark_result['execution_time']:.4f}s")
    print(f"  Average time per sequence: {benchmark_result['execution_time']/len(test_sequences):.6f}s")
    print(f"  Attention stability: {benchmark_result['attention_stability']:.6f}")
    
    # Complexity evolution analysis
    complexity_evolution = benchmark_result['complexity_evolution']
    print(f"  Initial complexity: {complexity_evolution[0]:.3f}")
    print(f"  Final complexity: {complexity_evolution[-1]:.3f}")
    print(f"  Complexity growth: {complexity_evolution[-1] - complexity_evolution[0]:.3f}")
    
    # Final system status
    final_status = benchmark_result['final_status']
    print(f"\nFinal System Status:")
    for key, value in final_status.items():
        print(f"  {key}: {value}")
    
    return benchmark_result

def demonstrate_attention_dynamics(kernel):
    """Demonstrate attention allocation dynamics"""
    print("=" * 60)
    print("ATTENTION DYNAMICS DEMONSTRATION")
    print("=" * 60)
    
    # Test attention response to different input patterns
    attention_tests = [
        ('High activation', ['attention', 'focus', 'concentrate', 'intense']),
        ('Low activation', ['rest', 'idle', 'quiet', 'calm']),
        ('Mixed patterns', ['start', 'attention', 'rest', 'focus']),
        ('Recursive patterns', ['process', 'process', 'process', 'process'])
    ]
    
    attention_history = []
    
    for test_name, symbols in attention_tests:
        print(f"\n{test_name}: {' -> '.join(symbols)}")
        result = kernel.process_symbols(symbols)
        attention = result['attention']
        attention_history.append(attention)
        
        print(f"  Attention allocation: {attention[:4]}")  # Show first 4 values
        print(f"  Max attention: {np.max(attention):.3f}")
        print(f"  Min attention: {np.min(attention):.3f}")
        print(f"  Attention variance: {np.var(attention):.6f}")
    
    # Analyze attention stability
    if len(attention_history) > 1:
        print(f"\nAttention Stability Analysis:")
        for i in range(1, len(attention_history)):
            correlation = np.corrcoef(attention_history[i-1], attention_history[i])[0, 1]
            print(f"  Test {i} vs {i+1}: {correlation:.6f}")

def main():
    """Main demonstration orchestrator"""
    print("RCNLP AGENTIC COGNITIVE KERNEL")
    print("Comprehensive System Demonstration")
    print("Neural-Symbolic Integration with Dynamic Attention")
    print("=" * 80)
    
    # Run Scheme demonstration
    scheme_success = run_scheme_demo()
    
    # Run Python demonstration
    results, kernel = run_python_demo()
    
    # Run attention dynamics demonstration
    demonstrate_attention_dynamics(kernel)
    
    # Run comprehensive benchmark
    benchmark_result = run_benchmark_demo(kernel)
    
    # Final summary
    print("=" * 80)
    print("DEMONSTRATION SUMMARY")
    print("=" * 80)
    
    print(f"✓ Scheme kernel: {'SUCCESS' if scheme_success else 'SKIPPED'}")
    print(f"✓ Python kernel: SUCCESS")
    print(f"✓ Cognitive tasks processed: {len(results)}")
    print(f"✓ Benchmark sequences: {benchmark_result['sequences_processed']}")
    print(f"✓ Total execution time: {benchmark_result['execution_time']:.4f}s")
    print(f"✓ Final system complexity: {benchmark_result['complexity_evolution'][-1]:.3f}")
    
    print("\nThe agentic cognitive kernel demonstrates:")
    print("  • Neural-symbolic integration through ESN + hypergraph grammars")
    print("  • ECAN-style adaptive attention allocation")
    print("  • Real-time cognitive state evolution and logging")
    print("  • P-System inspired frame problem resolution")
    print("  • Scheme-Python hybrid cognitive architecture")
    
    print("\n🎭 \"Each kernel a membrane, each symbol a recursive note")
    print("    in the grand cognitive orchestra of agentic computation.\" 🎭")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)