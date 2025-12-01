"""
Test script for HOPE architecture.

Verifies:
1. Model creation and forward pass
2. Delta rule implementation (Eq. 28-29)
3. Memory state management
4. Continuum Memory System
5. Optimizers

Run: python test_hope.py
"""

import torch
import torch.nn as nn
import math
from typing import Optional

# Import HOPE components
from hope.config import HopeConfig, HopeSmallConfig
from hope.model import Hope, HopeForCausalLM, createHopeModel
from hope.modules.titans import SelfModifyingTitans, SelfModifyingTitansChunk
from hope.modules.continuum_memory import ContinuumMemorySystem, FrequencyFFN
from hope.modules.hope_block import HopeBlock
from hope.layers.associative_memory import DeltaRuleMemory, LinearAttentionMemory
from hope.layers.neural_memory import FastNeuralMemory, NeuralMemory
from hope.optimizers import (
    DeepMomentumGD,
    DeltaRuleOptimizer,
    AdamWithDeltaRule,
    MuonOptimizer,
)


def testDeltaRuleMemory():
    """Test Delta Rule Memory implementation (Eq. 28-29)."""
    print("Testing DeltaRuleMemory...")

    batch_size = 2
    dim_key = 64
    dim_value = 64

    memory_module = DeltaRuleMemory(
        dim_key=dim_key,
        dim_value=dim_value,
        learning_rate=0.1,
    )

    # Initialize memory
    memory = memory_module.initMemory(batch_size, torch.device("cpu"))
    assert memory.shape == (batch_size, dim_value, dim_key)

    # Create test key-value pairs
    key = torch.randn(batch_size, dim_key)
    value = torch.randn(batch_size, dim_value)

    # Update memory
    memory, _ = memory_module.update(memory, key, value)

    # Retrieve using the same key
    retrieved = memory_module.retrieve(memory, key)
    assert retrieved.shape == (batch_size, dim_value)

    # The retrieved value should be close to the stored value
    # (not exact due to delta rule dynamics)
    print(f"  Retrieval error: {(retrieved - value).abs().mean().item():.4f}")

    print("  DeltaRuleMemory: PASSED")


def testSelfModifyingTitans():
    """Test Self-Modifying Titans with delta rule."""
    print("Testing SelfModifyingTitans...")

    batch_size = 2
    seq_len = 32
    dim = 256
    num_heads = 4
    head_dim = 64

    titans = SelfModifyingTitans(
        dim=dim,
        head_dim=head_dim,
        num_heads=num_heads,
        learning_rate=0.1,
        use_delta_rule=True,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output, memory_state = titans(x, memory_state=None, return_memory=True)

    assert output.shape == (batch_size, seq_len, dim)
    assert memory_state.shape == (batch_size, num_heads, head_dim, head_dim)

    # Test with previous memory state
    x2 = torch.randn(batch_size, seq_len, dim)
    output2, memory_state2 = titans(x2, memory_state=memory_state, return_memory=True)

    assert output2.shape == (batch_size, seq_len, dim)

    # Memory should have changed
    memory_diff = (memory_state2 - memory_state).abs().mean().item()
    print(f"  Memory state change: {memory_diff:.4f}")

    print("  SelfModifyingTitans: PASSED")


def testContinuumMemorySystem():
    """Test Continuum Memory System (Eq. 30-31)."""
    print("Testing ContinuumMemorySystem...")

    batch_size = 2
    seq_len = 32
    dim = 256
    num_levels = 3
    chunk_sizes = [4, 16, 64]

    cms = ContinuumMemorySystem(
        dim=dim,
        num_levels=num_levels,
        chunk_sizes=chunk_sizes,
        expansion=4,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output = cms(x, enable_online_learning=False)

    assert output.shape == (batch_size, seq_len, dim)

    # Test with online learning
    cms.resetAccumulators()
    output_online = cms(x, enable_online_learning=True)

    assert output_online.shape == (batch_size, seq_len, dim)

    # Check update schedule
    schedule = cms.getUpdateSchedule(100)
    print(f"  Update schedule (first 100 steps): {len(schedule)} updates")

    print("  ContinuumMemorySystem: PASSED")


def testHopeBlock():
    """Test complete HOPE block."""
    print("Testing HopeBlock...")

    batch_size = 2
    seq_len = 32
    dim = 256

    block = HopeBlock(
        dim=dim,
        head_dim=64,
        num_heads=4,
        num_memory_levels=3,
        chunk_sizes=[4, 16, 64],
        use_delta_rule=True,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output, memory_state = block(x, memory_state=None, return_memory=True)

    assert output.shape == (batch_size, seq_len, dim)
    assert memory_state is not None

    # Test with memory
    x2 = torch.randn(batch_size, seq_len, dim)
    output2, memory_state2 = block(x2, memory_state=memory_state, return_memory=True)

    assert output2.shape == (batch_size, seq_len, dim)

    print("  HopeBlock: PASSED")


def testHopeModel():
    """Test full HOPE model."""
    print("Testing Hope model...")

    config = HopeSmallConfig(vocab_size=1000)
    model = Hope(config)

    batch_size = 2
    seq_len = 64

    # Forward pass
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    logits = model(input_ids)

    assert logits.shape == (batch_size, seq_len, config.vocab_size)

    # Test with memory
    logits, memory_states = model(input_ids, return_memory=True)
    assert len(memory_states) == config.num_layers

    # Test generation
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    assert generated.shape[1] == 30  # 10 prompt + 20 generated

    num_params = model.getNumParams()
    print(f"  Model parameters: {num_params:,}")

    print("  Hope model: PASSED")


def testHopeForCausalLM():
    """Test HOPE for causal language modeling."""
    print("Testing HopeForCausalLM...")

    config = HopeSmallConfig(vocab_size=1000)
    model = HopeForCausalLM(config)

    batch_size = 2
    seq_len = 64

    # Forward pass with labels
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels)

    assert "logits" in outputs
    assert "loss" in outputs
    assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    loss = outputs["loss"]
    print(f"  Loss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    print("  HopeForCausalLM: PASSED")


def testOptimizers():
    """Test custom optimizers."""
    print("Testing optimizers...")

    # Create a simple model
    model = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    target = torch.randn(8, 64)

    optimizers = {
        "DeepMomentumGD": DeepMomentumGD(model.parameters(), lr=0.01, memory_depth=2),
        "DeltaRuleOptimizer": DeltaRuleOptimizer(model.parameters(), lr=0.01),
        "AdamWithDeltaRule": AdamWithDeltaRule(model.parameters(), lr=0.01),
        "MuonOptimizer": MuonOptimizer(model.parameters(), lr=0.01),
    }

    for name, optimizer in optimizers.items():
        # Reset model
        model.reset_parameters()
        optimizer.zero_grad()

        # Forward-backward
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        loss.backward()

        # Step
        optimizer.step()

        print(f"  {name}: OK (loss={loss.item():.4f})")

    print("  Optimizers: PASSED")


def testFastNeuralMemory():
    """Test FastNeuralMemory with delta rule."""
    print("Testing FastNeuralMemory...")

    batch_size = 2
    seq_len = 16
    dim = 128
    num_heads = 2
    head_dim = 64

    memory = FastNeuralMemory(
        dim=dim,
        head_dim=head_dim,
        num_heads=num_heads,
        learning_rate=0.1,
        use_delta_rule=True,
    )

    # Forward pass
    x = torch.randn(batch_size, seq_len, dim)
    output, mem_state, mom_buffer = memory(x)

    assert output.shape == (batch_size, seq_len, dim)
    assert mem_state.shape == (batch_size, num_heads, head_dim, head_dim)

    # Test with previous state
    x2 = torch.randn(batch_size, seq_len, dim)
    output2, mem_state2, mom_buffer2 = memory(x2, mem_state, mom_buffer)

    # Memory should evolve
    mem_diff = (mem_state2 - mem_state).abs().mean().item()
    print(f"  Memory evolution: {mem_diff:.4f}")

    print("  FastNeuralMemory: PASSED")


def testDeltaRuleFormula():
    """
    Verify delta rule formula matches Eq. 28-29.

    M_{t+1} = M_t - M_t * k * k^T - eta * (M_t * k - v) * k^T
    """
    print("Testing Delta Rule formula (Eq. 28-29)...")

    # Manual implementation
    def deltaRuleManual(M, k, v, eta=0.1):
        """Manual delta rule for verification."""
        # Normalize key
        k_norm = k / (k.norm() + 1e-8)

        # Prediction: M @ k
        predicted = M @ k_norm

        # Surprise: M*k - v
        surprise = predicted - v

        # Forgetting term: (M @ k) @ k^T
        forget_term = torch.outer(predicted, k_norm)

        # Learning term: surprise @ k^T
        learn_term = torch.outer(surprise, k_norm)

        # Update: M = M - forget_term - eta * learn_term
        M_new = M - forget_term - eta * learn_term

        return M_new

    # Compare with DeltaRuleMemory
    dim = 32
    memory_module = DeltaRuleMemory(dim_key=dim, dim_value=dim, learning_rate=0.1)

    # Initialize
    M = torch.randn(dim, dim)
    k = torch.randn(dim)
    v = torch.randn(dim)

    # Manual update
    M_manual = deltaRuleManual(M.clone(), k, v, eta=0.1)

    # Module update
    M_batch = M.unsqueeze(0)  # Add batch dim
    k_batch = k.unsqueeze(0)
    v_batch = v.unsqueeze(0)

    M_module, _ = memory_module.update(M_batch, k_batch, v_batch)
    M_module = M_module.squeeze(0)

    # Compare
    diff = (M_manual - M_module).abs().max().item()
    print(f"  Max difference between manual and module: {diff:.6f}")

    assert diff < 1e-5, f"Delta rule mismatch: {diff}"

    print("  Delta Rule formula: PASSED")


def testMemoryPersistence():
    """Test that memory persists across sequences."""
    print("Testing memory persistence...")

    config = HopeSmallConfig(vocab_size=1000)
    model = Hope(config)

    batch_size = 1
    seq_len = 32

    # Process first sequence
    input1 = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    _, memory_states = model(input1, return_memory=True)

    # Process second sequence with previous memory
    input2 = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    _, memory_states2 = model(input2, memory_states=memory_states, return_memory=True)

    # Memory should have evolved
    for i, (m1, m2) in enumerate(zip(memory_states, memory_states2)):
        if m1 is not None and m2 is not None:
            diff = (m2 - m1).abs().mean().item()
            print(f"  Layer {i} memory change: {diff:.4f}")

    print("  Memory persistence: PASSED")


def runAllTests():
    """Run all tests."""
    print("=" * 60)
    print("HOPE Architecture Tests")
    print("=" * 60)

    tests = [
        testDeltaRuleMemory,
        testDeltaRuleFormula,
        testSelfModifyingTitans,
        testContinuumMemorySystem,
        testHopeBlock,
        testHopeModel,
        testHopeForCausalLM,
        testFastNeuralMemory,
        testOptimizers,
        testMemoryPersistence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = runAllTests()
    exit(0 if success else 1)
