"""
Example usage of HOPE architecture.

This script demonstrates:
1. Creating a HOPE model
2. Forward pass with memory management
3. Text generation
4. Using different optimizers
5. Continuum Memory System

Reference: Nested Learning paper
"""

import torch
import torch.nn as nn

from src.config import HopeConfig, HopeSmallConfig, HopeBaseConfig
from src.model import Hope, HopeForCausalLM, createHopeModel
from src.modules.titans import SelfModifyingTitans
from src.modules.continuum_memory import ContinuumMemorySystem
from src.optimizers import AdamWithDeltaRule


def exampleBasicUsage():
    """Basic model usage example."""
    print("\n" + "=" * 50)
    print("Example 1: Basic Model Usage")
    print("=" * 50)

    # Create a small HOPE model
    config = HopeSmallConfig(
        vocab_size=10000,
        max_seq_len=512,
    )

    model = HopeForCausalLM(config)
    print(f"Model created with {model.getNumParams():,} parameters")

    # Create sample input
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    outputs = model(input_ids=input_ids, labels=input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")


def exampleMemoryManagement():
    """Example of memory state management across sequences."""
    print("\n" + "=" * 50)
    print("Example 2: Memory State Management")
    print("=" * 50)

    config = HopeSmallConfig(vocab_size=10000)
    model = Hope(config)

    # Process multiple sequences while maintaining memory
    memory_states = None

    for i in range(3):
        input_ids = torch.randint(0, config.vocab_size, (1, 32))

        # Forward pass with memory
        logits, memory_states = model(
            input_ids,
            memory_states=memory_states,
            return_memory=True,
        )

        # Check memory state
        if memory_states[0] is not None:
            mem_norm = memory_states[0].norm().item()
            print(f"Sequence {i+1}: Memory norm = {mem_norm:.4f}")

    print("Memory persists across sequences!")


def exampleTextGeneration():
    """Example of text generation with HOPE."""
    print("\n" + "=" * 50)
    print("Example 3: Text Generation")
    print("=" * 50)

    config = HopeSmallConfig(vocab_size=10000)
    model = Hope(config)

    # Create a prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10))
    print(f"Prompt tokens: {prompt[0].tolist()}")

    # Generate with different sampling strategies
    # Greedy (temperature close to 0)
    generated_greedy = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.1,
    )
    print(f"Greedy generation: {generated_greedy[0, 10:].tolist()}")

    # Sampling with temperature
    generated_sample = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=50,
    )
    print(f"Sampled generation: {generated_sample[0, 10:].tolist()}")

    # Nucleus sampling
    generated_nucleus = model.generate(
        prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_p=0.9,
    )
    print(f"Nucleus generation: {generated_nucleus[0, 10:].tolist()}")


def exampleCustomOptimizer():
    """Example using custom NL-based optimizer."""
    print("\n" + "=" * 50)
    print("Example 4: Custom Optimizer (Adam with Delta Rule)")
    print("=" * 50)

    config = HopeSmallConfig(vocab_size=1000)
    model = HopeForCausalLM(config)

    # Use Adam with delta rule (from Section 2.3)
    optimizer = AdamWithDeltaRule(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        delta_beta=0.01,  # Delta rule forgetting factor
        weight_decay=0.01,
    )

    # Training step
    input_ids = torch.randint(0, config.vocab_size, (2, 64))

    for step in range(5):
        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs["loss"]

        loss.backward()
        optimizer.step()

        print(f"Step {step+1}: Loss = {loss.item():.4f}")


def exampleContinuumMemory():
    """Example of Continuum Memory System."""
    print("\n" + "=" * 50)
    print("Example 5: Continuum Memory System")
    print("=" * 50)

    # Create CMS with multiple frequency levels
    cms = ContinuumMemorySystem(
        dim=256,
        num_levels=4,
        chunk_sizes=[16, 256, 4096, 65536],  # Different update frequencies
        expansion=4,
    )

    print("CMS Configuration:")
    for i, chunk_size in enumerate(cms.chunk_sizes):
        print(f"  Level {i}: Updates every {chunk_size} steps")

    # Forward pass
    x = torch.randn(1, 64, 256)
    output = cms(x, enable_online_learning=False)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # Show update schedule
    schedule = cms.getUpdateSchedule(100)
    print(f"\nUpdate schedule (first 100 steps):")
    for step, levels in list(schedule.items())[:10]:
        print(f"  Step {step}: Update levels {levels}")


def exampleSelfModifyingTitans():
    """Example of Self-Modifying Titans module."""
    print("\n" + "=" * 50)
    print("Example 6: Self-Modifying Titans")
    print("=" * 50)

    titans = SelfModifyingTitans(
        dim=256,
        head_dim=64,
        num_heads=4,
        learning_rate=0.1,
        use_delta_rule=True,  # Use delta rule from Eq. 28-29
    )

    print("Titans Configuration:")
    print(f"  Dimension: {titans.dim}")
    print(f"  Heads: {titans.num_heads}")
    print(f"  Head dim: {titans.head_dim}")
    print(f"  Delta rule: {titans.use_delta_rule}")

    # Process sequence
    x = torch.randn(1, 32, 256)
    memory_state = None

    print("\nProcessing sequences:")
    for i in range(3):
        output, memory_state = titans(x, memory_state=memory_state, return_memory=True)

        mem_norm = memory_state.norm().item()
        out_norm = output.norm().item()
        print(f"  Iteration {i+1}: Output norm = {out_norm:.2f}, Memory norm = {mem_norm:.2f}")


def exampleFactoryFunction():
    """Example using factory function to create models."""
    print("\n" + "=" * 50)
    print("Example 7: Model Factory Function")
    print("=" * 50)

    sizes = ["small", "base", "large"]

    for size in sizes:
        model = createHopeModel(model_size=size, vocab_size=10000)
        num_params = model.getNumParams()
        print(f"  {size.capitalize():6s}: {num_params:>12,} parameters")


def main():
    print("HOPE Architecture Examples")
    print("Based on: Nested Learning - The Illusion of Deep Learning Architectures")

    # Run all examples
    exampleBasicUsage()
    exampleMemoryManagement()
    exampleTextGeneration()
    exampleCustomOptimizer()
    exampleContinuumMemory()
    exampleSelfModifyingTitans()
    exampleFactoryFunction()

    print("\n" + "=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
