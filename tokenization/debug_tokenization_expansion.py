#!/usr/bin/env python3
"""
Debug why 144 elements become 3,374 tokens - analyze the tokenization expansion.
"""

import numpy as np
from ..dataset.tetrominoes_generation import LTetromino, TetrominoConfig
from .arc_tokenizer import get_or_build_arc_tokenizer
from ..dataset.gen_tetrominoes_dataset import generate_io_pairs
from ..dataset.gen_dataset import reformat_arc_tokens, replace_digits_with_arc

def debug_tokenization_expansion():
    # Initialize components
    config = TetrominoConfig(grid_size=6)
    tetromino_generator = LTetromino(config)
    tokenizer = get_or_build_arc_tokenizer()
    
    # Generate a simple example
    example = generate_io_pairs(tetromino_generator, num_operations=1)  # Just 1 operation for simplicity
    
    print("="*80)
    print("DEBUG: WHY 144 ELEMENTS → 3,374 TOKENS?")
    print("="*80)
    
    # Step 1: Look at a single grid transformation
    print("\n1. SINGLE GRID ANALYSIS:")
    input1 = np.array(example['input1'])
    print(f"   Raw grid shape: {input1.shape}")
    print(f"   Raw grid (6x6 = 36 elements):")
    for row in input1:
        print(f"   {row}")
    
    # Step 2: See what replace_digits_with_arc does
    print(f"\n2. AFTER replace_digits_with_arc:")
    arc_version = replace_digits_with_arc(example["input1"])
    print(f"   Type: {type(arc_version)}")
    print(f"   Shape if array: {np.array(arc_version).shape if hasattr(arc_version, '__len__') else 'N/A'}")
    print(f"   Sample row: {arc_version[0] if hasattr(arc_version, '__getitem__') else arc_version}")
    
    # Step 3: See what reformat_arc_tokens does
    print(f"\n3. AFTER reformat_arc_tokens:")
    formatted = reformat_arc_tokens(arc_version)
    print(f"   String length: {len(formatted)} characters")
    print(f"   First 200 chars: '{formatted[:200]}'")
    print(f"   Last 100 chars: '{formatted[-100:]}'")
    
    # Step 4: Count specific patterns
    print(f"\n4. PATTERN ANALYSIS:")
    arc_0_count = formatted.count('<arc_0>')
    arc_1_count = formatted.count('<arc_1>')
    arc_end_count = formatted.count('<arc_end')
    arc_pad_count = formatted.count('<arc_pad>')
    
    print(f"   <arc_0> occurrences: {arc_0_count}")
    print(f"   <arc_1> occurrences: {arc_1_count}")  
    print(f"   <arc_end*> occurrences: {arc_end_count}")
    print(f"   <arc_pad> occurrences: {arc_pad_count}")
    print(f"   Total major tokens: {arc_0_count + arc_1_count + arc_end_count + arc_pad_count}")
    
    # Step 5: Tokenize and see the explosion
    print(f"\n5. TOKENIZATION EXPLOSION:")
    tokens = tokenizer.tokenize(formatted)
    print(f"   Number of tokens: {len(tokens)}")
    print(f"   Expansion ratio: {len(tokens) / 36:.1f}x (tokens per grid element)")
    print(f"   First 20 tokens: {tokens[:20]}")
    
    # Step 6: Analyze what each grid element becomes
    print(f"\n6. PER-ELEMENT BREAKDOWN:")
    # Let's manually trace a few elements
    print("   Looking at how individual grid values are tokenized:")
    
    # Test individual values
    test_cases = [0, 1, 2, 7]
    for val in test_cases:
        if val <= np.max(input1):
            single_val_list = [[val]]
            single_arc = replace_digits_with_arc(single_val_list)
            single_formatted = reformat_arc_tokens(single_arc)
            single_tokens = tokenizer.tokenize(single_formatted)
            print(f"   Value {val} → '{single_formatted[:50]}...' → {len(single_tokens)} tokens")
    
    # Step 7: Full pipeline analysis
    print(f"\n7. FULL PIPELINE (3 inputs + 1 target):")
    input1_str = reformat_arc_tokens(replace_digits_with_arc(example["input1"]))
    output1_str = reformat_arc_tokens(replace_digits_with_arc(example["output1"]))
    input2_str = reformat_arc_tokens(replace_digits_with_arc(example["input2"]))
    output2_str = reformat_arc_tokens(replace_digits_with_arc(example["output2"]))
    
    print(f"   Input1 string: {len(input1_str)} chars → {len(tokenizer.tokenize(input1_str))} tokens")
    print(f"   Output1 string: {len(output1_str)} chars → {len(tokenizer.tokenize(output1_str))} tokens")  
    print(f"   Input2 string: {len(input2_str)} chars → {len(tokenizer.tokenize(input2_str))} tokens")
    print(f"   Output2 string: {len(output2_str)} chars → {len(tokenizer.tokenize(output2_str))} tokens")
    
    combined_tokens = (len(tokenizer.tokenize(input1_str)) + 
                      len(tokenizer.tokenize(output1_str)) + 
                      len(tokenizer.tokenize(input2_str)))
    print(f"   Combined input tokens: {combined_tokens}")
    print(f"   Target tokens: {len(tokenizer.tokenize(output2_str))}")
    
    # Step 8: Identify the efficiency problem
    elements_per_grid = 36
    total_elements = 4 * elements_per_grid  # 144
    tokens_per_element = combined_tokens / (3 * elements_per_grid)  # 3 grids in input
    
    print(f"\n8. EFFICIENCY ANALYSIS:")
    print(f"   Total raw elements: {total_elements}")
    print(f"   Input tokens per grid element: {tokens_per_element:.1f}")
    print(f"   This means each 0 or 1 becomes ~{tokens_per_element:.0f} tokens!")
    print(f"   Root cause: ARC tokenization is designed for complex multi-color grids,")
    print(f"              but tetrominoes only use 0/1 values - massive overkill!")

if __name__ == "__main__":
    debug_tokenization_expansion()