#!/usr/bin/env python3
"""
Analyze tokenization sizes at different stages of the tetrominoes dataset processing.
"""

import numpy as np
from dataset.tetrominoes_generation import LTetromino, TetrominoConfig
from tokenization.arc_tokenizer import get_or_build_arc_tokenizer
from dataset.gen_dataset import generate_io_pairs, reformat_arc_tokens, replace_digits_with_arc, replace_digits_with_arc

def analyze_tokenization_sizes():
    # Initialize components
    config = TetrominoConfig(grid_size=6)
    tetromino_generator = LTetromino(config)
    tokenizer = get_or_build_arc_tokenizer()
    
    # Generate a sample
    example = generate_io_pairs(tetromino_generator, num_operations=3)
    
    print("="*80)
    print("TOKENIZATION SIZE ANALYSIS")
    print("="*80)
    
    # Step 1: Raw grids (before any processing)
    print("\n1. RAW GRIDS (before tokenization):")
    print(f"   Input1 grid: {len(example['input1'])}x{len(example['input1'][0])} = {len(example['input1']) * len(example['input1'][0])} elements")
    print(f"   Output1 grid: {len(example['output1'])}x{len(example['output1'][0])} = {len(example['output1']) * len(example['output1'][0])} elements")
    print(f"   Input2 grid: {len(example['input2'])}x{len(example['input2'][0])} = {len(example['input2']) * len(example['input2'][0])} elements")
    print(f"   Output2 grid: {len(example['output2'])}x{len(example['output2'][0])} = {len(example['output2']) * len(example['output2'][0])} elements")
    print(f"   Total raw elements: {4 * 6 * 6} = 144 elements")
    
    # Step 2: Convert to ARC tokens (string format)
    input1_str = reformat_arc_tokens(replace_digits_with_arc(example["input1"]))
    output1_str = reformat_arc_tokens(replace_digits_with_arc(example["output1"]))
    input2_str = reformat_arc_tokens(replace_digits_with_arc(example["input2"]))
    output2_str = reformat_arc_tokens(replace_digits_with_arc(example["output2"]))
    
    print(f"\n2. ARC TOKEN STRINGS (before tokenization):")
    print(f"   Input1 string length: {len(input1_str)} chars")
    print(f"   Output1 string length: {len(output1_str)} chars")
    print(f"   Input2 string length: {len(input2_str)} chars")
    print(f"   Output2 string length: {len(output2_str)} chars")
    print(f"   Input1 sample: '{input1_str[:50]}...'")
    print(f"   Output2 sample: '{output2_str[:50]}...'")
    
    # Step 3: Combined strings with special tokens
    bos = tokenizer.bos_token
    eos = tokenizer.eos_token
    sep = tokenizer.sep_token
    
    model_input_str = (
        f"{bos}{input1_str}{eos}"
        f"{sep}"
        f"{bos}{output1_str}{eos}"
        f"{sep}"
        f"{bos}{input2_str}{eos}"
    )
    model_target_str = f"{bos}{output2_str}{eos}"
    
    print(f"\n3. COMBINED STRINGS (with special tokens, before tokenization):")
    print(f"   Model input string length: {len(model_input_str)} chars")
    print(f"   Model target string length: {len(model_target_str)} chars")
    print(f"   Special tokens used: BOS='{bos}', EOS='{eos}', SEP='{sep}'")
    
    # Step 4: After tokenization (before padding)
    input_tokens_raw = tokenizer.tokenize(model_input_str)
    target_tokens_raw = tokenizer.tokenize(model_target_str)
    
    print(f"\n4. AFTER TOKENIZATION (before padding):")
    print(f"   Input tokens count: {len(input_tokens_raw)}")
    print(f"   Target tokens count: {len(target_tokens_raw)}")
    print(f"   Input tokens sample: {input_tokens_raw[:15]}...")
    print(f"   Target tokens sample: {target_tokens_raw[:10]}...")
    
    # Step 5: After tokenization with IDs (before padding)
    input_ids_raw = tokenizer.encode(model_input_str, add_special_tokens=False)
    target_ids_raw = tokenizer.encode(model_target_str, add_special_tokens=False)
    
    print(f"\n5. TOKEN IDs (before padding):")
    print(f"   Input token IDs count: {len(input_ids_raw)}")
    print(f"   Target token IDs count: {len(target_ids_raw)}")
    print(f"   Input IDs sample: {input_ids_raw[:15]}...")
    print(f"   Target IDs sample: {target_ids_raw[:10]}...")
    
    # Step 6: After padding (final form)
    MAX_INPUT_LENGTH = 200
    MAX_TARGET_LENGTH = 50
    
    tokenized_input = tokenizer(
        model_input_str,
        max_length=MAX_INPUT_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    tokenized_target = tokenizer(
        model_target_str,
        max_length=MAX_TARGET_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors=None
    )
    
    print(f"\n6. AFTER PADDING (final form):")
    print(f"   Input token IDs length: {len(tokenized_input.input_ids)} (padded to {MAX_INPUT_LENGTH})")
    print(f"   Target token IDs length: {len(tokenized_target.input_ids)} (padded to {MAX_TARGET_LENGTH})")
    print(f"   Input attention mask length: {len(tokenized_input.attention_mask)}")
    print(f"   Non-padding tokens in input: {sum(tokenized_input.attention_mask)}")
    print(f"   Non-padding tokens in target: {sum(1 for x in tokenized_target.input_ids if x != tokenizer.pad_token_id)}")
    
    # Summary table
    print(f"\n" + "="*80)
    print("SUMMARY TABLE:")
    print("="*80)
    print(f"{'Stage':<30} {'Input Size':<15} {'Target Size':<15} {'Notes'}")
    print("-" * 80)
    print(f"{'Raw grids':<30} {'144 elements':<15} {'36 elements':<15} {'6x6 grids'}")
    print(f"{'ARC token strings':<30} {f'{len(input1_str + output1_str + input2_str)} chars':<15} {f'{len(output2_str)} chars':<15} {'String format'}")
    print(f"{'Combined strings':<30} {f'{len(model_input_str)} chars':<15} {f'{len(model_target_str)} chars':<15} {'With special tokens'}")
    print(f"{'After tokenization':<30} {f'{len(input_tokens_raw)} tokens':<15} {f'{len(target_tokens_raw)} tokens':<15} {'Before padding'}")
    print(f"{'After padding':<30} {f'{MAX_INPUT_LENGTH} tokens':<15} {f'{MAX_TARGET_LENGTH} tokens':<15} {'Final form'}")
    
    # Show actual utilization
    input_utilization = (sum(tokenized_input.attention_mask) / MAX_INPUT_LENGTH) * 100
    target_utilization = (sum(1 for x in tokenized_target.input_ids if x != tokenizer.pad_token_id) / MAX_TARGET_LENGTH) * 100
    
    print(f"\nPADDING UTILIZATION:")
    print(f"   Input utilization: {input_utilization:.1f}% ({sum(tokenized_input.attention_mask)}/{MAX_INPUT_LENGTH})")
    print(f"   Target utilization: {target_utilization:.1f}% ({sum(1 for x in tokenized_target.input_ids if x != tokenizer.pad_token_id)}/{MAX_TARGET_LENGTH})")

if __name__ == "__main__":
    analyze_tokenization_sizes()