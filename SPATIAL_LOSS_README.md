# Spatial-Aware Loss Function for ARC Grid Generation

## Problem Statement

The `compression_vitarc_gumbel` model was predicting smaller grids than the actual targets, particularly in sections where it was confident. This occurred because:

1. **Token-based representation**: Grids are converted to token sequences using `<arc_nl>` as row separators
2. **Cross-entropy incentive**: The model discovered it could minimize loss by generating fewer tokens (smaller grids)
3. **No spatial understanding**: The original loss function treated grid generation as pure sequence modeling

## Solution: Spatial-Aware Loss Function

### Key Components

#### 1. `tokens_to_grid_array()` Function
- Converts token sequences back to 2D numpy arrays during loss computation
- Handles both `<arc_nl>`-separated grids and flat sequences
- Robust error handling and fallback mechanisms

#### 2. `spatial_reconstruction_loss()` Function
- **Size Penalty**: Heavily penalizes `(pred_height - target_height)² + (pred_width - target_width)²`
- **Content Loss**: MSE on overlapping region between predicted and target grids
- **Missing Content Penalty**: Additional penalty when target is larger than prediction
- **Per-batch Processing**: Handles batched inputs correctly

#### 3. Modified `compute_loss()` Function
- New parameters:
  - `tokenizer`: Required for grid conversion
  - `w_spatial`: Weight for spatial penalties (default: 1.0, recommended: 10.0)
  - `use_spatial_loss`: Toggle between spatial and regular cross-entropy loss
- Backward compatibility with existing code

### Usage

```python
# In training loop (run_example.py)
loss, metrics = compute_loss(
    outputs_train, labels_train, aux_train, 
    tokenizer=tokenizer, 
    w_entropy=0.0, 
    w_spatial=10.0,  # Strong spatial penalty
    use_spatial_loss=True
)
```

### Test Results

Our test case demonstrates the effectiveness:
- **Scenario**: Model predicts 2x2 grid when target is 3x3
- **Regular loss**: 8.18
- **Spatial-aware loss**: 20.50 (**2.5x higher penalty**)

This strong penalty incentivizes the model to predict correctly-sized grids.

## Implementation Details

### Grid Parsing Logic
1. **Primary**: Parse `<arc_nl>`-separated rows with `<arc_X>` tokens
2. **Fallback**: Reshape flat sequences into square-ish grids
3. **Error Handling**: Return minimal 1x1 grid on parsing errors

### Loss Computation
```
total_loss = content_loss + (w_spatial × size_penalty) + missing_penalty

where:
- size_penalty = (Δheight² + Δwidth²)
- content_loss = MSE on overlapping region
- missing_penalty = 0.1 × missing_cells
```

### Integration Points
- Modified `models/vit_arc/train_step.py`
- Updated `run_example.py` training loop
- Created `test_spatial_loss.py` verification

## Expected Impact

1. **Reduced size mismatch**: Model will be heavily penalized for predicting wrong grid dimensions
2. **Improved content accuracy**: Content loss on overlapping regions maintains quality
3. **Better spatial understanding**: Model learns 2D structure rather than pure sequence patterns

## Monitoring

The loss includes new metrics:
- `loss_spatial`: Size and content penalties combined
- `loss_rec`: Reconstruction component
- Standard metrics: `kl_uniform`, `entropy`, `temperature`

Track these during training to verify the model is learning proper grid sizing.