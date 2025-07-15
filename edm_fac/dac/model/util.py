import torch
import torch.nn.functional as F
from typing import List


# Repeat ADSR embedding for each onset position
def repeat_adsr_by_onset(adsr_embed, onset_flags):
    """
    Repeat ADSR envelope embedding based on onset positions.
    
    Args:
        adsr_embed: (B, 64, L0) ADSR envelope embedding
        onset_flags: (B, 1, T) Binary flags indicating onset positions (1 = onset)
    
    Returns:
        (B, 64, T) Repeated ADSR embedding for each onset segment
    """
    B, _, L0  = adsr_embed.shape          # L0 = 87 (ADSR envelope length)
    _, _, T   = onset_flags.shape

    # Initialize output tensor
    adsr_expanded = torch.zeros(B, 64, T, device=adsr_embed.device, dtype=adsr_embed.dtype)

    # Create frame indices for all batches
    frame_indices = torch.arange(T, device=onset_flags.device)[None, :].expand(B, -1)  # (B, T)

    # For each batch
    for b in range(B):
        onset_sequence = onset_flags[b, 0]  # (T,)
        onset_positions = torch.where(onset_sequence == 1)[0]

        if len(onset_positions) == 0:
            continue

        # Create segment boundaries
        segment_starts = onset_positions
        segment_ends = torch.cat([onset_positions[1:], torch.tensor([T], device=onset_flags.device)])

        # For each frame, find which segment it belongs to
        segment_idx = torch.zeros(T, dtype=torch.long, device=onset_flags.device)
        for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            segment_idx[start:end] = i

        # Calculate frame position within its segment
        frame_in_segment = frame_indices[b] - segment_starts[segment_idx]

        # Only apply ADSR for frames within the ADSR length
        valid_mask = (frame_in_segment >= 0) & (frame_in_segment < L0)

        # Apply ADSR using advanced indexing
        adsr_expanded[b, :, valid_mask] = adsr_embed[b, :, frame_in_segment[valid_mask]]

    return adsr_expanded


# Build phase grid for resampling
def build_phase_grid(on_idx: List[torch.Tensor],
                     T: int,
                     L: int,
                     device) -> torch.Tensor:
    """
    Build phase grid for resampling ADSR features.
    
    Args:
        on_idx: List of onset indices for each batch
        T: Total time length
        L: Target length for resampling
        device: Device to create tensors on
    
    Returns:
        (B, N_max, L) Phase grid for resampling
    """
    B = len(on_idx)
    N_max = max(len(x) for x in on_idx)
    grid = torch.zeros(B, N_max, L, device=device)

    for b, idx in enumerate(on_idx):
        idx = idx.tolist()
        for n, start in enumerate(idx):
            end = idx[n + 1] if n + 1 < len(idx) else T
            length = max(end - start, 1)
            phase = torch.linspace(0, 1, L, device=device)
            grid[b, n] = start + phase * (length - 1)
    return grid                      # (B, N_max, L)


# Gather note segments with padding
def gather_notes_pad(adsr_feat: torch.Tensor,
                     on_idx: list[list[int]],
                     L: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract note segments from ADSR features and pad to fixed length.
    
    Args:
        adsr_feat: (B, 64, T) ADSR features
        on_idx: List of onset indices for each batch (sorted)
        L: Fixed length for padding
    
    Returns:
        note_E: (B, N_max, 64, L) Padded note segments
        mask: (B, N_max, L) Validity mask (1=valid, 0=padded)
    """
    B, C, T = adsr_feat.shape
    N_max = max(len(seq) for seq in on_idx)
    device = adsr_feat.device

    note_E = torch.zeros(B, N_max, C, L, device=device)
    mask   = torch.zeros(B, N_max, L, device=device)

    # Process each batch
    for b, starts in enumerate(on_idx):
        if len(starts) == 0:
            continue
            
        # Convert to tensors
        starts_tensor = torch.tensor(starts, device=device, dtype=torch.long)
        ends_tensor = torch.cat([
            starts_tensor[1:], 
            torch.tensor([T], device=device, dtype=torch.long)
        ])
        
        # Calculate segment lengths
        seg_lengths = torch.minimum(ends_tensor - starts_tensor, torch.tensor(L, device=device))
        valid_mask = seg_lengths > 0
        
        if not valid_mask.any():
            continue
            
        # Get valid segments
        valid_starts = starts_tensor[valid_mask]
        valid_lengths = seg_lengths[valid_mask]
        valid_note_indices = torch.where(valid_mask)[0]
        
        # Assign segments to output
        for note_idx, start, length in zip(valid_note_indices, valid_starts, valid_lengths):
            end = start + length
            note_E[b, note_idx, :, :length] = adsr_feat[b, :, start:end]
            mask[b, note_idx, :length] = 1.0
            
    return note_E, mask


# Resample 64×T  →  note_E  B×N×64×L (Differentiable)
def resample_adsr(adsr_feat: torch.Tensor,
                  t_grid: torch.Tensor) -> torch.Tensor:
    B, C, T = adsr_feat.shape
    B2, N, L = t_grid.shape
    assert B == B2
    
    # Normalize grid coordinates to [-1, 1]
    grid_x = (t_grid / (T - 1) * 2 - 1).view(B * N, 1, L, 1)
    # Create 2D grid for grid_sample (second dimension is always 0)
    grid_y = torch.zeros_like(grid_x)
    grid = torch.cat([grid_x, grid_y], dim=-1)  # (B*N, 1, L, 2)
    
    # Reshape input for grid_sample
    adsr4d = adsr_feat[:, :, None, :]          # (B,C,1,T)
    adsr4d = adsr4d.expand(-1, -1, N, -1).reshape(B * N, C, 1, T)
    
    sampled = F.grid_sample(adsr4d, grid,
                            mode='bilinear',
                            align_corners=True)      # (B*N,C,1,L)
    note_E = sampled.squeeze(2).view(B, N, C, L)
    return note_E                                    # (B,N,64,L)


# Apply ADSR envelope to onset sequence
def sequencer(proto_E: torch.Tensor,
              onset_flags: torch.Tensor) -> torch.Tensor:
    """
    Apply ADSR envelope prototype to onset sequence.
    
    Args:
        proto_E: (B, 64, L) ADSR envelope prototype
        onset_flags: (B, 1, T) Onset sequence flags
    
    Returns:
        (B, 64, T) ADSR stream applied to onset sequence
    """
    B, C, L = proto_E.shape
    _, _, T = onset_flags.shape

    # Reshape for grouped convolution
    weight = proto_E.view(B * C, 1, L)            # (B*C, 1, L)
    inp    = onset_flags.expand(-1, C, -1)        # (B,64,T)
    inp    = inp.reshape(1, B * C, T)             # (1, B*C, T)

    # Apply grouped convolution
    stream = F.conv1d(inp, weight,
                      groups=B * C,
                      padding=L - 1)              # (1, B*C, T+L-1)
    stream = stream[:, :, :T]                     # Trim to original length

    # Reshape back to batch and channel dimensions
    stream = stream.view(B, C, T)
    return stream


if __name__ == "__main__":

    # Your example onset flags
    onset = torch.tensor([
        1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    onset = onset.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 87)

    # Create dummy ADSR embedding
    tmp_len = 24
    adsr_embed = torch.zeros(1, 64, 87)
    adsr_embed[:, :, 0] = 99 * torch.ones(1, 64)
    adsr_embed[:, :, 1:tmp_len+1] = torch.randn(1, 64, tmp_len)

    print(f"Onset shape: {onset.shape}")
    print(f"ADSR embed shape: {adsr_embed.shape}")
    print(f"Onset positions: {torch.where(onset[0, 0] == 1)[0].tolist()}")

    # Test the function
    result = repeat_adsr_by_onset(adsr_embed, onset)
    print(f"Result shape: {result.shape}")

    # Verify the concept: check that ADSR is applied correctly
    onset_positions = torch.where(onset[0, 0] == 1)[0]
    print(f"\nOnset positions: {onset_positions.tolist()}")

    for i, start_pos in enumerate(onset_positions):
        if i + 1 < len(onset_positions):
            end_pos = onset_positions[i + 1]
        else:
            end_pos = onset.shape[-1]

        segment_length = end_pos - start_pos
        print(f"Segment {i}: position {start_pos} to {end_pos} (length: {segment_length})")

        # Check that the segment is not all zeros (ADSR was applied)
        segment_data = result[0, :, start_pos:end_pos]
        non_zero_count = (segment_data != 0).sum().item()
        print(f"  Non-zero elements in segment: {non_zero_count}/{segment_data.numel()}")

    print(result[:,:,0])
    print(result[:,:,1])
    print(result[:,:,43])
    print(result[:,:,44])