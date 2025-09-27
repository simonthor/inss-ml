import numpy as np
import torch
def compute_validation_output(model, val_truth, val_input_feats, val_input_coords, val_batch_size=10):
    """Compute the model output for the validation data"""
    val_n_voxels = [len(e) for e in val_truth]
    val_n_voxels_cumsum = np.cumsum(val_n_voxels)

    preds = []

    for i, batch in enumerate(range(0, len(val_truth), val_batch_size)):
        voxel_i_lower = 0 if batch == 0 else val_n_voxels_cumsum[batch - 1]
        voxel_i_upper = val_n_voxels_cumsum[batch + val_batch_size-1] if (batch + val_batch_size-1) < len(val_truth) else val_n_voxels_cumsum[-1]
        batch_input_feats = val_input_feats[voxel_i_lower:voxel_i_upper]
        batch_input_coords = val_input_coords[voxel_i_lower:voxel_i_upper].detach().clone()
        
        print(batch_input_coords[:, 0].min().item(), batch_input_coords[:, 0].max().item())
        # From batch_input_coords, subtract the batch number to make indices start from 0
        batch_input_coords[:, 0] = batch_input_coords[:, 0] - batch
        # Defensive: skip if batch_input_coords is empty
        if batch_input_coords.shape[0] == 0:
            print("Empty batch. Skip")
            continue

        # Defensive: check max batch index only if not empty
        # if batch_input_coords.shape[0] > 0:
            # assert batch_input_coords[:, 0].max().item() == val_batch_size - 1, batch_input_coords[:, 0].max().item()
            # assert batch_input_coords[:, 0].min().item() == 0, batch_input_coords[:, 0].min().item()
        
        model.eval()
        with torch.no_grad():
            output_sparse = model(batch_input_feats, batch_input_coords, batch_size=val_batch_size if batch + val_batch_size < len(data) else len(data) - batch)

            preds.append(output_sparse.features)

    preds = torch.cat(preds, dim=0)
    
    return preds