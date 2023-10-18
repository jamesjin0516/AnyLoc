# GeM pooling over Dino (v1) descriptors
"""
    Extract Dino features and do GeM pooling over them to get global
    descriptors
    Currently, there's no caching in GeM.
"""

# %%
import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')


# %%
import torch
from torch.nn import functional as F
from dino_extractor import ViTExtractor
from PIL import Image
import numpy as np
import tyro
from dataclasses import dataclass, field
from utilities import VLAD, get_top_k_recall, seed_everything
import einops as ein
import wandb
import matplotlib.pyplot as plt
import time
import joblib
import traceback
from tqdm.auto import tqdm
from dvgl_benchmark.datasets_ws import BaseDataset
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
        wandb_group="Direct-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Dino parameters
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino to use as the base model.
    """
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
    # Facet for extracting descriptors
    desc_facet: Literal["key", "query", "value", "token"] = "key"
    # Apply log binning to the descriptor
    desc_bin: bool = False
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Configure the behavior of GeM
    gem_use_abs: bool = False
    """
        If True, the `abs` is applied to the patch descriptors (all
        values are strictly positive). Otherwise, a gimmick involving
        complex numbers is used. If False, the `gem_p` should be an
        integer (fractional will give complex numbers when applied to
        negative numbers as power).
    """
    # Do GeM element-by-element (only if gem_use_abs = False)
    gem_elem_by_elem: bool = False
    """
        Do the GeM element-by-element (only if `gem_use_abs` = False).
        This can be done to prevent the RAM use from exploding for 
        large datasets.
    """
    # GeM Pooling Parameter
    gem_p: float = 3


# %%
# ---------------- Functions ----------------
@torch.no_grad()
def build_gems(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build GeM (global) vectors for database and query images.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        
        Returns:
        - db_gems:      GeM descriptors of database of shape 
                        [n_db, d_dim]
                        - n_db: Number of database images
                        - d_dim: Descriptor dimensionality for the
                            (patch) features
        - qu_gems:      GeM descriptors of queries of shape 
                        [n_qu, d_dim], 'n_qu' is num. of queries
    """
    
    extractor = ViTExtractor(largs.model_type, largs.vit_stride, 
                device=device)
    if verbose:
        print("Dino model loaded")
    
    def extract_patch_descriptors(indices):
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            desc = extractor.extract_descriptors(img,
                    layer=largs.desc_layer, facet=largs.desc_facet,
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
            patch_descs.append(desc.squeeze().cpu())
        patch_descs = torch.stack(patch_descs)
        return patch_descs
    
    def get_gem_descriptors(patch_descs: torch.Tensor):
        assert len(patch_descs.shape) == len(("N", "n_p", "d_dim"))
        g_res = None
        if largs.gem_use_abs:
            g_res = torch.mean(torch.abs(patch_descs)**largs.gem_p, 
                    dim=-2) ** (1/largs.gem_p)
        else:
            if largs.gem_elem_by_elem:
                g_res_all = []
                for patch_desc in patch_descs:
                    x = torch.mean(patch_desc**largs.gem_p, dim=-2)
                    g_res = x.to(torch.complex64) ** (1/largs.gem_p)
                    g_res = torch.abs(g_res) * torch.sign(x)
                    g_res_all.append(g_res)
                g_res = torch.stack(g_res_all)
            else:
                x = torch.mean(patch_descs**largs.gem_p, dim=-2)
                g_res = x.to(torch.complex64) ** (1/largs.gem_p)
                g_res = torch.abs(g_res) * torch.sign(x)
        return g_res    # [N, d_dim]
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    # Get GeM descriptors of the database
    if verbose:
        print("Building GeMs for database...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    # All database descs (local descriptors): [n_db, n_d, d_dim]
    full_db = extract_patch_descriptors(db_indices)
    if verbose:
        print(f"Full database descriptor shape: {full_db.shape}")
    db_gems: torch.Tensor = get_gem_descriptors(full_db)
    del full_db
    if verbose:
        print(f"Database GeMs shape: {db_gems.shape}")
    
    # Get GeM of the queries
    if verbose:
        print("Building GeMs for queries...")
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    full_qu = []
    # Get global descriptors for queries
    full_qu = extract_patch_descriptors(qu_indices)
    if verbose:
        print(f"Full query descriptor shape: {full_qu.shape}")
    qu_gems: torch.Tensor = get_gem_descriptors(full_qu)
    del full_qu
    if verbose:
        print(f"Query GeMs shape: {qu_gems.shape}")
    
    # Return VLADs
    return db_gems, qu_gems


# %%
@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    
    if largs.prog.use_wandb:
        # Launch WandB
        wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, config=largs,
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
        print(f"Initialized WandB run: {wandb_run.name}")
    
    print("--------- Generating VLADs ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}, split: {largs.data_split}")
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                            largs.data_split)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir)
    elif ds_name=="Oxford_25m":
        vpr_ds = Oxford(ds_dir, override_dist=25)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    
    db_gems, qu_gems = build_gems(largs, vpr_ds)
    print("--------- Generated GeMs ---------")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_gems, qu_gems, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.model_type),
        "Desc-Layer": str(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
        "Desc-Dim": str(db_gems.shape[1]),
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_gems)),
        "Num-QU": str(len(qu_gems)),
        "Agg-Method": "GeM",
        "Timestamp": str(ts)
    }
    print("Results: ")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls: ")
    for k in recalls:
        results[f"R@{k}"] = recalls[k]
        print(f"  - R@{k}: {recalls[k]:.5f}")
    if largs.show_plot:
        plt.plot(recalls.keys(), recalls.values())
        plt.ylim(0, 1)
        plt.xticks(largs.top_k_vals)
        plt.xlabel("top-k values")
        plt.ylabel(r"% recall")
        plt_title = "Recall curve"
        if largs.exp_id is not None:
            plt_title = f"{plt_title} - Exp {largs.exp_id}"
        if largs.prog.use_wandb:
            plt_title = f"{plt_title} - {wandb_run.name}"
        plt.title(plt_title)
        plt.show()
    
    # Log to WandB
    if largs.prog.use_wandb:
        wandb.log(results)
        for tk in recalls:
            wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
    
    # Add retrievals
    results["Qual-Dists"] = dists
    results["Qual-Indices"] = indices
    save_res_file = None
    if largs.exp_id == True:
        save_res_file = caching_directory
    elif type(largs.exp_id) == str:
        save_res_file = f"{caching_directory}/experiments/"\
                        f"{largs.exp_id}"
    if save_res_file is not None:
        if not os.path.isdir(save_res_file):
            os.makedirs(save_res_file)
        save_res_file = f"{save_res_file}/results_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")
    
    if largs.prog.use_wandb:
        wandb.finish()
    print("--------------------- END ---------------------")


# %%
if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)


# %%
