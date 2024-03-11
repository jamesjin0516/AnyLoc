import os
import torch
from torchvision import transforms
from torchvision.transforms import functional
from .utilities import DinoV2ExtractFeatures
from .utilities import VLAD


class VLADDinoV2FeatureExtractor:

    def __init__(self, root, content):
        self.max_image_size = content["max_image_size"]
        self.device = "cuda" if content["cuda"] else "cpu"

        # DinoV2 & VLAD settings
        self.dinov2_extractor = DinoV2ExtractFeatures("dinov2_vitg14", content["desc_layer"], content["desc_facet"], device=self.device)
        vocab_dir = os.path.join("dinov2_vitg14", f"l{content['desc_layer']}_{content['desc_facet']}_c{content['num_clusters']}")
        c_centers_file = os.path.join(root, content["cache_dir"], "vocabulary", vocab_dir, content["domain"], "c_centers.pt")
        c_centers = torch.load(c_centers_file)
        assert c_centers.shape[0] == content["num_clusters"], f"Wrong number of clusters! (from file {c_centers.shape[0]} != from config {content['num_clusters']})"
        self.vlad = VLAD(content["num_clusters"], desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
        self.vlad.fit(None)

    def __call__(self, images):
        with torch.no_grad():
            scaled_imgs = self.downscale(images)
            b, c, h, w = scaled_imgs.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            scaled_imgs = transforms.CenterCrop((h_new, w_new))(scaled_imgs)
            # Extract descriptor
            features = self.dinov2_extractor(scaled_imgs) # [num_imgs, num_patches, desc_dim]
            vlad_feats = [self.vlad.generate(features[i].cpu().squeeze()) for i in range(features.shape[0])] # VLAD:  [agg_dim]
            return torch.stack(vlad_feats).detach() # shape: [num_imgs, agg_dim]
    
    def downscale(self, images):
        if max(images.shape[-2:]) > self.max_image_size:
            b, c, h, w = images.shape
            # Maintain aspect ratio
            if h == max(images.shape[-2:]):
                w = int(w * self.max_image_size / h)
                h = self.max_image_size
            else:
                h = int(h * self.max_image_size / w)
                w = self.max_image_size
            return functional.resize(images, (h, w), interpolation=functional.InterpolationMode.BICUBIC)
        return images

    @property
    def feature_length(self): return self.vlad.desc_dim
