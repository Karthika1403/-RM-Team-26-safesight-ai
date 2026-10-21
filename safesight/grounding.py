"""S2: selective grounding check -> support score g_k in [0,1].
MockGrounder : uses scene tags supplied by the user/dataset (demo, tests, ablations on labelled data).
ClipGrounder : CLIP image-text contrastive support (CPU-friendly). Replace with VBackChecker-style
               pixel grounding when a GPU is available."""

class MockGrounder:
    def __init__(self, tags): self.tags = [t.strip().lower() for t in tags if t.strip()]
    def score(self, image, claim_text):
        low = claim_text.lower()
        return 0.9 if any(t in low for t in self.tags) else 0.1

class ClipGrounder:
    NEG = ["a photo of something else", "an empty scene", "an unrelated image"]
    def __init__(self, name="openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        import torch
        self.torch = torch
        self.m, self.p = CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name)
    def score(self, image, claim_text):
        with self.torch.no_grad():
            inp = self.p(text=[claim_text] + self.NEG, images=image, return_tensors="pt", padding=True, truncation=True)
            return float(self.m(**inp).logits_per_image.softmax(-1)[0][0])
