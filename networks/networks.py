from . import hybrid
from . import vit

MODEL_MAPPING = {
    "VIT3_FC" : vit.VIT3_FC,
    "HYBRID_RESNET18_VITS_FC": hybrid.HYBRID_RESNET18_VITS_FC,
}