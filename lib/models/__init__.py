

from .bisenetv1 import BiSeNetV1
from .bisenetv2 import BiSeNetV2
from .bisenetv1_swin import BiSeNetV1_Swin
from .bisenetv2_contrast import BiSeNetV2_Contrast
from .bisenetv2_contrast_wn import BiSeNetV2_Contrast_WN
from .bisenetv2_contrast_km import BiSeNetV2_Contrast_KM


model_factory = {
    'bisenetv1': BiSeNetV1,
    'bisenetv2': BiSeNetV2,
    'bisenetv1_swin': BiSeNetV1_Swin,
    'bisenetv2_contrast': BiSeNetV2_Contrast,
    'bisenetv2_contrast_wn': BiSeNetV2_Contrast_WN,
    'bisenetv2_contrast_km': BiSeNetV2_Contrast_KM
}
