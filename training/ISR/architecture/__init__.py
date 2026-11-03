import torch
from .FSR import FSR

def model_generator(method, pretrained_model_path=None):
    if method != 'fsr':
        raise ValueError("This release includes the ISR model: --method fsr")
    model = FSR().cuda()
    if pretrained_model_path is not None:
        checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict({k.removeprefix('module.'): v for k,v in state.items()}, strict=True)
    return model
