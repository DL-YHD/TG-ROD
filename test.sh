# Validating (no split the visual results)
# CUDA_VISIBLE_DEVICES=0 python test.py -c config/TG-RAPMN.yaml -f valid -r model_checkpoints/20251128/TGRAPMNet_20251128_085306/checkpoint_20.pt

# Validating (split the visual results)
# CUDA_VISIBLE_DEVICES=0 python test.py -c config/TG-RAPMN.yaml -i -f valid -r model_checkpoints/20251128/TGRAPMNet_20251128_085306/checkpoint_13.pt

# Testing
CUDA_VISIBLE_DEVICES=0 python test.py -c config/TG-RAPMN.yaml -f test -r model_checkpoints/20251128/TGRAPMNet_20251128_085306/checkpoint_13.pt
