# Training the model without pretrain weight
# python train.py -c config/TG-RAPMN.yaml

# Training the model use the pretrain weight
python train.py -c config/TG-RAPMN.yaml -r model_checkpoints/20251128/TGRAPMNet_20251128_085306/checkpoint_11.pt
