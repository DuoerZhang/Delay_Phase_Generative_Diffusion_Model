from cond_liver_box.Train import train, eval
import os

def main(model_config = None):
    modelConfig = {
        "state": "eval", # train or eval
        "epoch": 400,
        "iters": 200001,
        "printfreq": 1000,
        "testfreq": 2000,
        "savefreq": 20000,
        "batch_size": 4,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1,1,2,2,2],#[1,2,2,4,8]
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "multiplier": 2.,
        "beta_1": 0.0001,
        "beta_T": 0.02,
        "img_size": 256,
        "grad_clip": 1.,
        "device": "cuda:2",
        "train_data_dir":"./data",
        "val_data_dir":"./data",
        "test_data_dir":"./data",

        # "training_load_weight": "checkpoint_200000.pth.tar",
        "training_load_weight": None,
        "save_weight_dir": "./Checkpoints/",
        "sampled_dir": "./SampledImgs/",
        "sampledImgName": "",
        "nrow": 4
        }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        # Create results directory
        if not os.path.isdir(modelConfig["save_weight_dir"]):
            os.makedirs(modelConfig["save_weight_dir"])
        print('save at',modelConfig["save_weight_dir"])
        train(modelConfig)
    elif modelConfig["state"] == "eval":
        if not os.path.isdir(modelConfig["sampled_dir"]):
            os.makedirs(modelConfig["sampled_dir"])
        print('sample at ',modelConfig["sampled_dir"])
        eval(modelConfig)

if __name__ == '__main__':
    main()
