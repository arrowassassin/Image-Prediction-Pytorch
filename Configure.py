import os
import argparse




def configure():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default = "/content/drive/MyDrive/Project-mobilenet/Project/", help = "main directory path")
    parser.add_argument("--oper", default = "train", help = "train,test,predict")
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument("--save_interval", type=int, default=10,
                        help='save the checkpoint when epoch MOD save_interval == 0')
    parser.add_argument("--batch_size", type=int, default=256, help='training batch size')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay rate')
    parser.add_argument("--maxepochs", type=int, default=200, help='maxepochs')
    parser.add_argument("--testepoch", type=int, default=190, help='testepoch')
    parser.add_argument("--modeldir", type=str, default="model_v11", help='model directory')
    return parser.parse_args()