import argparse


def getargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/home/lijunhao/Seg/CauSSL-main', help='Name of Experiment')
    parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
    parser.add_argument('--base_lr', type=float, default=1e-2, help='maximum epoch number to train')
    parser.add_argument('--epoch_num', type=int, default=200, help='epoch_num')
    parser.add_argument('--deterministic', type=int, default=True, help='whether use deterministic training')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--device', type=str, default="cuda:3", help='device')
    parser.add_argument('--dataset', type=str, default="Throat", help="dataset choice from Throat or Throat2 or Throat3")
    parser.add_argument('--consistency', type=float, default=1, help="consistency weight")
    parser.add_argument('--consistency_type', type=str, default="MT", help="consistency type")
    parser.add_argument('--set_moe_layer', type=int, default=3, help="each layer to insert moe")
    parser.add_argument('--label_num', type=int, help="label_num")
    parser.add_argument('-n', '--nodes', default=1, type=int, help='the number of nodes/computer')
    parser.add_argument('--cuda', type=str, default="1,2", help="cuda id")
    parser.add_argument('--aug', type=str, default="strong", help="strong or weak")
    parser.add_argument('--num_pt', type=int, default=8, help='number of prototype tokens')
    parser.add_argument('--num_expert', type=int, default=8, help='number of experts')
    parser.add_argument('--label_shape', type=int, default=256, help='label shape')
    parser.add_argument('--moe_loss_coef', type=float, default=0.01, help="consistency weight")

    parser.add_argument('-g', '--gpus', default=2, type=int, help='the number of gpus per nodes')
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--port', type=str, default="24354", help="ddp port")

    args = parser.parse_args()
    return args
