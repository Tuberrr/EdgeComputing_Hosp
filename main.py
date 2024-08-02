import torch
from utils.options import *
from utils.set_seed import *
from utils.get_dataset import *
from models.Nets import *
if __name__ == "__main__":
    # parse args解析命令行  #加一个数据中心数量
    args = args_parser()
    args.device = torch.device(
        "cuda:{}".format(-1)
        if torch.cuda.is_available()
        else "cpu"
    )
    set_random_seed(args.seed)
    
    hosp_datavolume = [75,50,20,16]
    
    dataset_train, dataset_test, dict_public, dict_hosp = get_dataset(hosp_datavolume) 
    
    net_glob = CNNCifar10()
    
    net_glob.to(args.device)
    print(net_glob)  # 初始化模型对象