import sys
import logging
import copy
import torch
from utils import factory
from utils.data import use_multi_domain_dataset
from utils.data_manager import DataManager
from utils.domain_data_manager import DomainDataManager
from utils.toolkit import count_parameters
import os
import numpy as np


def train(args:dict):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args:dict):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    if use_multi_domain_dataset(args["dataset"]):
        data_manager_cls = DomainDataManager
    else:
        data_manager_cls = DataManager

    data_manager = data_manager_cls(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)
    model.register_data_info(data_manager)
    logging.info("Class ID pairs: {}".format(model._class_id_pairs))

    cnn_curve, nme_curve = {"top1": [], f"top{model.topk}": []}, {"top1": [], f"top{model.topk}": []}
    cnn_matrix, nme_matrix = [], []

    if use_multi_domain_dataset(args["dataset"]):
        cnn_curve_per_domain, nme_curve_per_domain = {}, {}
        cnn_matrix_per_domain, nme_matrix_per_domain = {}, {}
        for domain_id, domain_name in enumerate(data_manager.domain_names):
            cnn_curve_per_domain[domain_name] = {"top1": [], f"top{model.topk}": []}
            nme_curve_per_domain[domain_name] = {"top1": [], f"top{model.topk}": []}
            cnn_matrix_per_domain[domain_name] = []
            nme_matrix_per_domain[domain_name] = []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()

        # report global accuracy
        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]    
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_values = [nme_accy["grouped"][key] for key in nme_keys]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[f"top{model.topk}"].append(cnn_accy[f"top{model.topk}"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve[f"top{model.topk}"].append(nme_accy[f"top{model.topk}"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top{} curve: {}".format(model.topk, cnn_curve[f"top{model.topk}"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top{} curve: {}\n".format(model.topk, nme_curve[f"top{model.topk}"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve[f"top{model.topk}"].append(cnn_accy[f"top{model.topk}"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top{} curve: {}\n".format(model.topk, cnn_curve[f"top{model.topk}"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))


        # report domain wise accuracy
        if use_multi_domain_dataset(args["dataset"]):
            cnn_accy_per_domain, nme_accy_per_domain = model.eval_task_per_domain()
            for domain_id, domain_name in enumerate(data_manager.domain_names):
                cnn_accy = cnn_accy_per_domain[domain_name]
                nme_accy = nme_accy_per_domain[domain_name]
                if nme_accy is not None:
                    logging.info("Domain {}: CNN: {}".format(domain_name, cnn_accy["grouped"]))
                    logging.info("Domain {}: NME: {}".format(domain_name, nme_accy["grouped"]))
                    
                    cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                    cnn_keys_sorted = sorted(cnn_keys)
                    cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                    cnn_matrix_per_domain[domain_name].append(cnn_values)

                    nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
                    nme_keys_sorted = sorted(nme_keys)
                    nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
                    nme_matrix_per_domain[domain_name].append(nme_values)

                    cnn_curve_per_domain[domain_name]["top1"].append(cnn_accy["top1"])
                    cnn_curve_per_domain[domain_name][f"top{model.topk}"].append(cnn_accy[f"top{model.topk}"])

                    nme_curve_per_domain[domain_name]["top1"].append(nme_accy["top1"])
                    nme_curve_per_domain[domain_name][f"top{model.topk}"].append(nme_accy[f"top{model.topk}"])

                    logging.info("Domain {}: CNN top1 curve: {}".format(domain_name, cnn_curve_per_domain[domain_name]["top1"]))
                    logging.info("Domain {}: CNN top{} curve: {}".format(domain_name, model.topk, cnn_curve_per_domain[domain_name][f"top{model.topk}"]))
                    logging.info("Domain {}: NME top1 curve: {}".format(domain_name, nme_curve_per_domain[domain_name]["top1"]))
                    logging.info("Domain {}: NME top{} curve: {}\n".format(domain_name, model.topk, nme_curve_per_domain[domain_name][f"top{model.topk}"]))
                    logging.info("Domain {}: Average Accuracy (CNN): {}".format(domain_name, sum(cnn_curve_per_domain[domain_name]["top1"])/len(cnn_curve_per_domain[domain_name]["top1"])))
                    logging.info("Domain {}: Average Accuracy (NME): {}".format(domain_name, sum(nme_curve_per_domain[domain_name]["top1"])/len(nme_curve_per_domain[domain_name]["top1"])))
                else:
                    logging.info("Domain {}: No NME accuracy.".format(domain_name))
                    logging.info("Domain {}: CNN: {}".format(domain_name, cnn_accy["grouped"]))

                    cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                    cnn_keys_sorted = sorted(cnn_keys)
                    cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                    cnn_matrix_per_domain[domain_name].append(cnn_values)

                    cnn_curve_per_domain[domain_name]["top1"].append(cnn_accy["top1"])
                    cnn_curve_per_domain[domain_name][f"top{model.topk}"].append(cnn_accy[f"top{model.topk}"])

                    logging.info("Domain {}: CNN top1 curve: {}".format(domain_name, cnn_curve_per_domain[domain_name]["top1"]))
                    logging.info("Domain {}: CNN top{} curve: {}\n".format(domain_name, model.topk, cnn_curve_per_domain[domain_name][f"top{model.topk}"]))
                    logging.info("Domain {}: Average Accuracy (CNN): {}".format(domain_name, sum(cnn_curve_per_domain[domain_name]["top1"])/len(cnn_curve_per_domain[domain_name]["top1"])))

        model.after_task()

    if args.get('print_forget', False):
        # report global accuracy matrix and forgetting
        if len(cnn_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(cnn_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            # print('Accuracy Matrix (CNN):')
            # print(np_acctable)
            logging.info('Accuracy Matrix (CNN): \n{}'.format(np_acctable))
            logging.info('Forgetting (CNN): {}'.format(forgetting))
        if len(nme_matrix) > 0:
            np_acctable = np.zeros([task + 1, task + 1])
            for idxx, line in enumerate(nme_matrix):
                idxy = len(line)
                np_acctable[idxx, :idxy] = np.array(line)
            np_acctable = np_acctable.T
            forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
            # print('Accuracy Matrix (NME):')
            # print(np_acctable)
            logging.info('Accuracy Matrix (NME): \n{}'.format(np_acctable))
            logging.info('Forgetting (NME): {}'.format(forgetting))

        # report domain wise accuracy matrix and forgetting
        if use_multi_domain_dataset(args["dataset"]):
            for domain_name in data_manager.domain_names:
                cnn_matrix = cnn_matrix_per_domain[domain_name]
                if len(cnn_matrix)>0:
                    np_acctable = np.zeros([task + 1, task + 1])
                    for idxx, line in enumerate(cnn_matrix):
                        idxy = len(line)
                        np_acctable[idxx, :idxy] = np.array(line)
                    np_acctable = np_acctable.T
                    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
                    # print('Domain {}: Accuracy Matrix (CNN):'.format(domain_name))
                    # print(np_acctable)
                    logging.info('Domain {}: Accuracy Matrix (CNN): \n{}'.format(domain_name, np_acctable))
                    logging.info('Domain {}: Forgetting (CNN): {}'.format(domain_name, forgetting))
                nme_matrix = nme_matrix_per_domain[domain_name]
                if len(nme_matrix)>0:
                    np_acctable = np.zeros([task + 1, task + 1])
                    for idxx, line in enumerate(nme_matrix):
                        idxy = len(line)
                        np_acctable[idxx, :idxy] = np.array(line)
                    np_acctable = np_acctable.T
                    forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
                    # print('Domain {}: Accuracy Matrix (NME):'.format(domain_name))
                    # print(np_acctable)
                    logging.info('Domain {}: Accuracy Matrix (NME): \n{}'.format(domain_name, np_acctable))
                    logging.info('Domain {}: Forgetting (NME): {}'.format(domain_name, forgetting))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))