# def FedAvg(args, net_glob, dataset_train, dataset_test, dict_users):

#     net_glob.train()

#     times = []
#     total_time = 0

#     # training
#     acc = []
#     loss = []
#     train_loss = []

#     for iter in range(args.epochs):

#         print("*" * 80)
#         print("Round {:3d}".format(iter))

#         w_locals = []
#         lens = []
#         m = max(int(args.frac * args.num_users), 1)
#         idxs_users = np.random.choice(range(args.num_users), m, replace=False)
#         max_time = 0
#         for idx in idxs_users:
#             local = LocalUpdate_FedAvg(
#                 args=args, dataset=dataset_train, idxs=dict_users[idx]
#             )
#             w = local.train(net=copy.deepcopy(net_glob).to(args.device))

#             w_locals.append(copy.deepcopy(w))
#             lens.append(len(dict_users[idx]))
#             run_time = (
#                 asyn_clients[idx].get_train_time() + asyn_clients[idx].get_comm_time()
#             )
#             if max_time < run_time:
#                 max_time = run_time
#         total_time += max_time
#         # update global weights
#         w_glob = Aggregation(w_locals, lens)

#         # copy weight to net_glob
#         net_glob.load_state_dict(w_glob)

#         # if iter % 10 == 9:
#         #     item_acc, item_loss = test_with_loss(net_glob, dataset_test, args)
#         #     ta, tl = test_with_loss(net_glob, dataset_train, args)
#         #     acc.append(item_acc)
#         #     loss.append(item_loss)
#         #     train_loss.append(tl)
#         #     times.append(total_time)
#         item_acc, item_loss = test_with_loss(net_glob, dataset_test, args)
#         ta, tl = test_with_loss(net_glob, dataset_train, args)
#         acc.append(item_acc)
#         loss.append(item_loss)
#         train_loss.append(tl)
#         times.append(total_time)

#     save_result(acc, "test_acc", args)
#     save_result(loss, "test_loss", args)
#     save_result(times, "test_time", args)
#     save_result(train_loss, "test_train_loss", args)
#     save_model(net_glob.state_dict(), "test_model", args)
