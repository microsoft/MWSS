'''
Copyright (c) Microsoft Corporation, Yichuan Li and Kai Shu.
Licensed under the MIT license.
Authors: Guoqing Zheng (zheng@microsoft.com), Yichuan Li and Kai Shu
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

try:
    from apex import amp
except:
    print("Install AMP!")

def modify_parameters(net, deltas, eps):
    for param, delta in zip(net.parameters(), deltas):
        if delta is None:
            continue
        param.data.add_(eps, delta)



def update_params_sgd(params, grads, opt, eta, args):
    # supports SGD-like optimizers
    ans = []

    wdecay = opt.defaults.get('weight_decay', 0.)
    momentum = opt.defaults.get('momentum', 0.)
    # eta = opt.defaults["lr"]
    for i, param in enumerate(params):
        if grads[i] is None:
            ans.append(param)
            continue
        try:
            moment = opt.state[param]['momentum_buffer'] * momentum
        except:
            moment = torch.zeros_like(param)

        dparam = grads[i] + param * wdecay

        # eta is the learning tate
        ans.append(param - (dparam + moment) * eta)

    return ans

def update_params_adam(params, grads, opt):
        ans = []
        group = opt.param_groups[0]
        assert len(opt.param_groups) == 1
        for p, grad in zip(params, grads):
            if grad is None:
                ans.append(p)
                continue
            amsgrad = group['amsgrad']
            state = opt.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data)

            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']
            beta1, beta2 = group['betas']

            state['step'] += 1
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            if group['weight_decay'] != 0:
                grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(1 - beta1, grad)
            exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1


            # ans.append(p.data.addcdiv(-step_size, exp_avg, denom))
            ans.append(torch.addcdiv(p, -step_size, exp_avg, denom))

        return ans

# ============== l2w step procedure debug ===================
# NOTE: main_net is implemented as nn.Module as usual

def step_l2w(main_net, main_opt, main_scheduler, g_input, s_input, train_input, args, gold_ratio):
    # init eps to 0
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    try:
        eta = main_scheduler.get_lr()[0]
    except:
        eta = main_opt.defaults.get("lr", 0)



    eps = nn.Parameter(torch.zeros_like(s_input['labels'].float()))
    eps = eps.view(-1)

    # flat the weight for multi head


    # calculate current weighted loss
    main_net.train()

    loss_s = main_net(**s_input)[0]
    # {reduction: "none"} in s_inputs
    
    loss_s = (eps * loss_s).sum()
    if gold_ratio > 0:
        loss_train = main_net(**train_input)[0]
        loss_s = (loss_train + loss_s) / 2



    # get theta grads
    # 1. update w to w'
    param_grads = torch.autograd.grad(loss_s, main_net.parameters(), allow_unused=True)


    params_new = update_params_sgd(main_net.parameters(), param_grads, main_opt, eta, args)
    # params_new = update_params_adam(main_net.parameters(), param_grads, main_opt)

    # 2. set w as w'
    params = []
    for i, param in enumerate(main_net.parameters()):
        params.append(param.data.clone())
        param.data = params_new[i].data # use data only
    # 3. compute d_w' L_{D}(w')

    loss_g = main_net(**g_input)[0]

    params_new_grad = torch.autograd.grad(loss_g, main_net.parameters(), allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = params[i]

    # change main_net parameter
    _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)

    loss_s_p = main_net(**s_input)[0]
    loss_s_p = (eps * loss_s_p).sum()
    if gold_ratio > 0:
        loss_train_p = main_net(**train_input)[0]
        loss_s_p = (loss_s_p + loss_train_p) / 2



    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2 * _eps)
    loss_s_n = main_net(**s_input)[0]
    loss_s_n = (eps * loss_s_n).sum()
    if gold_ratio > 0:
        loss_train_n = main_net(**train_input)[0]
        loss_s_n = (loss_train_n + loss_s_n)

    proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)

    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)
    eps_grad = torch.autograd.grad(proxy_g, eps, allow_unused=True)[0]

    # update eps
    w = F.relu(-eps_grad)

    if w.max() == 0:
        w = torch.ones_like(w)
    else:
        w = w / w.sum()


    loss_s = main_net(**s_input)[0]
    loss_s = (w * loss_s).sum()
    if gold_ratio > 0:
        loss_train = main_net(**train_input)[0]
        loss_s = (loss_s + loss_train) / 2

    # if info['step'] is not None:
    #     writer.add_histogram("weight/GoldRatio_{}_InstanceWeight".format(info['gold_ratio']), w.detach(), global_step=info['step'])

    if gold_ratio > 0:
        loss_s += main_net(**train_input)[0]



    # main_opt.zero_grad()
    main_net.zero_grad()
    if args.fp16:
        with amp.scale_loss(loss_s, main_opt) as loss_s:
            loss_s.backward()
    else:
        loss_s.backward()
    main_opt.step()


    if type(main_scheduler).__name__ is "LambdaLR":
        main_scheduler.step(loss_s)
    else:
        main_scheduler.step()

    main_net.eval()
    loss_g = main_net(**g_input)[0]
    return loss_g, loss_s



# w_net now computes both (ins_weight, g_weight)
def step_l2w_group_net(main_net, main_opt, main_scheduler, g_input, s_input, train_input, args, gw, gw_opt,
                       gw_scheduler, gold_ratio):
    #if args.fp16:
    #    try:
    #        from apex import amp
    #    except ImportError:
    #        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    # ATTENTION: s_input["labels"] is [bs, K]
    # forward function of gw is:
    #    # group weight shape is [1, K]
    #    group_weight = torch.sigmoid(self.pesu_group_weight)
    #    final_weight = torch.matmul(iw.view(-1, 1), group_weight)
    #    return (final_weight * (item_loss.view(final_weight.shape))).sum()
    
    # get learn rate from optimizer or scheduler
    '''
    try:
        eta_group = gw_scheduler.get_lr()
    except:
        eta_group = gw_opt.defaults.get("lr", 0)
    '''
    
    try:
        eta = main_scheduler.get_lr()
        if type(eta).__name__ == "list":
            eta = eta[0]
    except:
        eta = main_opt.defaults.get("lr", 0)

    # calculate current weighted loss
    # ATTENTION: loss_s shape: [bs * K, 1]
    y_weak = s_input['labels']
    outputs_s = main_net(**s_input)
    s_feature = outputs_s[2]
    loss_s = outputs_s[0]
    loss_s, _ = gw(s_feature, y_weak, loss_s)

    if gold_ratio > 0:
        loss_train = main_net(**train_input)[0]
        loss_s = (loss_s + loss_train) / 2
    else:
        loss_s = loss_s

    # get theta grads
    # 1. update w to w'
    param_grads = torch.autograd.grad(loss_s, main_net.parameters(), allow_unused=True)

    # 2. set w as w'
    params = [param.data.clone() for param in main_net.parameters()]
    for i, param in enumerate(main_net.parameters()):
        if param_grads[i] is not None:
            param.data.sub_(eta*param_grads[i])

    # 3. compute d_w' L_{D}(w')
    loss_g = main_net(**g_input)[0]

    params_new_grad = torch.autograd.grad(loss_g, main_net.parameters(), allow_unused=True)

    # 4. revert from w' to w for main net
    for i, param in enumerate(main_net.parameters()):
        param.data = params[i]

    # change main_net parameter
    _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before

    # modify w to w+
    modify_parameters(main_net, params_new_grad, _eps)
    outputs_s_p = main_net(**s_input)
    loss_s_p = outputs_s_p[0]
    s_p_feature = outputs_s_p[2]
    loss_s_p,_ = gw(s_p_feature, y_weak, loss_s_p)
    if gold_ratio > 0:
        loss_train = main_net(**train_input)[0]
        loss_s_p = (loss_s_p + loss_train ) / 2

    # loss_s_p = (eps * F.cross_entropy(logit_s_p, target_s, reduction='none')).sum()

    # modify w to w- (from w+)
    modify_parameters(main_net, params_new_grad, -2 * _eps)
    outputs_s_n = main_net(**s_input)
    loss_s_n = outputs_s_n[0]
    s_n_feature = outputs_s_n[2]
    loss_s_n, _ = gw(s_n_feature, y_weak, loss_s_n)
    if gold_ratio > 0:
        loss_train = main_net(**train_input)[0]
        loss_s_n = (loss_s_n + loss_train) / 2
    
    # loss_s_n = (eps * F.cross_entropy(logit_s_n, target_s, reduction='none')).sum()

    proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)

    # modify to original w
    modify_parameters(main_net, params_new_grad, _eps)

    # eps_grad = torch.autograd.grad(proxy_g, eps)[0]
    # update gw
    gw_opt.zero_grad()
    if args.fp16:
        with amp.scale_loss(proxy_g, gw_opt) as proxy_gg:
            proxy_gg.backward()
    else:
        proxy_g.backward()
    gw_opt.step()

    if type(main_scheduler).__name__ == "LambdaLR":
        gw_scheduler.step(proxy_g)
    else:
        gw_scheduler.step()
    # call scheduler for gw if applicable here


    outputs_s = main_net(**s_input)
    loss_s = outputs_s[0]
    s_feature = outputs_s[2]
    loss_s, instance_weight = gw(s_feature, y_weak, loss_s)

    
    # write the group weight and instance weight

    # mean reduction
    if gold_ratio != 0:
        loss_train = main_net(**train_input)[0]
        loss_s = (loss_s + loss_train) / 2

    main_opt.zero_grad()
    if args.fp16:
        with amp.scale_loss(loss_s, main_opt) as loss_ss:
            loss_ss.backward()
    else:
        loss_s.backward()
    main_opt.step()
    if type(main_scheduler).__name__ is "LambdaLR":
        main_scheduler.step(loss_s)
    else:
        main_scheduler.step()

    return loss_g, loss_s, instance_weight



# def group_step_l2w(main_net, main_opt, group_weight, group_opt, val_input, s_input, g_input, args, scheduler,
#                    group_scheduler, step=None, writer=None):
#     # init eps to 0
#     try:
#         eta = scheduler.get_lr()[0]
#     except:
#         eta = main_opt.defaults.get("lr", 0)
#
#
#     eps = nn.Parameter(torch.zeros_like(s_input['labels'][:,0].float()))
#     eps = eps.view(-1)
#
#
#
#     # calculate current weighted loss
#     main_net.train()
#     loss_s = main_net(**s_input)[0]
#     # {reduction: "none"} in s_inputs
#     loss_s = (group_weight(eps, loss_s)).sum()
#
#     # get theta grads
#     # 1. update w to w'
#     param_grads = torch.autograd.grad(loss_s, main_net.parameters(), allow_unused=True)
#
#     params_new = update_params_sgd(main_net.parameters(), param_grads, main_opt, args, eta)
#     # params_new = update_params_adam(main_net.parameters(), param_grads, main_opt)
#
#     # 2. set w as w'
#     params = []
#     for i, param in enumerate(main_net.parameters()):
#         params.append(param.data.clone())
#         param.data = params_new[i].data # use data only
#
#     # 3. compute d_w' L_{D}(w')
#     loss_g = main_net(**val_input)[0]
#
#     params_new_grad = torch.autograd.grad(loss_g, main_net.parameters(), allow_unused=True)
#
#     # 4. revert from w' to w for main net
#     for i, param in enumerate(main_net.parameters()):
#         param.data = params[i]
#
#     # change main_net parameter
#     _eps = 1e-6  # 1e-3 / _concat(params_new_grad).norm # eta 1e-6 before
#
#     # modify w to w+
#     modify_parameters(main_net, params_new_grad, _eps)
#     loss_s_p = main_net(**s_input)[0]
#     loss_s_p = (group_weight(eps, loss_s_p)).sum()
#
#     # modify w to w- (from w+)
#     modify_parameters(main_net, params_new_grad, -2 * _eps)
#     loss_s_n = main_net(**s_input)[0]
#     loss_s_n = (group_weight(eps, loss_s_n)).sum()
#
#     proxy_g = -eta * (loss_s_p - loss_s_n) / (2. * _eps)
#
#     # modify to original w
#     modify_parameters(main_net, params_new_grad, _eps)
#
#     grads = torch.autograd.grad(proxy_g, [eps, group_weight.pesu_group_weight], allow_unused=True)
#     eps_grad = grads[0]
#     group_weight_grad = grads[1]
#
#     # update eps
#     w = F.relu(-eps_grad)
#
#     if w.max() == 0:
#         w = torch.ones_like(w)
#     else:
#         w = w / w.sum()
#
#     group_opt.zero_grad()
#     group_weight.pesu_group_weight.grad = group_weight_grad
#     group_opt.step()
#     group_scheduler.step(proxy_g)
#
#     loss_s = main_net(**s_input)[0]
#
#     loss_s = (group_weight(w, loss_s)).sum()
#
#     if step is not None:
#         writer.add_histogram("weight/InstanceWeight", w.detach(), global_step=step)
#         if group_weight is not None:
#             writer.add_histogram("Weight/GroupWeight", group_weight.pesu_group_weight.data, global_step=step)
#
#     if g_input is not None:
#         loss_s += main_net(**g_input)[0]
#
#
#     # main_opt.zero_grad()
#     main_net.zero_grad()
#     loss_s.backward()
#     main_opt.step()
#
#     if scheduler is not None:
#         scheduler.step(loss_s)
#
#     main_net.eval()
#     loss_g = main_net(**val_input)[0]
#     return loss_g, loss_s
