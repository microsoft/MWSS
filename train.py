'''
Copyright (c) Microsoft Corporation, Yichuan Li and Kai Shu.
Licensed under the MIT license.
Authors: Guoqing Zheng (zheng@microsoft.com), Yichuan Li and Kai Shu
'''
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from itertools import chain
from correction_matrix import correction_result, get_correction_matrix
import model
from itertools import chain

def train(gold_iter, sliver_iter, val_iter, model, args, C_hat=None, statues=""):
    if args.cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    gold_batch_count_1 = len(gold_iter)
    sliver_batch_count = len(sliver_iter)
    sliver_time = sliver_batch_count / gold_batch_count_1
    gold_batch_count = int(sliver_time) * gold_batch_count_1
    gold_iter_list = [gold_iter for i in range(int(sliver_time))]
    gold_iter_list.append(sliver_iter)
    for epoch in range(1, args.epochs+1):
        sliver_gt_label = []
        sliver_target_label = []
        sliver_predic_pro = []

        for batch_idx, batch in enumerate(chain(gold_iter_list)):
            model.train()
            feature, target = batch.text, batch.label
            feature = torch.transpose(feature, 1, 0)
            target = target - 1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            if batch_idx >= gold_batch_count and C_hat is not None:
                # switch to the sliver mode

                sliver_gt_label.append((batch.gt_label.numpy() - 1).tolist())
                logit = correction_result(logit, C_hat)

                sliver_target_label.append(target.cpu().numpy().tolist())
                sliver_predic_pro.append(torch.argmax(logit, dim=-1).cpu().numpy().tolist())

            logit = torch.log(logit)
            th1 = target[target > 1]
            th2 = target[target < 0]
            assert len(th1) == 0 and len(th2) == 0

            loss = F.nll_loss(logit, target)

            loss.backward()
            optimizer.step()


            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                # sys.stdout.write(
                #     '\r{} Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(statues, steps,
                #                                                              loss.data,
                #                                                              accuracy,
                #                                                              corrects,
                #                                                              batch.batch_size))
            if steps % args.test_interval == 0:
            # if steps % 1== 0:
                dev_acc = eval(val_iter, model, args)
                dev_acc = dev_acc[0]
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    # save the best model
                    if args.save_best:
                        save(model, args.save_dir, 'best_{}'.format(statues), 0)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)
            steps += 1
        if C_hat is not None:
            sliver_gt_label = list(chain.from_iterable(sliver_gt_label))
            sliver_target_label = list(chain.from_iterable(sliver_target_label))
            sliver_predic_pro = list(chain.from_iterable(sliver_predic_pro))
            acc = accuracy_score(sliver_gt_label, sliver_target_label)
            precision = precision_score(sliver_gt_label, sliver_target_label, average="macro")
            recall = recall_score(sliver_gt_label, sliver_target_label, average="macro")
            acc_sliver_target = accuracy_score(sliver_target_label, sliver_predic_pro)
            acc_sliver_gt = accuracy_score(sliver_gt_label, sliver_predic_pro)
            print("\n" + statues + "\tSliver " + "\t Acc {}, \tPrecision {}, \tRecall {} \n acc_target {}, acc_gt {}"
                  .format(acc, precision, recall, acc_sliver_target, acc_sliver_gt))
            # print("\n" + statues + "\tSliver " + "\t Acc {}, \tPrecision {}, \tRecall {}".format(acc, precision, recall))
            # print("\n" + statues + "\tSliver " + "\t Acc {}, \tPrecision {}, \tRecall {}".format(acc, precision, recall))

def train_hydra_base(gold_iter, sliver_iter, val_iter, model, args, alpha, statues=""):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    model.train()
    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    gold_batch_count = len(gold_iter)
    for epoch in range(1, args.epochs + 1):
        sliver_gt_all = []
        sliver_labels_all = []
        sliver_predic_pro = []
        for batch_idx, batch in enumerate(chain(gold_iter, sliver_iter)):
            model.train()
            feature, target = batch.text, batch.label
            feature = torch.transpose(feature, 1, 0).contiguous()

            target = target - 1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()

            if batch_idx >= gold_batch_count:
                # switch to the sliver mode
                logit = model.forward_sliver(feature)

            else:
                logit = model.forward_gold(feature)

            logit = torch.log(logit)
            loss = F.nll_loss(logit, target)
            if batch_idx >= gold_batch_count:
                loss = loss * alpha

            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                # sys.stdout.write(
                #     '\r{} Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(statues, steps,
                #                                                              loss.data,
                #                                                              accuracy,
                #                                                              corrects,
                #                                                              batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(val_iter, model, args)
                dev_acc = dev_acc[0]
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    # save the best model
                    if args.save_best:
                        save(model, args.save_dir, 'best_{}'.format(statues), 0)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


        sliver_labels_all = list(chain.from_iterable(sliver_labels_all))
        sliver_gt_all = list(chain.from_iterable(sliver_gt_all))
        sliver_predic_pro = list(chain.from_iterable(sliver_predic_pro))
        acc = accuracy_score(sliver_gt_all,sliver_labels_all)
        recall = recall_score(sliver_gt_all, sliver_labels_all, average="macro")
        precesion = precision_score(sliver_gt_all, sliver_labels_all, average='macro')
        acc_gt = accuracy_score(sliver_gt_all, sliver_predic_pro)
        acc_target = accuracy_score(sliver_labels_all, sliver_predic_pro)
        print("\n\n[Correction Label Result] acc: {}, recall: {}, precesion: {}, \n acc_target: {}, acc_gt: {}"
              .format(acc, recall, precesion, acc_target, acc_gt))

def train_with_glc_label(gold_iter, sliver_iter, val_iter, glc_model, train_model, args, alpha, statues=""):
    if args.cuda:
        glc_model.cuda()
        train_model.cuda()

    optimizer = torch.optim.Adam(train_model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    train_model.train()
    glc_model.eval()
    gold_batch_count = len(gold_iter)
    for epoch in range(1, args.epochs + 1):
        sliver_gt_all = []
        sliver_labels_all = []
        sliver_predic_pro = []
        for batch_idx, batch in enumerate(chain(gold_iter, sliver_iter)):
            feature, target = batch.text, batch.label
            feature = torch.transpose(feature, 1, 0).contiguous()
            train_model.train()
            target = target - 1
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()

            if batch_idx >= gold_batch_count:
                # switch to the sliver mode
                sliver_logit = glc_model(feature)
                target = torch.argmax(sliver_logit, dim=-1)
                logit = train_model.forward_sliver(feature)

                sliver_predic_pro.append(torch.argmax(logit, dim=-1).cpu().numpy().tolist())
                sliver_labels_all.append(target.cpu().numpy().tolist())
                sliver_gt_target = batch.gt_label - 1
                sliver_gt_all.append(sliver_gt_target.numpy().tolist())

            else:
                logit = train_model.forward_gold(feature)

            logit = torch.log(logit)
            loss = F.nll_loss(logit, target)
            if batch_idx >= gold_batch_count:
                loss = loss * alpha

            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects / batch.batch_size
                # sys.stdout.write(
                #     '\r{} Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(statues, steps,
                #                                                              loss.data,
                #                                                              accuracy,
                #                                                              corrects,
                #                                                              batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(val_iter, train_model, args)
                dev_acc = dev_acc[0]
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    # save the best model
                    if args.save_best:
                        save(train_model, args.save_dir, 'best_{}'.format(statues), 0)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(train_model, args.save_dir, 'snapshot', steps)


        sliver_labels_all = list(chain.from_iterable(sliver_labels_all))
        sliver_gt_all = list(chain.from_iterable(sliver_gt_all))
        sliver_predic_pro = list(chain.from_iterable(sliver_predic_pro))
        acc = accuracy_score(sliver_gt_all,sliver_labels_all)
        recall = recall_score(sliver_gt_all, sliver_labels_all, average="macro")
        precesion = precision_score(sliver_gt_all, sliver_labels_all, average='macro')
        acc_gt = accuracy_score(sliver_gt_all, sliver_predic_pro)
        acc_target = accuracy_score(sliver_labels_all, sliver_predic_pro)
        print("\n\n[Correction Label Result] acc: {}, recall: {}, precesion: {}, \n acc_target: {}, acc_gt: {}"
              .format(acc, recall, precesion, acc_target, acc_gt))



def estimate_c(model, gold_iter, args):
    # load the best pesudo-clf model
    model.eval()
    gold_pro_all = []
    gold_label_all = []
    with torch.no_grad():
        for batch in gold_iter:
            gold_feature, gold_target = batch.text, batch.label
            gold_target = gold_target - 1
            gold_feature = torch.transpose(gold_feature, 1, 0).contiguous()
            if args.cuda:
                gold_feature, gold_target = gold_feature.cuda(), gold_target.cuda()
            gold_pro_all.append(model(gold_feature))
            gold_label_all.append(gold_target)

        gold_pro_all = torch.cat(gold_pro_all, dim=0)
        gold_label_all = torch.cat(gold_label_all, dim=0)

        C_hat = get_correction_matrix(gold_pro=gold_pro_all, gold_label=gold_label_all, method=args.gold_method)
    return C_hat


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    prediction = []
    labels = []
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = torch.transpose(feature, 1, 0).contiguous()
        target = target - 1
        # feature.data.t_(), target.data.sub_(1)  # batch first, index align
        # if args.cuda:

        if args.cuda:
            feature, target = feature.cuda(), target.cuda()


        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.data
        prediction += torch.argmax(logit, 1).cpu().numpy().tolist()
        labels.extend(target.cpu().numpy().tolist())
    accuracy = accuracy_score(y_true=labels, y_pred=prediction)
    size = len(data_iter.dataset.examples)
    avg_loss /= size
    corrects = accuracy * size

    recall = recall_score(y_true=labels, y_pred=prediction, average="macro")
    precision = precision_score(y_true=labels, y_pred=prediction, average="macro")
    f1 = f1_score(y_true=labels, y_pred=prediction, average="macro")

    print('\nEvaluation - loss: {:.6f} recall: {:.4f}, precision: {:.4f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       recall,
                                                                        precision,
                                                                       accuracy,
                                                                       corrects, 
                                                                       size))
    return accuracy, recall, precision, f1


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    #return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

def load_model(model, save_dir, save_prefix, steps):
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    model.load_state_dict(torch.load(save_path))
    return model


def train_in_one(args, gold_iter, sliver_iter, val_iter, test_iter, gold_frac, alpha):
    torch.manual_seed(123)
    fout = open(os.path.join(args.save_dir, "a.result"), "a")
    fout.write("-" * 90 + "Gold Ratio: {} Alpha: {}".format(gold_frac, alpha) + "-" * 90 + "\n")
    # train only on the weak data
    cnn = model.BiLSTM(args)
    train(gold_iter=gold_iter, sliver_iter=sliver_iter, val_iter=val_iter, model=cnn, args=args, statues="only_weak")
    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("only_weak"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("Weak Acc: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision),
                                                                             str(f1)))
    fout.write("\n")
    del cnn
    del test_model

    # train only on the weak and gold data
    cnn = model.BiLSTM(args)
    train(gold_iter=gold_iter, sliver_iter=sliver_iter, val_iter=val_iter, model=cnn, args=args, statues="weak_gold")
    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("weak_gold"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("WeakGold Acc: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision),
                                                                             str(f1)))
    fout.write("\n")
    del cnn
    del test_model



    # train only on the golden data
    cnn = model.BiLSTM(args)
    train(gold_iter=gold_iter, sliver_iter=gold_iter, val_iter=val_iter, model=cnn, args=args, statues="test")
    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("test"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("Only Gold Acc: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision), str(f1)))
    fout.write("\n")
    del cnn
    del test_model

    # hydra-base model
    cnn = model.BiLSTM(args)
    train_hydra_base(gold_iter=gold_iter, sliver_iter=sliver_iter, val_iter=val_iter, model=cnn, args=args, statues="hydra_base", alpha=alpha)


    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("hydra_base"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("HydraBase Acc: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision),
                                                                             str(f1)))
    fout.write("\n")
    del cnn
    del test_model
    # //////////////////////// train for estimation ////////////////////////
    cnn = model.BiLSTM(args)
    if args.cuda:
        cnn = cnn.cuda()

    print("\n" + "*" * 40 + "Training in Base Estimation" + "*" * 40)
    train(gold_iter=sliver_iter, sliver_iter=sliver_iter, val_iter=val_iter, model=cnn, args=args, statues="esti")
    print("*" * 40 + "Finish in Base Estimation" + "*" * 40)
    del cnn

    # # //////////////////////// estimate C ////////////////////////
    cnn = model.BiLSTM(args)
    cnn = load_model(cnn, args.save_dir, 'best_{}'.format("esti"), 0)
    if args.cuda:
        cnn.cuda()
    C_hat = estimate_c(cnn, gold_iter, args)

    del cnn
    # //////////////////////// retrain with correction ////////////////////////
    cnn = model.BiLSTM(args)

    print("\n" + "*"*40 + "Training in Correction" + "*"*40)

    if args.cuda:
        cnn = cnn.cuda()
    train(gold_iter=gold_iter, sliver_iter=sliver_iter, val_iter=val_iter, model=cnn, args=args, statues="glc", C_hat=C_hat)
    # eval(data_iter=test_iter, model=cnn, args=args)

    # del cnn
    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("glc"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("GLC Result: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision), str(f1)))
    fout.write("\n")
    print("\n" + "*"*40 + "Finish in Correction" + "*"*40)

    # //////////////////////// Using GLC labels for training ////////////////////////
    # correction labels for training the last classifier
    print("\n" + "*"*40 + "Training with GLC label" + "*"*40)
    glc_model = model.BiLSTM(args)
    glc_model = load_model(glc_model, args.save_dir, 'best_{}'.format("glc"), 0)

    final_clf_model = model.BiLSTM(args)
    # final_clf_model = load_model(final_clf_model, args.save_dir, 'best_{}'.format("glc"), 0 )

    train_with_glc_label(gold_iter=gold_iter, sliver_iter=sliver_iter, val_iter=val_iter, glc_model=glc_model,
                         train_model=final_clf_model, args=args, statues="final", alpha=alpha)
    del glc_model
    del final_clf_model
    print("\n" + "*" * 40 + "Finish in GLC" + "*" * 40)

    # //////////////////////// Test the model on Test dataset ////////////////////////
    print("\n" + "*" * 40 + "Evaluating in Test data" + "*" * 40)
    test_model = model.BiLSTM(args)
    test_model = load_model(test_model, args.save_dir, 'best_{}'.format("final"), 0)
    if args.cuda:
        test_model.cuda()

    accuracy, recall, precision, f1 = eval(test_iter, test_model, args)
    fout.write("Hydra Acc: {}, recall: {}, precision: {}, f1: {}".format(str(accuracy), str(recall), str(precision), str(f1)))
    fout.write("\n")

    fout.write("-" * 90 + "END THIS" + "-" * 90 + "\n\n\n")
    fout.close()



