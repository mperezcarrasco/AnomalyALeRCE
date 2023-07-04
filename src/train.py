

from src.models.main import build_network
from src.evaluate import evaluation
from src.utils.utils import weights_init_normal, EarlyStopping, save_metrics, print_progress, print_and_log, pretrain_and_setC
import pickle

from torch import optim
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

def train(args, writer, dataloader_train, dataloader_val):
    """Train the unsupervised model."""
    model = build_network(args).to(args.device)
    if args.model in ['deepsvdd', 'classvdd']:
        
        model = pretrain_and_setC(args, model, dataloader_train)
    else:
        model.apply(weights_init_normal)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Setting the early stopping.
    es = EarlyStopping(args)
    for epoch in range(args.epochs):
        model.set_metrics()
        print('Epoch: {}/{}'.format(epoch, args.epochs))
        for _, x, y_cl , _ in dataloader_train:
            model.train()
            x = x.float().to(args.device)
            y_cl = y_cl.long().to(args.device)

            if args.model in ['classvdd']:
                loss = model.compute_loss(x, y_cl)
            else:
                loss = model.compute_loss(x)
            metrics = model.compute_metrics(x)

            #Computing gradients
            loss.backward()
            optimizer.step()

            #Zero grading for next iteration.
            optimizer.zero_grad()

        #print_progress(metrics, writer, epoch, 'train')
        print_progress(epoch, args.epochs,  'Train', writer, metrics)
        losses_v, metrics_v = evaluation(args, model, dataloader_val)
        print_progress(epoch, args.epochs, 'Val', writer, metrics_v )
        stop, is_best = es.count(losses_v.avg, model)

        if is_best:
            save_metrics(metrics, args.directory, mode='train')
            save_metrics(metrics_v, args.directory, mode='val')
        if stop:
            break


def train_ML(args, dataloader_train):
    if args.model == 'iforest':
        clf = IsolationForest(n_estimators=100, max_samples=256, contamination=0.001).fit(dataloader_train)
    else: clf = OneClassSVM(kernel='rbf', nu=0.01).fit(dataloader_train)
    pickle.dump(clf, open('{}/model.pkl'.format(args.directory), 'wb'))
    
    return clf



