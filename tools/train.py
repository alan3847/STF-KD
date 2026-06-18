import os
import argparse
import torch
from torch.utils.data import DataLoader
from openstl.api import create_dataset, create_model
from openstl.core import metric, Recorder
from openstl.method.kd import GRAHSC_KD

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mmnist')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('--ex_name', type=str, default='exp')
    parser.add_argument('--method', type=str, default='simvp')
    parser.add_argument('--teacher_ckpt', type=str, default='')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--max_iterations', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', default='cuda')
    return parser

def main():
    args = get_args_parser().parse_args()
    recorder = Recorder(args.ex_name)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_dataset, val_dataset, test_dataset = create_dataset(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    student_model = create_model(args, device=device)
    teacher_model = None
    kd_handler = None
    optimizer = None
    if args.distill:
        teacher_model = create_model(args, device=device)
        teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        kd_handler = GRAHSC_KD(args, device)
        optimizer = kd_handler.configure_optimizers(student_model)
    else:
        optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        student_model.train()
        train_total = 0.0
        for cur_iter, batch in enumerate(train_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)
            if args.distill:
                _, log_dict = kd_handler.training_one_step(batch, teacher_model, student_model, cur_iter)
                loss = log_dict['train/total_loss']
                train_total += loss
            else:
                pred = student_model(x)
                loss = torch.norm(pred - y, p='fro') ** 2 / pred.numel()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_total += loss.item()
        avg_train = train_total / len(train_loader)
        student_model.eval()
        val_mse_sum = 0.0
        for batch in val_loader:
            x, y = batch
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                if args.distill:
                    log_dict = kd_handler.validation_one_step(batch, teacher_model, student_model)
                    val_mse = log_dict['val/mse']
                else:
                    pred = student_model(x)
                    val_mse = metric.mse(pred, y)
                val_mse_sum += val_mse
        avg_val_mse = val_mse_sum / len(val_loader)
        print(f'Epoch {epoch:03d} TrainLoss:{avg_train:.4f} ValMSE:{avg_val_mse:.4f}')
    recorder.save_model(student_model, save_path=f'./ckpt_{args.ex_name}.pth')

if __name__ == '__main__':
    main()