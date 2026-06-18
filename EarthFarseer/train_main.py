import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import EarthFarseer
from Dataset import NSLoader, WeatherLoader, TaxiLoader, SEVIRLoader
from method.kd import GRAHSC_KD
from utils import calc_mse, calc_rmse, calc_csi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--teacher_ckpt', type=str, default='')
    parser.add_argument('--max_iter', type=int, default=300)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = None
    val_dataset = None
    if 'Navier' in args.data_path:
        train_dataset = NSLoader(args.data_path, train=True)
        val_dataset = NSLoader(args.data_path, train=False)
    elif 'SEVIR' in args.data_path:
        train_dataset = SEVIRLoader(args.data_path, train=True)
        val_dataset = SEVIRLoader(args.data_path, train=False)
    elif 'Weather' in args.data_path:
        train_dataset = WeatherLoader(args.data_path, train=True)
        val_dataset = WeatherLoader(args.data_path, train=False)
    elif 'Taxi' in args.data_path:
        train_dataset = TaxiLoader(args.data_path, train=True)
        val_dataset = TaxiLoader(args.data_path, train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    student_model = EarthFarseer().to(device)
    teacher_model = None
    kd_tool = None
    optimizer = None
    if args.distill:
        teacher_model = EarthFarseer().to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_ckpt, map_location=device))
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        kd_tool = GRAHSC_KD(device, args.max_iter)
        optimizer = kd_tool.get_optimizer(student_model, args.lr)
    else:
        optimizer = Adam(student_model.parameters(), lr=args.lr)
    for epoch in range(args.num_epochs):
        student_model.train()
        total_train_loss = 0.0
        for cur_iter, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            if args.distill:
                with torch.no_grad():
                    teacher_out = teacher_model(x)
                student_out = student_model(x)
                loss, l_pred, l_gra, l_hsc = kd_tool.calculate_total_loss(student_out, teacher_out, y, cur_iter)
            else:
                student_out = student_model(x)
                loss = torch.norm(student_out - y, p='fro') ** 2 / student_out.numel()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        student_model.eval()
        total_val_mse = 0.0
        total_val_csi = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = student_model(x)
                mse = calc_mse(pred, y)
                csi = calc_csi(pred, y)
                total_val_mse += mse
                total_val_csi += csi
        avg_val_mse = total_val_mse / len(val_loader)
        avg_val_csi = total_val_csi / len(val_loader)
        print(f"Epoch:{epoch} TrainLoss:{avg_train_loss:.4f} ValMSE:{avg_val_mse:.4f} ValCSI:{avg_val_csi:.2f}")
    torch.save(student_model.state_dict(), './student_earthfarsser.pth')

if __name__ == "__main__":
    main()