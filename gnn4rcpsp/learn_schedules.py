import random
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.nn import Sequential

import numpy as np
import os

from tqdm import tqdm
import datetime

from graph import load_data
from models import ResTransformer, ResGINE


def compute_loss(data, device):
    data_batch = data
    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    batch_loss = torch.tensor(0., dtype=torch.float32).to(device)
    data_list = data_batch.to_data_list()
    for data in data_list:
        batch_loss += F.mse_loss(data.out[len(data.rc):,0], data.solution_starts)
    return batch_loss / len(data_list)


def check_solution(data):
    data_batch = data
    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    total_violations = {
            'all_positive_per': [],
            'all_positive_mag': [],
            'precedence_per': [],
            'precedence_mag': [],
            'resource_per': [],
            'resource_mag': [],
            'makespan': [],
        }
    for data in data_batch.to_data_list():
        
        t2t, dur, r2t, rc, con, ref_makespan = (
            data.t2t.view(len(data.dur), -1).data.cpu().detach().numpy(),
            data.dur.data.cpu().detach().numpy(),
            data.r2t.view(len(data.rc), len(data.dur)).data.cpu().detach().numpy(),
            data.rc.data.cpu().detach().numpy(),
            data.con.view(*data.con_shape).data.cpu().detach().numpy(),
            data.reference_makespan
        )
        nb_tasks = len(dur)
        nb_res = len(rc)
        s = data.out.data.cpu().detach().numpy()[nb_res:].flatten()
        s = s - s.min()
    
        # All starting times must be positive
        negative_s = (s < 0.0).astype(int)
        all_positive_violations = np.sum(negative_s) / nb_tasks * 100.0
        all_positive_magnitude = np.sum(np.dot(s, negative_s)) / len(negative_s) if len(negative_s) > 0 else 0

        # Precedence constraints
        pred_viol_idx = [(np.array([s[p] + t2t[t][p] - s[t]
                                    for p in range(t2t.shape[1]) if t2t[t][p] > 0]) > 0).astype(int)
                         for t in range(t2t.shape[0])]
        precedence_violations = np.sum([np.sum(pred_viol_idx[t])
                                        for t in range(t2t.shape[0])]) / (t2t > 0).sum() * 100.0
        precedence_magnitude = np.mean([np.dot(np.array([s[p] + t2t[t][p] - s[t]
                                                         for p in range(t2t.shape[1]) if t2t[t][p] > 0]),
                                               pred_viol_idx[t]) / len(pred_viol_idx[t]) if len(pred_viol_idx[t]) > 0 else 0
                                        for t in range(t2t.shape[0])])

        # Resource consumption constraints
        resource_viol_idx = [(np.array([r2t[r][t] +
                                        np.dot((np.array([(s[t] - s[tp])*(s[tp] + dur[tp] - s[t])
                                                            for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]) > 0).astype(int).T,
                                                np.array([r2t[r][tp] for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]))
                                        for t in range(r2t.shape[1]) if r2t[r][t] > 0]) > rc[r]).astype(int)
                             for r in range(r2t.shape[0])]
        resource_violations = np.sum([np.sum(resource_viol_idx[r]) for r in range(r2t.shape[0])]) / (r2t > 0).sum() * 100.0
        resource_magnitude = np.mean([np.dot(np.array([r2t[r][t] - rc[r] +
                                                       np.dot((np.array([(s[t] - s[tp])*(s[tp] + dur[tp] - s[t])
                                                                         for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]) > 0).astype(int).T,
                                                               np.array([r2t[r][tp] for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]))
                                                       for t in range(r2t.shape[1]) if r2t[r][t] > 0]),
                                             resource_viol_idx[r]) / len(resource_viol_idx[r]) if len(resource_viol_idx[r]) > 0 else 0
                                      for r in range(r2t.shape[0])])

        # Makespan cost
        makespan = np.max(s + dur) / ref_makespan #(nb_tasks * nb_res)

        total_violations['all_positive_per'].append(all_positive_violations)
        total_violations['all_positive_mag'].append(all_positive_magnitude)
        total_violations['precedence_per'].append(precedence_violations)
        total_violations['precedence_mag'].append(precedence_magnitude)
        total_violations['resource_per'].append(resource_violations)
        total_violations['resource_mag'].append(resource_magnitude)
        total_violations['makespan'].append(makespan)
        
    for k in total_violations:
        total_violations[k] = np.mean(total_violations[k])
        
    return total_violations


def train(train_loader, model, optimizer, device, writer):
    model.train()
    NUM_EPOCHS = 1000
    step = 0
    for epoch in tqdm(range(NUM_EPOCHS)):
        epoch_violations = {
            'all_positive_per': [],
            'all_positive_mag': [],
            'precedence_per': [],
            'precedence_mag': [],
            'resource_per': [],
            'resource_mag': [],
            'makespan': [],
        }
        COMPUTE_METRICS = True if epoch % 100 == 0 else False

        for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")):
            COMPUTE_DEBUG = True if batch_idx == 0 else False

            data.to(device)
            optimizer.zero_grad()

            if COMPUTE_DEBUG:
                hooks = []
                for name, m in model.named_modules():
                    if type(m) is Sequential:
                        def hook(module, input, output, name=name):
                            # print(output.std().item())
                            # print(output.std(-1))
                            # print(name)
                            writer.add_scalar(f"forward/std/{name}", output.std(), step)
                            writer.add_histogram(f"forward/{name}", output, step)
                        hooks.append(m.register_forward_hook(hook))
            out = model(data)
            data.out = out
            # TODO: is there a cleaner way to do this?
            data._slice_dict['out'] = data._slice_dict['x']
            data._inc_dict['out'] = data._inc_dict['x']
    #         data.cpu()
            
            if COMPUTE_DEBUG:
                for hook in hooks:
                    hook.remove()

            loss = compute_loss(data=data, device=device)

            if COMPUTE_DEBUG:
                if loss.grad_fn is None:
                    continue
                loss.backward(retain_graph=True)
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    writer.add_scalar(f'grads/std/{name}/loss', p.grad.std(), step)
                model.zero_grad()
        
            loss.backward()
            optimizer.step()
            step += 1

            writer.add_scalar('loss', loss.item(), step)
            
            if COMPUTE_METRICS:

                violations = check_solution(data)
                for k, v in violations.items():
                    epoch_violations[k].append(v)

        if COMPUTE_METRICS:
            for k, v in epoch_violations.items():
                epoch_violations[k] = np.mean(epoch_violations[k])
            writer.add_scalar('all_positive_per', epoch_violations['all_positive_per'], step)
            writer.add_scalar('all_positive_mag', epoch_violations['all_positive_mag'], step)
            writer.add_scalar('precedence_per', epoch_violations['precedence_per'], step)
            writer.add_scalar('precedence_mag', epoch_violations['precedence_mag'], step)
            writer.add_scalar('resource_per', epoch_violations['resource_per'], step)
            writer.add_scalar('resource_mag', epoch_violations['resource_mag'], step)
            writer.add_scalar('makespan', epoch_violations['makespan'], step)

            torch.save(model.state_dict(), f"model_{epoch}.tch")

        writer.flush()


if __name__ == "__main__":
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data_list.tch')):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        kobe_rcpsp_dir = os.path.join(root_dir, 'kobe-rcpsp/data/rcpsp')
        solutions_dir = os.path.join(root_dir, 'cp_solutions')
        load_data(kobe_rcpsp_directory=kobe_rcpsp_dir,
                solution_file=os.path.join(solutions_dir, 'cpsat_solutions.json'))
    
    data_list = torch.load('data_list.tch')
    train_list = random.sample(range(len(data_list)), int(0.8 * len(data_list)))
    train_loader = DataLoader([data_list[d] for d in train_list], batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    MAKESPAN_MAX_SCALE = model.scale.max()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-0)#, weight_decay=5e-4)
    run_id = timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    print(f"Run ID: {run_id}")
    # writer = SummaryWriter(f's3://iooda-gnn4rcpsp-bucket/tensorboard_logs/{run_id}')
    writer = SummaryWriter(f'../tensorboard_logs/{run_id}')
    
    torch.save(train_list, './train_list.tch')
    train(train_loader=train_loader, model=model, optimizer=optimizer, device=device, writer=writer)