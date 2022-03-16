import torch
import torch.nn.functional as F
from torch.utils.tensorboard._convert_np import make_np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import TransformerConv, DenseGCNConv, Sequential

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from math import sqrt
import os

from tqdm import tqdm

from graph import load_data
from models import ResTransformer, ResGINE


def check_solution(data):
    data_batch = data
    # Iterate over batch elements for simplicity
    # TODO: vectorize?
    total_violations = {
            'all_positive': [],
            'precedence': [],
            'makespan': [],
            'resource': [],
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
        s = data.out.data.cpu().detach().numpy()[nb_res:]
        s = s - s.min()
    
        # All starting times must be positive
        all_positive_violations = np.sum((s < 0.0).astype(int)) / nb_tasks * 100.0

        # Precedence constraints
        precedence_violations = np.sum([np.sum((np.array([s[p] + t2t[t][p] - s[t]
                                                          for p in range(t2t.shape[1]) if t2t[t][p] > 0]) > 0).astype(int))
                                        for t in range(t2t.shape[0])]) / (t2t > 0).sum() * 100.0

        # Resource consumption constraints
        resource_violations = np.sum([
            np.sum((np.array([
                r2t[r][t] + np.dot((np.array([(s[t] - s[tp])*(s[tp] + dur[tp] - s[t])
                                     for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]) > 0).astype(int).T,
                                   np.array([r2t[r][tp] for tp in range(r2t.shape[1]) if tp != t and r2t[r][tp] > 0]))
                for t in range(r2t.shape[1]) if r2t[r][t] > 0
             ]) > rc[r]).astype(int))
            for r in range(r2t.shape[0])]) / (r2t > 0).sum() * 100.0

        # Makespan cost
        makespan = np.max(s + dur) / ref_makespan #(nb_tasks * nb_res)

        total_violations['all_positive'].append(all_positive_violations)
        total_violations['precedence'].append(precedence_violations)
        total_violations['resource'].append(resource_violations)
        total_violations['makespan'].append(makespan)
        
    for k in total_violations:
        total_violations[k] = np.mean(total_violations[k])
        
    return total_violations


def train(train_loader, model, optimizer):
    model.train()
    NUM_EPOCHS = 1000
    step = 0
    for epoch in tqdm(range(NUM_EPOCHS)):
        epoch_violations = {
            'all_positive': [],
            'precedence': [],
            'makespan': [],
            'resource': [],
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
            data.__slices__['out'] = data.__slices__['x']
            data.__cat_dims__['out'] = data.__cat_dims__['x']
            data.__cumsum__['out'] = data.__cumsum__['x']
    #         data.cpu()
            
            if COMPUTE_DEBUG:
                for hook in hooks:
                    hook.remove()

            loss = F.mse_loss(data.out, data.solution_starts)

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
            writer.add_scalar('all_positive', epoch_violations['all_positive'], step)
            writer.add_scalar('precedence', epoch_violations['precedence'], step)
            writer.add_scalar('resource', epoch_violations['resource'], step)
            writer.add_scalar('makespan', epoch_violations['makespan'], step)

            torch.save(model.state_dict(), f"model_{epoch}.tch")

        writer.flush()

import datetime

if __name__ == "__main__":
    
    if not os.path.exists(os.path.join(os.path.dirname(__file__), 'data_list.tch')):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        kobe_rcpsp_dir = os.path.join(root_dir, 'kobe-rcpsp/data/rcpsp')
        solutions_dir = os.path.join(root_dir, 'cp_solutions')
        load_data(kobe_rcpsp_directory=kobe_rcpsp_dir,
                solution_file=os.path.join(solutions_dir, 'postpro_benchmark_merged_single_modes.json'))
    
    data_list = torch.load('data_list.tch')
    train_loader = DataLoader(data_list, batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    MAKESPAN_MAX_SCALE = model.scale.max()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=5e-4)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-0)#, weight_decay=5e-4)
    run_id = timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    print(f"Run ID: {run_id}")
    writer = SummaryWriter(f's3://iooda-gnn4rcpsp-bucket/tensorboard_logs/{run_id}')
    # writer = SummaryWriter(f's3://iooda-gnn4rcpsp-bucket/xai_experiments/no_makespan')
    
    train(train_loader=train_loader, model=model, optimizer=optimizer)