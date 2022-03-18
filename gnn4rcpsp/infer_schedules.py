import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from tqdm import tqdm

from learn_schedules import check_solution, compute_loss

from models import ResTransformer, ResGINE


def test(test_loader, model, device, writer):
    model.eval()
    
    for batch_idx, data in enumerate(tqdm(test_loader)):
        batch_violations = {
            'all_positive_per': [],
            'all_positive_mag': [],
            'precedence_per': [],
            'precedence_mag': [],
            'resource_per': [],
            'resource_mag': [],
            'makespan': [],
        }
        
        data.to(device)
        out = model(data)
        data.out = out
        # TODO: is there a cleaner way to do this?
        data._slice_dict['out'] = data._slice_dict['x']
        data._inc_dict['out'] = data._inc_dict['x']
        
        loss = compute_loss(data=data, device=device)
        writer.add_scalar('loss', loss.item(), batch_idx)
        
        violations = check_solution(data)
        for k, v in violations.items():
            batch_violations[k].append(v)
        
        for k, v in batch_violations.items():
            batch_violations[k] = np.mean(batch_violations[k])
        writer.add_scalar('all_positive_per', batch_violations['all_positive_per'], batch_idx)
        writer.add_scalar('all_positive_mag', batch_violations['all_positive_mag'], batch_idx)
        writer.add_scalar('precedence_per', batch_violations['precedence_per'], batch_idx)
        writer.add_scalar('precedence_mag', batch_violations['precedence_mag'], batch_idx)
        writer.add_scalar('resource_per', batch_violations['resource_per'], batch_idx)
        writer.add_scalar('resource_mag', batch_violations['resource_mag'], batch_idx)
        writer.add_scalar('makespan', batch_violations['makespan'], batch_idx)
    
        writer.flush()


if __name__ == "__main__":
    data_list = torch.load('data_list.tch')
    train_list = torch.load('train_list.tch')
    test_list = list(set(range(len(data_list))) - set(train_list))
    test_loader = DataLoader([data_list[d] for d in test_list], batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Net = ResTransformer
    # Net = ResGINE
    model = Net().to(device)
    model.load_state_dict(torch.load('model_900.tch'))
    run_id = timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    print(f"Run ID: {run_id}")
    writer = SummaryWriter(f's3://iooda-gnn4rcpsp-bucket/tensorboard_logs/{run_id}')
    test(test_loader=test_loader, model=model, device=device, writer=writer)