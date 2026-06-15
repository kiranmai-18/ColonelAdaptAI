import argparse, copy, yaml
from pathlib import Path
from utils.seed import set_seed, get_device
from utils.logger import ExperimentLogger
from data.datasets import build_tasks
from models.infoperturbnet import InfoPerturbNet
from trainers.continual_trainer import ContinualTrainer
from baselines.rehearsal import RehearsalBaselineTrainer

SUPPORTED_METHODS = ['colonel', 'infoperturbnet', 'finetune', 'replay', 'agem', 'gem', 'der', 'derpp', 'mir']

def load_cfg(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

def build_model(cfg, input_shape, num_tasks):
    return InfoPerturbNet(
        input_shape=input_shape,
        backbone_name=cfg['model']['backbone'],
        max_tasks=max(cfg['model']['max_tasks'], num_tasks),
        task_embedding_dim=cfg['model']['task_embedding_dim'],
        perturb_init_std=cfg['model']['perturb_init_std']
    )

def main():
    p=argparse.ArgumentParser(description='ColonelAdaptAI / InfoPerturbNet continual learning runner')
    p.add_argument('--config', default='config.yaml')
    p.add_argument('--dataset', default=None, choices=['split_mnist','permuted_mnist','split_cifar100'])
    p.add_argument('--method', default=None, choices=SUPPORTED_METHODS, help='colonel/infoperturbnet or baseline method')
    p.add_argument('--run-baselines', action='store_true', help='run all methods listed in baselines.methods')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--output-dir', default=None)
    args=p.parse_args()
    cfg=load_cfg(args.config)
    if args.dataset: cfg['experiment']['dataset']=args.dataset
    if args.epochs is not None: cfg['training']['epochs_per_task']=args.epochs
    if args.batch_size is not None: cfg['training']['batch_size']=args.batch_size
    if args.output_dir: cfg['experiment']['output_dir']=args.output_dir
    if args.method: cfg.setdefault('experiment', {})['method'] = args.method
    set_seed(cfg.get('seed',42)); device=get_device(cfg.get('device','auto'))
    logger=ExperimentLogger(cfg['experiment']['output_dir'])
    logger.log(f'Using device: {device}')
    tasks,input_shape = build_tasks(cfg['experiment']['dataset'], batch_size=cfg['training']['batch_size'], num_workers=cfg['experiment'].get('num_workers',2), seed=cfg.get('seed',42))
    logger.log(f'Loaded {len(tasks)} tasks from {cfg["experiment"]["dataset"]}')

    if args.run_baselines or cfg.get('baselines', {}).get('run', False):
        methods = cfg.get('baselines', {}).get('methods', ['finetune','replay','agem','der','derpp','mir'])
        for method in methods:
            set_seed(cfg.get('seed',42))
            logger.log(f'Running baseline: {method}')
            model = build_model(cfg, input_shape, len(tasks))
            RehearsalBaselineTrainer(model, tasks, cfg, device, logger, method).fit()
        logger.log('Finished baseline suite.')
        return

    method = cfg['experiment'].get('method', 'colonel').lower()
    model = build_model(cfg, input_shape, len(tasks))
    if method in {'colonel', 'infoperturbnet'}:
        trainer=ContinualTrainer(model, tasks, cfg, device, logger)
    else:
        trainer=RehearsalBaselineTrainer(model, tasks, cfg, device, logger, method)
    trainer.fit()
    logger.save_json(cfg, 'used_config.json')
    logger.log('Finished. Results saved in output directory.')

if __name__ == '__main__': main()
