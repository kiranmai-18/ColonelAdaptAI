def apply(cfg):
    cfg = dict(cfg)
    cfg['loss'] = dict(cfg['loss'])
    cfg['loss']['ewc_lambda'] = 0.0
    cfg['loss']['proto_lambda'] = 0.0
    cfg['loss']['trust_lambda'] = 0.0
    cfg['loss']['contrastive_lambda'] = 0.0
    return cfg
