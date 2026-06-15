from trainers.continual_trainer import ContinualTrainer

class BaselineTrainer(ContinualTrainer):
    """Simple shared trainer. Set loss.ewc_lambda/proto_lambda/trust_lambda to 0 for fine-tuning baseline."""
    pass
