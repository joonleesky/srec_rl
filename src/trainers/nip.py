from .base import AbstractTrainer


class NIPTrainer(AbstractTrainer):
    @classmethod
    def code(cls):
        return 'nip'
    
    @classmethod
    def _create_criterion(self):
        pass

    @classmethod
    def calculate_loss(self, batch):
        pass

    @classmethod
    def calculate_metrics(self, batch):
        pass