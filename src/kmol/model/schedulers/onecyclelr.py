from torch.optim.lr_scheduler import OneCycleLR

from ...core.observers import AbstractEventHandler, EventManager
from ...core.helpers import Namespace


class OneCycleLRWrapper(OneCycleLR):
    """Wrapper for OneCycleLR. The goal is to reset total_steps to allow additional training if required
    Reference: https://discuss.pytorch.org/t/lr-scheduler-onecyclelr-causing-tried-to-step-57082-times-the-specified-number-of-total-steps-is-57080/90083/3
    """

    _TOTAL_STEPS: int

    class BeforeLoadEventHandler(AbstractEventHandler):
        def run(self, payload: Namespace):
            OneCycleLRWrapper._TOTAL_STEPS = payload.executor.scheduler.total_steps

    class AfterLoadEventHandler(AbstractEventHandler):
        def run(self, payload: Namespace):
            payload.executor.scheduler.total_steps = OneCycleLRWrapper._TOTAL_STEPS

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        three_phase=False,
        last_epoch=-1,
        verbose=False,
    ):
        super(OneCycleLRWrapper, self).__init__(
            optimizer,
            max_lr,
            total_steps,
            epochs,
            steps_per_epoch,
            pct_start,
            anneal_strategy,
            cycle_momentum,
            base_momentum,
            max_momentum,
            div_factor,
            final_div_factor,
            three_phase,
            last_epoch,
            verbose,
        )
        before_handler = self.BeforeLoadEventHandler()
        after_handler = self.AfterLoadEventHandler()
        EventManager.add_event_listener("before_checkpoint_load", before_handler)
        EventManager.add_event_listener("after_checkpoint_load", after_handler)
