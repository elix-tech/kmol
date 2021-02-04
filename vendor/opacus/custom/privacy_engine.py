from opacus.privacy_engine import PrivacyEngine as BasePrivacyEngine


class PrivacyEngine(BasePrivacyEngine):

    def step(self):
        """
        :overwrite: opacus.privacy_engine.PrivacyEngine::step()

        The base functionality raises exceptions if one of the layers has a batch size higher than
        the specified batch size. Pytorch geometric uses a dynamic batch size, it makes the functionality
        unusable. As the real batch size is not actually affected, and we still process only a "batch_size"
        number of samples per iteration, there is little risk for the engine to underestimate the privacy loss,
        which is the main reason why the precautions were set. However, there is still a chance of a pessimistic
        estimate if the last batch contains fewer examples.
        """
        self.steps += 1
        self.clipper.clip_and_accumulate()
        clip_values, batch_size = self.clipper.pre_step()

        params = (p for p in self.module.parameters() if p.requires_grad)
        for p, clip_value in zip(params, clip_values):
            noise = self._generate_noise(clip_value, p)
            if self.loss_reduction == "mean":
                noise /= batch_size
            p.grad += noise
