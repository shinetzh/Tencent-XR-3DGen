from dataclasses import dataclass

import torch
import torch.nn as nn

from vae.utils.config import parse_structured
from vae.utils.misc import get_device, load_module_weights
from vae.utils.typing import *


class Configurable:
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Optional[dict] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)


class Updateable:
    def do_update_step(
        self, epoch: int, global_step: int, on_load_weights: bool = False
    ):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step(
                    epoch, global_step, on_load_weights=on_load_weights
                )
        self.update_step(epoch, global_step, on_load_weights=on_load_weights)

    def do_update_step_end(self, epoch: int, global_step: int):
        for attr in self.__dir__():
            if attr.startswith("_"):
                continue
            try:
                module = getattr(self, attr)
            except:
                continue  # ignore attributes like property, which can't be retrived using getattr?
            if isinstance(module, Updateable):
                module.do_update_step_end(epoch, global_step)
        self.update_step_end(epoch, global_step)

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # override this method to implement custom update logic
        # if on_load_weights is True, you should be careful doing things related to model evaluations,
        # as the models and tensors are not guarenteed to be on the same device
        pass

    def update_step_end(self, epoch: int, global_step: int):
        pass


def update_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step(epoch, global_step)


def update_end_if_possible(module: Any, epoch: int, global_step: int) -> None:
    if isinstance(module, Updateable):
        module.do_update_step_end(epoch, global_step)


class BaseObject(Updateable):
    @dataclass
    class Config:
        pass

    cfg: Config  # add this to every subclass of BaseObject to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)

    def configure(self, *args, **kwargs) -> None:
        pass


class BaseModule(nn.Module, Updateable):
    @dataclass
    class Config:
        weights: Optional[str] = None

    cfg: Config  # add this to every subclass of BaseModule to enable static type checking

    def __init__(
        self, cfg: Optional[Union[dict, DictConfig]] = None, *args, **kwargs
    ) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.device = get_device()
        self.configure(*args, **kwargs)
        if self.cfg.weights is not None:
            # breakpoint()
            # format: path/to/weights:module_name
            weights_path, module_name = self.cfg.weights.split(":")
            state_dict, epoch, global_step = load_module_weights(
                weights_path, module_name=module_name, map_location="cpu"
            )
            # breakpoint()
            # if we update query latent shape in vae
            try:
                if self.encoder.query.shape != state_dict['encoder.query'].shape:
                    old_query_size = state_dict['encoder.query'].shape[0]
                    new_query_size = self.encoder.query.shape[0]
                    
                    tmp_encoder_query = state_dict['encoder.query'].data
                    del state_dict['encoder.query']

                    self.load_state_dict(state_dict, strict=False)
                    # update query latents
                    with torch.no_grad():
                        self.encoder.query = torch.nn.Parameter(torch.zeros_like(self.encoder.query)+1e-4)
                        self.encoder.query[:old_query_size, :] = tmp_encoder_query
                                            
                else:
                    self.load_state_dict(state_dict)
            except:
                self.load_state_dict(state_dict, strict=False)

            self.do_update_step(
                epoch, global_step, on_load_weights=True
            )  # restore states
        # dummy tensor to indicate model state
        self._dummy: Float[Tensor, "..."]
        self.register_buffer("_dummy", torch.zeros(0).float(), persistent=False)

    def configure(self, *args, **kwargs) -> None:
        pass
