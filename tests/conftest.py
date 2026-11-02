import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pytest, torch

from steadyvoice.adapt.common import Ctx, compute_reference
from steadyvoice.data.stream import SessionStream
from steadyvoice.data.synthetic import SynthConfig, make_dataset
from steadyvoice.decode.lm import PhonemeNGramLM
from steadyvoice.models.backbone import freeze_for_adaptation
from steadyvoice.train.backbone_train import TrainCfg, train_backbone

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tiny():
    cfg = SynthConfig(C=16, L=6, n_words=15, n_days=8, n_train_days=3, n_adapt_train=40,
                      n_adapt=12, n_eval=8, seed=3)
    world, days = make_dataset(cfg)
    model, states, _ = train_backbone(days[:3], cfg.n_phon + 1,
                                      TrainCfg(hidden=32, layers=1, steps=60, seed=3), "cpu", log=lambda s: None)
    freeze_for_adaptation(model)
    lm = PhonemeNGramLM(world.corpus(300, 5), cfg.n_phon)
    ctx = Ctx(model, cfg.C, "affine", 8, "cpu", lm)
    ctx.ref = compute_reference(ctx, days[:3], states)
    ctx.hp.update(sv_rounds=1, sv_epochs=1, corp_rounds=1, corp_epochs=1, tent_epochs=1, beam=4, topk=4)
    return dict(cfg=cfg, world=world, days=days, model=model, states=states, ctx=ctx, k=3)
