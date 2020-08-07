import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from models.ComposerVAE import InfoVAE
from datasets.collection import *

_MODELS = dict(InfoVAE=InfoVAE)
_DATASETS = dict(BigMIDI=BigMIDISet,
                 VideoGameMIDI=VideoGameMIDI)

distributed = False

def main(args):

    """ Main training routine specific for this project. """

    # 1 INIT LIGHTNING MODEL

    model = ComposerVAE(**vars(args))

    # 2 INIT TRAINER

    trainer = Trainer.from_argparse_args(args)

    # 3 START TRAINING

    trainer.fit(model)


def run_cli():

    # ------------------------

    # TRAINING ARGUMENTS

    # ------------------------

    # these are project-wide arguments

    root_dir = os.path.dirname(os.path.realpath(__file__))

    parent_parser = ArgumentParser(add_help=False)



    # each LightningModule defines arguments relevant to it

    parser = ComposerVAE.add_model_specific_args(parent_parser, root_dir)

    parser = Trainer.add_argparse_args(parser)

    parser.set_defaults(gpus=2)

    args = parser.parse_args()



    # ---------------------

    # RUN TRAINING

    # ---------------------

    main(args)





if __name__ == '__main__':

    run_cli()