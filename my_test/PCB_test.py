from datasets.PCB.calibration_layer import PrototypicalCalibrationBlock
from main import get_args_parser
import util.misc as utils
import argparse

parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()
utils.init_distributed_mode(args)
pcb = PrototypicalCalibrationBlock(args)