from argparse import ArgumentParser
from dataset.draw_face import draw_face


parser = ArgumentParser()

parser.add_argument("--dataset_raw", default="", type=str)
parser.add_argument("--dataset", default="", type=str)
parser.add_argument("--ctx", default=1024, type=int)
parser.add_argument("--enable_f0", default=0, type=int)
parser.add_argument("--enable_vo", default=0, type=int)
parser.add_argument("--enable_hubert", default=0, type=int)
parser.add_argument("--enable_face", default=1, type=int)
parser.add_argument("--enable_logit", default=0, type=int)
parser.add_argument("--enable_text", default=0, type=int)
parser.add_argument("--enable_phoneme", default=0, type=int)
args = parser.parse_args()

in_path = args.dataset_raw
out_path = args.dataset
ctx = args.ctx

if args.enable_face:
    draw_face(in_path, out_path, ctx)