import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', default='/media/mcao/Miguel/UBC_hard/valid/', type=str,
                    help='THe root directory for parsing stuff')

args = parser.parse_args()
