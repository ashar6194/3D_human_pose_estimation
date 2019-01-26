import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--root_dir', default='/media/mcao/Miguel/UBC_hard/train/', type=str,
                    help='THe root training directory for parsing stuff')

parser.add_argument('--test_dir', default='/media/mcao/Miguel/UBC_hard/test/', type=str,
                    help='THe root training directory for parsing stuff')

parser.add_argument('--show_td', default=True, type=bool,
                    help='THe root training directory for parsing stuff')

parser.add_argument('--batch_size', default=32, type=int, help='THe root training directory for parsing stuff')

parser.add_argument('--num_epochs', default=2, type=int, help='THe root training directory for parsing stuff')

parser.add_argument('--input_size', default=100, type=str, help='Size of input image to the network')

args = parser.parse_args()
