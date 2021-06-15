import numpy as np
import argparse
import kneron_preprocessing

def main_(args):
    image = args.input_file
    filefmt = args.file_fmt
    if filefmt == 'bin':
        raw_format = args.raw_format
        raw_w = args.input_width
        raw_h = args.input_height

        image_data = kneron_preprocessing.API.load_bin(image,raw_format,(raw_w,raw_h))
    else:
        image_data = kneron_preprocessing.API.load_image(image)


    npu_w = args.width
    npu_h = args.height

    crop_first = True if args.crop_first == "True" else False
    if crop_first:
        x1 = args.x_pos
        y1 = args.y_pos
        x2 = args.crop_w + x1
        y2 = args.crop_h + y1
        crop_box = [x1,y1,x2,y2]
    else:
        crop_box = None

    pad_mode = args.pad_mode
    norm_mode = args.norm_mode
    bitwidth = args.bitwidth
    radix = args.radix
    rotate = args.rotate_mode

    ##
    image_data = kneron_preprocessing.API.inproc_520(image_data,npu_size=(npu_w,npu_h),crop_box=crop_box,pad_mode=pad_mode,norm=norm_mode,rotate=rotate,radix=radix,bit_width=bitwidth)

    output_file = args.output_file
    kneron_preprocessing.API.dump_image(image_data,output_file,'bin','rgba')

    return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="preprocessing"
        )

    argparser.add_argument(
        '-i',
        '--input_file',
        help="input file name"
        )

    argparser.add_argument(
        '-ff',
        '--file_fmt',
        help="input file format, jpg or bin"
        )

    argparser.add_argument(
        '-rf',
        '--raw_format',
        help="input file image format, rgb or rgb565 or nir"
        )

    argparser.add_argument(
        '-i_w',
        '--input_width',
        type=int,
        help="input image width"
        )

    argparser.add_argument(
        '-i_h',
        '--input_height',
        type=int,
        help="input image height"
        )

    argparser.add_argument(
        '-o',
        '--output_file',
        help="output file name"
        )

    argparser.add_argument(
        '-s_w',
        '--width',
        type=int,
        help="output width for npu input",
        )

    argparser.add_argument(
        '-s_h',
        '--height',
        type=int,
        help="output height for npu input",
        )

    argparser.add_argument(
        '-c_f',
        '--crop_first',
        help="crop first True or False",
        )

    argparser.add_argument(
        '-x',
        '--x_pos',
        type=int,
        help="left up coordinate x",
        )

    argparser.add_argument(
        '-y',
        '--y_pos',
        type=int,
        help="left up coordinate y",
        )

    argparser.add_argument(
        '-c_w',
        '--crop_w',
        type=int,
        help="crop width",
        )

    argparser.add_argument(
        '-c_h',
        '--crop_h',
        type=int,
        help="crop height",
        )

    argparser.add_argument(
        '-p_m',
        '--pad_mode',
        type=int,
        help=" 0: pad 2 sides, 1: pad 1 side, 2: no pad.",
        )

    argparser.add_argument(
        '-n_m',
        '--norm_mode',
        help="normalizaton mode: yolo, kneron, tf."
        )

    argparser.add_argument(
        '-r_m',
        '--rotate_mode',
        type=int,
        help="rotate mode:0,1,2"
        )

    argparser.add_argument(
        '-bw',
        '--bitwidth',
        type=int,
        help="Int for bitwidth"
        )
    
    argparser.add_argument(
        '-r',
        '--radix',
        type=int,
        help="Int for radix"
        )

    args = argparser.parse_args()
    main_(args)