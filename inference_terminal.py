#!/usr/bin/env python
#coding: utf-8
import argparse
import sys
import inference
import os
''' script for single image or batch predicition. Accepts folder or single file path'''
def accept_artguments():
    script_path = sys.path[0]
    my_parser = argparse.ArgumentParser(description='Inference of single image or a folder')
    my_parser.add_argument('--image_path',
                       type=str,
                       help='Path to single image or folder')

    my_parser.add_argument('--model_path',
                       type=str,
                       help='Path to trained model')

    my_parser.add_argument('--result_path',
                       type=str,
                       help='Optional path for csv to save results',
                  default = None)

    my_parser.add_argument('--show_results',
			action='store_true',
                       default=False,
                       help='Showing resulting figure for single image or histogram for batch')

    args = my_parser.parse_args()

    img_path = args.image_path
    model_path = args.model_path
    result_path = args.result_path
    show_results = args.show_results

    if os.path.isfile(img_path):
        model, categories = inference.load_model(model_path)
        prediction = inference.show_img_with_pred(img_path, model, categories, show_results)
        return prediction
    else:
        model, categories = inference.load_model(model_path)
        df = inference.batch_prediction(img_path, model, categories, show_results, result_path)
        return df


if __name__ == '__main__':
    prediction = accept_artguments()
    print(prediction)
