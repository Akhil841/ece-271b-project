
import argparse
import os


def params():
    parser = argparse.ArgumentParser()

    # Experiment options

    parser.add_argument("--task", default="dl", type=str,\
                help="Deep-Leanring Approach;\n\
                      Statisitcal Learning Approacht;", choices=['dl','statistical'])


    # optional fine-tuning techiques parameters
    parser.add_argument("--reinit_n_layers", default=3, type=int,
                help="number of layers that are reinitialized. Count from last to first.")

    # Others
    parser.add_argument("--input-dir", default='data', type=str,
                help="The input training data directory.")
    parser.add_argument("--output-dir", default='results', type=str,
                help="Output directory where the model predictions and checkpoints are written.")
    parser.add_argument("--model", default='bert', type=str,
                help="The model architecture to be trained or fine-tuned.")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dataset", default="reddit_comment_body_dec_2024", type=str,
                help="dataset", choices=['reddit_comment_body_dec_2024', 'reddit_comments_dec_2024'])
    parser.add_argument("--save-every", default=3, type=int,
                help="How often to save the model checkpoints.")


    # Key settings
    parser.add_argument("--ignore-cache", action="store_true",
                help="Whether to ignore cache and create a new input data")
    parser.add_argument("--debug", action="store_true",
                help="Whether to run in debug mode which is exponentially faster")
    parser.add_argument("--do-train", action="store_true",
                help="Whether to run training.")
    parser.add_argument("--do-eval", action="store_true",
                help="Whether to run eval on the dev set.")

    # Hyper-parameters for tuning
    parser.add_argument("--batch-size", default=512, type=int,
                help="Batch size per GPU/CPU for training and evaluation.")
    parser.add_argument("--learning-rate", default=1e-2, type=float,
                help="Model learning rate starting point.")
    parser.add_argument("--hidden-dim", default=768, type=int,
                help="Model hidden dimension.")
    parser.add_argument("--drop-rate", default=0.9, type=float,
                help="Dropout rate for model training")
    parser.add_argument("--embed-dim", default=768, type=int,
                help="The embedding dimension of pretrained LM.")
    parser.add_argument("--adam-epsilon", default=1e-8, type=float,
                help="Epsilon for Adam optimizer.")
    parser.add_argument("--n-epochs", default=50, type=int,
                help="Total number of training epochs to perform.")
    parser.add_argument("--max-len", default=20, type=int,
                help="maximum sequence length to look back")
    parser.add_argument("--weight-decay", default=1e-4, type=int,
                help="maximum sequence length to look back")


    args = parser.parse_args()
    return args