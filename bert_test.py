# pytorch-transformers, tensorflowの読み込み
import os, sys
from transformers import BertForPreTraining, BertTokenizer
import tensorflow as tf

# hottoSNS-bertの読み込み
sys.path.append("/home/narita/hottoSNS-bert/src/")
import tokenization
from preprocess import normalizer