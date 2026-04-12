#!/usr/bin/env python

# ==============================================================================
# Copyright (c) 2022, Toda Laboratory, Nagoya University
# Author: Wen-Chin Huang (wen.chinhuang@g.sp.m.is.nagoya-u.ac.jp)
# All rights reserved.
# ==============================================================================

import sys
import os
import os.path

import numpy as np
import scipy
import scipy.stats
import csv

MAX_SCORE=100
MIN_SCORE=0

def calculate_scores(truth, submission_answer, prefix):

    # sanity check
    sanity_ok = True
    diff = []
    for wav_name in truth:
        if wav_name not in submission_answer:
            diff.append(wav_name)
            sanity_ok = False
    if not sanity_ok:
        print("Sanity check for {} track failed. These files are not in submission:".format(prefix))
        print(diff)
        return {
            prefix + "_UTT_MSE": MAX_SCORE,
            prefix + "_UTT_LCC": MIN_SCORE,
            prefix + "_UTT_SRCC": MIN_SCORE,
            prefix + "_UTT_KTAU": MIN_SCORE,
            prefix + "_SYS_MSE": MAX_SCORE,
            prefix + "_SYS_LCC": MIN_SCORE,
            prefix + "_SYS_SRCC": MIN_SCORE,
            prefix + "_SYS_KTAU": MIN_SCORE,
        }
    else:
        print("Sanity check for {} track succeeded.".format(prefix))
    
    # utterance level scores
    sorted_truth = np.array([truth[k] for k in sorted(truth)])
    sorted_submission_answer = np.array([submission_answer[k] for k in sorted(submission_answer) if k in truth])
    UTT_MSE=np.mean((sorted_truth-sorted_submission_answer)**2)
    UTT_LCC=np.corrcoef(sorted_truth, sorted_submission_answer)[0][1]
    UTT_SRCC=scipy.stats.spearmanr(sorted_truth, sorted_submission_answer)[0]
    UTT_KTAU=scipy.stats.kendalltau(sorted_truth, sorted_submission_answer)[0]

    # system level scores
    sorted_system_list = sorted(list(set([k.split("-")[0] for k in truth.keys()])))
    sys_truth = {system: [v for k, v in truth.items() if k.startswith(system)] for system in sorted_system_list}
    sys_submission = {system: [v for k, v in submission_answer.items() if k.startswith(system)] for system in sorted_system_list}
    sorted_sys_truth = np.array([np.mean(group) for group in sys_truth.values()])
    sorted_sys_submission = np.array([np.mean(group) for group in sys_submission.values()])
    SYS_MSE=np.mean((sorted_sys_truth-sorted_sys_submission)**2)
    SYS_LCC=np.corrcoef(sorted_sys_truth, sorted_sys_submission)[0][1]
    SYS_SRCC=scipy.stats.spearmanr(sorted_sys_truth, sorted_sys_submission)[0]
    SYS_KTAU=scipy.stats.kendalltau(sorted_sys_truth, sorted_sys_submission)[0]

    return {
        prefix + "_UTT_MSE": UTT_MSE,
        prefix + "_UTT_LCC": UTT_LCC,
        prefix + "_UTT_SRCC": UTT_SRCC,
        prefix + "_UTT_KTAU": UTT_KTAU,
        prefix + "_SYS_MSE": SYS_MSE,
        prefix + "_SYS_LCC": SYS_LCC,
        prefix + "_SYS_SRCC": SYS_SRCC,
        prefix + "_SYS_KTAU": SYS_KTAU,
    }


def read_file(filepath):
    with open(filepath, "r") as csvfile:
        rows = list(csv.reader(csvfile))
    return {os.path.splitext(row[0])[0]: float(row[1]) for row in rows}

input_dir = sys.argv[1]
output_dir = sys.argv[2]

submit_dir = os.path.join(input_dir, 'res')
truth_dir = os.path.join(input_dir, 'ref')

if not os.path.isdir(submit_dir):
    print("{} doesn't exist".format(submit_dir))

if os.path.isdir(submit_dir) and os.path.isdir(truth_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # open the score file
    output_filename = os.path.join(output_dir, 'scores.txt')
    output_file = open(output_filename, 'w')

    # open the truth file
    bvcc_truth_file = os.path.join(truth_dir, "BVCC_answer.txt")
    ood_truth_file = os.path.join(truth_dir, "OOD_answer.txt")
    bvcc_truth = read_file(bvcc_truth_file)
    ood_truth = read_file(ood_truth_file)

    # open the submission file
    submission_answer_file = os.path.join(submit_dir, "answer.txt")
    submission_answer = read_file(submission_answer_file)

    bvcc_scores = calculate_scores(bvcc_truth, submission_answer, "MAIN")
    ood_scores = calculate_scores(ood_truth, submission_answer, "OOD")

    for k, v, in bvcc_scores.items():
        output_file.write("{}:{}\n".format(k, v))
    for k, v, in ood_scores.items():
        output_file.write("{}:{}\n".format(k, v))

    output_file.close()
