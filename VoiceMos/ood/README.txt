=====================================================================
=== README: VoiceMOS Challenge training phase OOD track resources ===
=====================================================================

This package contains scripts for getting and preprocessing the samples for the
out-of-domain (OOD) track of the VOiceMOS Challenge.
"The VoiceMOS Challenge 2022," Wen-Chin Huang, Erica Cooper, Yu Tsao, Hsin-Min
Wang, Tomoki Toda, Junichi Yamagishi.  https://arxiv.org/abs/2203.11389

The data for the OOD track comes from the Blizzard Challenge 2019:
"The Blizzard Challenge 2019," Zhizheng Wu, Zhihang Xie, and Simon King.
Proc. Blizzard Challenge Workshop, 2019.

We do NOT distribute the samples since this is not permitted by their terms of
use.  Instead, please use these scripts to download and preprocess them.
By running these scripts and downloading the data, you are agreeing to the
Blizzard terms of use, and agree to NOT redistribute this data.
For more information on the Blizzard terms of use, please check here:
https://www.cstr.ed.ac.uk/projects/blizzard/data.html

Please read these instructions carefully.  You need to follow these steps to
obtain ALL of the data that was available in the challenge.


====================
=== Dependencies ===
====================

Please run these scripts using Python 3.
You need basic command line utilities such as make and wget.
Please make sure that sox and sv56demo are on your system path.
If you don't already have them, they can be obtained here:
 * SoX: http://sox.sourceforge.net
 * sv56: https://github.com/openitu/STL/tree/dev/src/sv56
   or use `install_sv56.sh` in this directory which downloads and compiles sv56.

PLEASE NOTE: The size of the final dataset is 4.4GB, and additional storage is
required for temporary files.  Please make sure that you have sufficient
storage.


=========================
=== Download the data ===
=========================

The terms of use of the Blizzard Challenge data stipulate that the data may NOT
be redistributed.  Some individual teams also stipulate that their samples may
be used for research purposes only, and may not be used for commercial purposes.
By using the data for the MOS Prediction Challenge, you are agreeing to abide by
the terms of use of the Blizzard data.  You may not redistribute the samples or
use them for commercial purposes.  For more information, please see:
https://www.cstr.ed.ac.uk/projects/blizzard/data.html

It takes some time to download the archived data.  Please make sure that you
have a stable internet connection, and please be patient!

Launch download and extraction of the archive by running:
  python 01_get.py
  

=====================================
=== Gather and preprocess samples ===
=====================================

1. Run:
     python 02_gather.py

2. Run: 
     python 03_preprocess.py

This script runs downsampling to 16kHz on the selected Blizzard samples,
followed by sv56 amplitude normalization.

Final processed audio data for the challenge can be found in
  DATA/wav/*.wav


=========================================================
=== Labeled train / unlabeled train / dev / test sets ===
=========================================================

Training and development partitions can be found in
  DATA/sets

train_mos_list.txt and val_mos_list.txt contain the list of wav files and their
AVERAGED MOS scores over all of the 10-17 listeners who rated them.

unlabeled_mos_list.txt contains the names of the .wav files in the unlabeled
dataset, but not their MOS ratings.  The ratings can be found in
unlabeled_mos_list_LABELS.txt.

All of the *_mos_list.txt sets are disjoint sets.

TRAINSET and DEVSET contain the individual ratings from each rater, along with
some demographic information for the rater.  (TRAINSET only contains information
about the labeled training data, not for the unlabeled samples.)
The format is as follows:

  sysID,uttID,rating,ignore,listenerinfo

The listener info is as follows:

  {}_na_LISTENERID_na_na_na_LISTENERTYPE

LISTENERTYPE may take the following values:
  EE: speech experts
  EP: paid listeners, native speakers of Chinese (any dialect)
  ER: voluntary listeners

DEVSET and val_mos_list.txt are the same set (the terms dev and val are used
interchangeably here).


============================
=== Secret mapping files ===
============================

The files `secret_sys_mappings.txt` and `secret_utt_mappings.txt` show the
mappings from the original filenames to their obfuscated ones.  Since the MOS
ratings (including for the test set) can be found in the publicly-available
Blizzard Challenge distribution, we obfuscated the filenames to make cheating
more difficult.  If you want to use information about the different synthesis
systems, you can check these mapping files to get the information about which
Blizzard team's system each sample comes from.


========================
=== Acknowledgements ===
========================

Many thanks to the organizers of the Blizzard Challenge 2019, who have kindly
made these samples available.

This work is supported by JST CREST grants JPMJCR18A6, JPMJCR20D3, and
JPMJCR19A3, and by MEXT KAKENHI grants 21K11951 and 21K19808.


===============
=== Authors ===
===============

Challenge organizers:
Wen-Chin Huang(1), Erica Cooper(2), Yu Tsao(3), Hsin-Min Wang(3),
Tomoki Toda(1), Junichi Yamagishi(2)

(1) Nagoya University, Japan
(2) National Institute of Informatics, Japan
(3) Academia Sinica, Taiwan
