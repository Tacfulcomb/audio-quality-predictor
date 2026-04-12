=========================================================
=== README: VoiceMOS Challenge Main Track Data (BVCC) ===
=========================================================

This package contains the preprocessed samples for the VoiceMOS Challenge that
we are allowed to distribute.  We are allowed to distribute samples from Voice
Conversion Challenges and ESPnet-TTS, so those are already included in this
distribution.

We do not distribute the Blizzard Challenge samples since this is not permitted
by the terms of use.  Instead, please use the included scripts to download and
preprocess them.

Please read these instructions carefully.  You need to follow these steps to
obtain ALL of the data needed for the training phase of the challenge.


The VoiceMOS Challenge:

"The VoiceMOS Challenge 2022," Wen-Chin Huang, Erica Cooper, Yu Tsao, Hsin-Min
Wang, Tomoki Toda, Junichi Yamagishi.  https://arxiv.org/abs/2203.11389


The Blizzard Challenges 2008, 2009, 2010, 2011, 2013, 2016:

V. Karaiskos, S. King, R. A. Clark, and C. Mayo, "The Blizzard Challenge 2008," in Proc. Blizzard Challenge Workshop, 2008.

A. W. Black, S. King, and K. Tokuda, "The Blizzard Challenge 2009," in Proc. Blizzard Challenge, 2009.

S. King and V. Karaiskos, "The Blizzard Challenge 2010," 2010.

S. King and V. Karaiskos, "The Blizzard Challenge 2011," 2011.

S. King and V. Karaiskos, "The Blizzard Challenge 2013," 2013.

S. King and V. Karaiskos, "The Blizzard Challenge 2016," 2016.


The Voice Conversion Challenges 2016, 2018, and 2020:

T. Toda, L.-H. Chen, D. Saito, F. Villavicencio, M. Wester, Z. Wu, and J. Yamagishi, "The Voice Conversion Challenge 2016," Interspeech, 2016.

J. Lorenzo-Trueba, J. Yamagishi, T. Toda, D. Saito, F. Villavicencio, T. Kinnunen, and Z. Ling, "The Voice Conversion Challenge 2018: Promoting development of parallel and nonparallel methods."

Z. Yi, W.-C. Huang, X. Tian, J. Yamagishi, R. K. Das, T. Kinnunen, Z. Ling, and T. Toda, "Voice Conversion Challenge 2020 — intra-lingual semi-parallel and cross-lingual voice conversion —," in Proc. Joint Workshop for the Blizzard Challenge and Voice Conversion Challenge 2020, 2020, pp. 80–98.


ESPnet-TTS:

S. Watanabe, T. Hori, S. Karita, T. Hayashi, J. Nishitoba, Y. Unno, N. Enrique Yalta Soplin, J. Heymann, M. Wiesner, N. Chen, A. Renduchintala, and T. Ochiai, "ESPnet: End-to-end speech processing toolkit," in Proceedings of Interspeech, 2018, pp. 2207–2211. [Online]. Available: http://dx.doi.org/10.21437/ Interspeech.2018- 1456


========================
=== The BVCC Dataset ===
========================

This dataset is named "BVCC," since it contains generated samples from past
Blizzard and Voice Conversion Challenges.  It also contains samples which have
been made public by ESPnet-TTS.  For more details about the dataset and how it
was collected, please see:

"How do Voices from Past Speech Synthesis Challenges Compare Today?"
Erica Cooper, Junichi Yamagishi
ISCA Speech Synthesis Workshop 2021
https://arxiv.org/abs/2105.02373

Because this dataset is composed of generated samples from multiple past
challenges, and because the terms of use of the Blizzard Challenge system
samples are that they may not be redistributed, you need to download data for
each challenge one by one from the servers where they are hosted.  Then, the
samples in the downloaded data need to be preprocessed in the same manner as we
used for our listening test and other experiments.  This is why the downloading
process has many steps and takes a long time.


====================
=== Dependencies ===
====================

 * Python 3
 Please use Python 3 to run these python scripts.

 * SoX
 SoX is used to preprocess audio by downsampling it to 16kHz.
 SoX can be obtained here: http://sox.sourceforge.net
 Please install it by yourself if you don't already have it.
 You also need to make sure it is on your system path.  To check this, run
   which sox
 and make sure it shows the location of the sox binary.

 * sv56
 sv56 is used for amplitude normalization.  Since the listener scores we
 distribute are based on listening tests that used normalized samples, it is
 important that you also use samples that are normalized in the same way.
 If you don't already have sv56demo on your path, you can find it here:
   https://github.com/openitu/STL/tree/dev/src/sv56
 Or use `install_sv56.sh` in this directory which downloads and compiles sv56.
 Please be sure to add the compiled binary `sv56demo` to your system path.

 To check that sv56 has been properly added to your path, run:
   which sv56demo
 It should show the path to your sv56demo binary.  If it shows an empty result,
 then it is not on your path and preprocessing will fail.

 * Other requirements
   * standard command line utilities such as wget, md5sum, gawk, gcc,
     automake, grep, etc.
   * PLEASE NOTE: The size of the final dataset is 74GB, and additional storage
     is required for temporary files.  Please make sure that you have sufficient
     storage.


========================================
=== Download Blizzard Challenge data ===
========================================

The terms of use of the Blizzard Challenge data stipulate that the data may NOT
be redistributed.  Some individual teams also stipulate that their samples may
be used for research purposes only, and may not be used for commercial purposes.

By downloading the Blizzard data, you are agreeing to abide by its terms of use.
You may NOT redistribute the samples or use them for commercial purposes.  For
more information, please see:
https://www.cstr.ed.ac.uk/projects/blizzard/data.html

It takes some time to download archives for multiple Blizzard Challenge years.
Please make sure that you have a stable internet connection, and please be
patient!

Launch download and extraction of Blizzard samples by running:
  python 01_get.py
  

========================================================
=== Gather and preprocess Blizzard Challenge samples ===
========================================================

1. Run:
     python 02_gather.py

This script selects the samples used for the VoiceMOS Challenge from the
various Blizzard datasets.


2. Run: 
     python 03_preprocess.py

This script runs downsampling to 16kHz on the selected Blizzard samples,
followed by sv56 amplitude normalization.

Final processed audio data for the challenge can be found in
  DATA/wav/*.wav


===============================
=== Train / dev / test sets ===
===============================

Training, development, and test partitions can be found in
  DATA/sets

train_mos_list.txt, val_mos_list.txt, and test_mos_list.txt contain the list of
wav files and their AVERAGED MOS scores over all 8 listeners who rated them.

TRAINSET, DEVSET, and TESTSET contain the individual ratings from each rater,
along with some demographic information for the rater.
The format is as follows:

  sysID,uttID,rating,ignore,listenerinfo

The listener info is as follows:

  {}_AGERANGE_LISTENERID_GENDER_[ignore]_[ignore]_HEARINGIMPAIRMENT

DEVSET and val_mos_list.txt are the same set (the terms dev and val are used
interchangeably here).

The file DATA/mydata_system.csv contains system-averaged MOS scores.
These averages are based on the training and development set samples.


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


===============================
=== TROUBLESHOOTING AND FAQ ===
===============================

Q. Why does this process take so long and have so many steps?

A. The terms of use of the Blizzard Challenge data state that we may not
   redistribute it.  So, in order to make the data available to you in the
   exact preprocessed form that we used for the BVCC listening test, we are
   providing these scripts for you to download and preprocess the data yourself.
   Even though we cannot distribute the data directly, the data after downloading
   and preprocessing should be identical to the BVCC data that we used for
   training baselines etc.

Q. What is silence.wav?  Why is there one silent audio file?

A. The audio data used in this challenge comes from past synthesis challenges
   such as the Blizzard Challenge and the Voice Conversion Challenge.  There
   is one silent audio file because that is what one team submitted for one of
   the Blizzard Challenges.  When you try to sv56-normalize a totally silent
   audio file, the header becomes corrupted.  So, we replace the corrupted audio
   file with a silent audio file that has a non-corrupted header.  If you found
   the silent audio file in the processed data, it's supposed to be there.

Q. Why have the audio filenames been obfuscated?

A. The audio samples in these datasets come from publicly-available MOS test
   data.  Although we conducted our own MOS test for these samples, the old and
   new MOS tests have strong correlations, year by year.  It would therefore be
   possible to cheat at the VoiceMOS Challenge by finding the MOS ratings for
   the test set samples in their original datasets.  We specifically prohibited
   this in the challenge rules, and we also obfuscated the filenames to make it
   more difficult to cheat in this way.  We retain the obfuscated filenames in
   this distribution to exactly match what was distributed to participants, but
   we also include the mapping files (secret_sys_mappings.txt and
   secret_sys_mappings.txt) in case you want to use information about which
   systems from which the samples originate.
   

========================
=== Acknowledgements ===
========================

Many thanks to the organizers of the Blizzard and Voice Conversion Challenges,
and to the authors of ESPnet-TTS, who have kindly made these samples available.

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
