Installazione versione >1.0

conda create --name mmsegmentation python=3.8 -y
conda activate mmsegmentation
pip3 install torch torchvision torchaudio
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmpretrain[multimodal]>=1.0.0rc8"
cd mmsegmentation
pip install -v -e .
pip install ftfy wandb


Known issues:

https://github.com/open-mmlab/mmsegmentation/issues/3406


##############    Contest EIM 24-25    ###########################

python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/VERSE_experiments/VERSE_config.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/CODE/work_dir/VERSE --amp
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/VERSE_experiments/VERSE_config_unet.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/CODE/work_dir/VERSE --amp

############## THYROID A ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/Thyroid_experiments/Thyroid_A.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/ThyroidA --amp

############## THYROID B ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/Thyroid_experiments/Thyroid_B.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/ThyroidB --amp

##############    IMT    ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/IMT_experiments/IMT.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/IMT --amp

##############    DTP    ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/DTP_experiments/DTP_config.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/DTP --amp

##############    DERMA    ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/DERMA_experiments/DERMA_config.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/DERMA
python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/DERMA_experiments/DERMA_config.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/DERMA

##############    RETINA    ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/RETINA_experiments/RETINA_config.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/RETINA --amp


##############    FSHD    ###########################

python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/FSHD --amp
python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_KNET.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/FSHD --amp
python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_KNET_newData.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/FSHD
python tools/train.py /Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_R50.py --work-dir /media/francesco/DEV001/PROJECT-THYROID/RESULTS/mmsegmentation/FSHD

python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0 --resume
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4

python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_binary.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_binary
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_binary.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_binary
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_binary.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_binary
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_binary.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_binary
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_binary.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_binary

# Biceps_brachii
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Biceps_brachii.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Biceps_brachii
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Biceps_brachii.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Biceps_brachii
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Biceps_brachii.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Biceps_brachii
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Biceps_brachii.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Biceps_brachii
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Biceps_brachii.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Biceps_brachii

# Deltoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Deltoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Deltoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Deltoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Deltoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Deltoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Deltoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Deltoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Deltoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Deltoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Deltoideus

# Depressor_anguli_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Depressor_anguli_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Depressor_anguli_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Depressor_anguli_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Depressor_anguli_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Depressor_anguli_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Depressor_anguli_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Depressor_anguli_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Depressor_anguli_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Depressor_anguli_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Depressor_anguli_oris

# Digastricus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Digastricus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Digastricus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Digastricus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Digastricus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Digastricus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Digastricus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Digastricus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Digastricus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Digastricus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Digastricus

# Gastrocnemius_medial_head
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Gastrocnemius_medial_head.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Gastrocnemius_medial_head
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Gastrocnemius_medial_head.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Gastrocnemius_medial_head
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Gastrocnemius_medial_head.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Gastrocnemius_medial_head
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Gastrocnemius_medial_head.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Gastrocnemius_medial_head
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Gastrocnemius_medial_head.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Gastrocnemius_medial_head

# Geniohyoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Geniohyoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Geniohyoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Geniohyoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Geniohyoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Geniohyoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Geniohyoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Geniohyoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Geniohyoideus
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Geniohyoideus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Geniohyoideus

# Masseter
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Masseter.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Masseter
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Masseter.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Masseter
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Masseter.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Masseter
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Masseter.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Masseter
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Masseter.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Masseter

# Mentalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Mentalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Mentalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Mentalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Mentalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Mentalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Mentalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Mentalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Mentalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Mentalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Mentalis

# Orbicularis_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Orbicularis_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Orbicularis_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Orbicularis_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Orbicularis_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Orbicularis_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Orbicularis_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Orbicularis_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Orbicularis_oris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Orbicularis_oris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Orbicularis_oris

# Rectus_abdominis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Rectus_abdominis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Rectus_abdominis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Rectus_abdominis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Rectus_abdominis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Rectus_abdominis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Rectus_abdominis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Rectus_abdominis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Rectus_abdominis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Rectus_abdominis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Rectus_abdominis

# Rectus_femoris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Rectus_femoris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Rectus_femoris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Rectus_femoris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Rectus_femoris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Rectus_femoris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Rectus_femoris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Rectus_femoris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Rectus_femoris
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Rectus_femoris.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Rectus_femoris

# Temporalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Temporalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Temporalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Temporalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Temporalis
-python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Temporalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Temporalis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Temporalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Temporalis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Temporalis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Temporalis

# Tibialis_anterior
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Tibialis_anterior.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Tibialis_anterior
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Tibialis_anterior.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Tibialis_anterior
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Tibialis_anterior.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Tibialis_anterior
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Tibialis_anterior.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Tibialis_anterior
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Tibialis_anterior.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Tibialis_anterior

# Trapezius
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Trapezius.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Trapezius
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Trapezius.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Trapezius
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Trapezius.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Trapezius
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Trapezius.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Trapezius
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Trapezius.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Trapezius

# Vastus_lateralis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Vastus_lateralis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Vastus_lateralis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Vastus_lateralis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Vastus_lateralis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Vastus_lateralis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Vastus_lateralis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Vastus_lateralis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Vastus_lateralis
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Vastus_lateralis.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Vastus_lateralis

# Zygomaticus
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f0_Zygomaticus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f0_Zygomaticus
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f1_Zygomaticus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f1_Zygomaticus
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f2_Zygomaticus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f2_Zygomaticus
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f3_Zygomaticus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f3_Zygomaticus
python tools/train.py /home/francesco/Desktop/POLI/RADBOUD/CODE/mmsegmentation/configs/FSHD_experiments/FSHD_config_SWIN_f4_Zygomaticus.py --work-dir /home/francesco/Desktop/POLI/RADBOUD/RESULTS/FSHD/FSHD_KNET_SWIN_f4_Zygomaticus


# define class and palette for better visualization
classes = [
           x'Biceps_brachii', # 001 - 1 for label
           x'Deltoideus', # 002
           x'Depressor_anguli_oris', # 003
           x'Digastricus', # 004
           x'Gastrocnemius_medial_head', # 008
           x'Geniohyoideus', # 009
           x'Masseter', # 011
           x'Mentalis', # 012
           x'Orbicularis_oris', # 013
           012'Rectus_abdominis', # 015
           'Rectus_femoris', # 016
           0'Temporalis', # 017
           'Tibialis_anterior', # 018
           'Trapezius', # 019
           'Vastus_lateralis', # 020
           'Zygomaticus']  # 021