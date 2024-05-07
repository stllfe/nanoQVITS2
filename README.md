# nanoQVITS2

clean minimal VITS2 implementation focused on multilingual synthesis and fine-grained speech control

## Notes

### MFA Alignment

```shell
conda create -n aligner -c conda-forge montreal-forced-aligner
```

```shell
conda activate aligner
mfa model download acoustic english_mfa
mfa model download dictionary english_mfa
```

```shell
mfa align -j 8 \
--use_mp \
--clean \
--single_speaker \
--no_textgrid_cleanup \
data/LJSpeech-1.1/wavs/ \
english_mfa \
english_mfa \
data/LJSpeech-1.1/alignment
```

```shell
mkdir vits2/monotonic_align
python setup.py build_ext --inplace
```
