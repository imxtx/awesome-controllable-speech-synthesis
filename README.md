# Awesome Controllabe Speech Synthesis

This is an evolving repo for the survey: [Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey](https://arxiv.org/abs/2412.06602). If you find our survey useful for your research, please 📚cite📚 the following paper:

```latex
@article{xie2024towards,
  title={Towards Controllable Speech Synthesis in the Era of Large Language Models: A Survey},
  author={Xie, Tianxin and Rong, Yan and Zhang, Pengfei and Liu, Li},
  journal={arXiv preprint arXiv:2412.06602},
  year={2024}
}
```

<p align="center">
    <img src="./images/sec1_summary.jpg" width="1000"/>
</p>
<p align="center">
    <img src="./images/sec2_pipeline.jpg" width="1000"/>
</p>
<p align="center">
    <img src="./images/sec5_control_strategies.jpg" width="1000"/>
</p>
<p align="center">
<img src="./images/sec6_challenges.jpg" width="500"/>
</p>

## 🚀 Non-autoregressive Controllable TTS

Below are representative non-autoregressive controllable TTS methods. Each entry follows this format: **method name, zero-shot capability, controllability, acoustic model, vocoder, acoustic feature, release date, and code/demo**.

> *NOTE*: MelS and LinS represent Mel Spectrogram and Linear Spectrogram, respectively. Among today’s TTS systems, MelS, latent features (from VAEs, diffusion models, and other flow-based methods), and various types of discrete tokens are the most commonly used acoustic representations.

- [FastSpeech](https://proceedings.neurips.cc/paper_files/paper/2019/hash/f63f65b503e22cb970527f23c9ad7db1-Abstract.html), Zero-shot (✗), Controllability (Speed, Prosody), Transformer, [WaveGlow](https://github.com/NVIDIA/waveglow), MelS, 2019.05, [Code (unofficial)](https://github.com/xcmyz/FastSpeech)
- [FastSpeech 2](https://arxiv.org/abs/2006.04558), Zero-shot (✗), Controllability (Pitch, Energy, Speed, Prosody), Transformer, [Parallel WaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), MelS, 2020.06, [Code (unofficial)](https://github.com/ming024/FastSpeech2)
- [FastPitch](https://ieeexplore.ieee.org/abstract/document/9413889), Zero-shot (✗), Controllability (Pitch, Prosody), Transformer, [WaveGlow](https://github.com/NVIDIA/waveglow), MelS, 2020.06, [Code](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/SpeechSynthesis/FastPitch)
- [Parallel Tacotron](https://ieeexplore.ieee.org/abstract/document/9414718), Zero-shot (✗), Controllability (Prosody), Transformer + CNN, [WaveRNN](https://github.com/fatchord/WaveRNN), MelS, 2020.10, [Demo](https://google.github.io/tacotron/publications/parallel_tacotron/)
- [StyleTagging-TTS](https://arxiv.org/abs/2104.00436), Zero-shot (✓), Controllability (Timbre, Emotion), Transformer + CNN, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2021.04, [Demo](https://gannnn123.github.io/styletaggingtts-demo/)
- [SC-GlowTTS](https://arxiv.org/abs/2104.05557), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2021.06, [Demo](https://edresson.github.io/SC-GlowTTS/), [Code](https://github.com/Edresson/SC-GlowTTS)
- [Meta-StyleSpeech](https://proceedings.mlr.press/v139/min21b.html), Zero-shot (✓), Controllability (Timbre), Transformer, [MelGAN](https://arxiv.org/abs/1910.06711), MelS, 2021.06, [Code](https://github.com/KevinMIN95/StyleSpeech)
- [DelightfulTTS](https://arxiv.org/abs/2110.12612), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), Transformer + CNN, [HiFiNet](https://github.com/yl4579/HiFTNet), MelS, 2021.11, [Demo](https://cognitivespeech.github.io/delightfultts)
- [YourTTS](https://proceedings.mlr.press/v162/casanova22a.html), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), LinS, 2021.12, [Demo & Checkpoint](https://github.com/Edresson/YourTTS)
- [StyleTTS](https://arxiv.org/abs/2205.15439), Zero-shot (✓), Controllability (Timbre), CNN + RNN, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2022.05, [Code](https://github.com/yl4579/StyleTTS)
- [GenerSpeech](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4730d10b22261faa9a95ebf7497bc556-Abstract-Conference.html), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2022.05, [Demo](https://generspeech.github.io/)
- [Cauliflow](https://arxiv.org/abs/2206.14165), Zero-shot (✗), Controllability (Speed, Prosody), BERT + Flow, [UP WaveNet](https://arxiv.org/abs/2102.01106), MelS, 2022.06
- [CLONE](https://arxiv.org/abs/2207.06088), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), Transformer + CNN, [WaveNet](https://arxiv.org/abs/1609.03499), MelS + LinS, 2022.07, [Demo](https://xcmyz.github.io/CLONE/)
- [PromptTTS](https://ieeexplore.ieee.org/abstract/document/10096285), Zero-shot (✗), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion, Description), Bert + Transformer, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2022.11, [Demo](https://speechresearch.github.io/prompttts/)
- [Grad-StyleSpeech](https://ieeexplore.ieee.org/abstract/document/10095515), Zero-shot (✓), Controllability (Timbre), Score-based Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2022.11, [Demo](https://nardien.github.io/grad-stylespeech-demo/)
- [NaturalSpeech 2](https://arxiv.org/abs/2304.09116), Zero-shot (✓), Controllability (Timbre), Diffusion, [RVQ-based Codec](https://arxiv.org/abs/2304.09116), Token, 2023.04, [Demo](https://speechresearch.github.io/naturalspeech2/), [Code (unofficial)](https://github.com/lucidrains/naturalspeech2-pytorch)
- [PromptStyle](https://arxiv.org/abs/2305.19522), Zero-shot (✓), Controllability (Pitch, Prosody, Timbre, Emotion, Description), [VITS](https://arxiv.org/abs/2106.06103) + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.05, [Demo](https://promptstyle.github.io/PromptStyle)
- [StyleTTS 2](https://proceedings.neurips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion), Flow-based Diffusion + GAN, [HiFi-GAN](https://github.com/jik876/hifi-gan) / [iSTFTNet](https://github.com/rishikksh20/iSTFTNet-pytorch), MelS, 2023.06, [Demo](https://styletts2.github.io/), [Code](https://github.com/yl4579/StyleTTS2)
- [VoiceBox](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2d8911db9ecedf866015091b28946e15-Abstract-Conference.html), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.06, [Demo](https://voicebox.metademolab.com/), [Code (unofficial)](https://github.com/lucidrains/voicebox-pytorch)
- [MegaTTS 2](https://openreview.net/forum?id=mvMI3N4AvD), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion), Decoder-only Transformer + GAN, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.07, [Demo](https://boostprompt.github.io/boostprompt/), [Code (unofficial)](https://github.com/LSimon95/megatts2)
- [PromptTTS 2](https://arxiv.org/abs/2309.02285), Zero-shot (✗), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Description), Diffusion, [RVQ-based Codec]((https://arxiv.org/abs/2309.02285)), Latent Feature, 2023.09, [Demo](https://speechresearch.github.io/prompttts2/)
- [VoiceLDM](https://ieeexplore.ieee.org/abstract/document/10448268), Zero-shot (✗), Controllability (Pitch, Prosody, Timbre, Emotion, Environment, Description), Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.09, [Demo](https://voiceldm.github.io/), [Code](https://github.com/glory20h/VoiceLDM)
- [DuIAN-E](https://arxiv.org/abs/2309.12792), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), CNN + RNN, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.09, [Demo](https://sounddemos.github.io/durian-e/)
- [PromptTTS++](https://ieeexplore.ieee.org/abstract/document/10448173), Zero-shot (✗), Controllability (Pitch, Speed, Prosody, Timbre, Emotion, Description), Transformer + Diffusion, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2023.09, [Demo](https://reppy4620.github.io/demo.promptttspp/), [Code](https://github.com/line/promptttspp)
- [SpeechFlow](https://arxiv.org/abs/2310.16338), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.10, [Demo](https://voicebox.metademolab.com/speechflow.html)
- [P-Flow](https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.10, [Demo](https://research.nvidia.com/labs/adlr/projects/pflow/), [Code (unofficial)](https://github.com/p0p4k/pflowtts_pytorch)
- [E3 TTS](https://ieeexplore.ieee.org/abstract/document/10389766), Zero-shot (✓), Controllability (Timbre), Diffusion, Not required, Waveform, 2023.11, [Demo](https://e3tts.github.io/)
- [HierSpeech++](https://arxiv.org/abs/2311.12454), Zero-shot (✓), Controllability (Timbre), Transformer + VAE + Flow, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2023.11, [Demo](https://sh-lee-prml.github.io/HierSpeechpp-demo/), [Code](https://github.com/sh-lee-prml/HierSpeechpp)
- [Audiobox](https://arxiv.org/abs/2312.15821), Zero-shot (✓), Controllability (Pitch, Speed, Prosody, Timbre, Environment, Description), Transformer + Flow, [EnCodec](https://github.com/facebookresearch/encodec), MelS, 2023.12, [Demo](https://audiobox.metademolab.com/)
- [FlashSpeech](https://dl.acm.org/doi/abs/10.1145/3664647.3681044), Zero-shot (✓), Controllability (Timbre), Latent Consistency Model, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2024.04, [Demo](https://flashspeech.github.io/), [Code](https://github.com/zhenye234/FlashSpeech)
- [NaturalSpeech 3](https://arxiv.org/abs/2403.03100), Zero-shot (✓), Controllability (Speed, Prosody, Timbre), Transformer + Diffusion, [FACodec](https://github.com/lifeiteng/naturalspeech3_facodec), Token, 2024.04, [Demo](https://speechresearch.github.io/naturalspeech3/)
- [InstructTTS](https://ieeexplore.ieee.org/abstract/document/10534832), Zero-shot (✗), Controllability (Pitch, Speed, Prosody, Timbre, Emotion, Description), Transformer + Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), Token, 2024.05, [Demo](https://dongchaoyang.top/InstructTTS/)
- [ControlSpeech](https://arxiv.org/abs/2406.01205), Zero-shot (✓), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion, Description), Transformer + Diffusion, [FACodec](https://github.com/lifeiteng/naturalspeech3_facodec), Token, 2024.06, [Demo](https://controlspeech.github.io/), [Code](https://github.com/jishengpeng/ControlSpeech)
- [AST-LDM](https://arxiv.org/abs/2406.12688), Zero-shot (✗), Controllability (Timbre, Environment, Description), Diffusion + VAE, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2024.06, [Demo](https://ast-ldm.github.io/demo/)
- [SimpleSpeech](https://arxiv.org/abs/2406.02328), Zero-shot (✓), Controllability (Timbre), Transformer + Diffusion, [SQ Codec](https://arxiv.org/abs/2406.02328), Token, 2024.06, [Demo](https://simplespeech.github.io/simplespeechDemo/), [Code](https://github.com/yangdongchao/SimpleSpeech)
- [DiTTo-TTS](https://arxiv.org/abs/2406.11427), Zero-shot (✓), Controllability (Speed, Timbre), DiT + VAE, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.06, [Demo](https://ditto-tts.github.io/)
- [E2 TTS](https://arxiv.org/abs/2406.18009), Zero-shot (✓), Controllability (Timbre), Transformer + Flow, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.06, [Demo](https://www.microsoft.com/en-us/research/project/e2-tts/), [Code (unofficial)](https://github.com/lucidrains/e2-tts-pytorch)
- [MobileSpeech](https://arxiv.org/abs/2402.09378), Zero-shot (✓), Controllability (Timbre), Transformer, [Vocos](https://github.com/gemelo-ai/vocos), Token, 2024.06, [Demo](https://mobilespeech.github.io/)
- [DEX-TTS](https://arxiv.org/abs/2406.19135), Zero-shot (✓), Controllability (Timbre), Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2024.06, [Code](https://github.com/winddori2002/DEX-TTS)
- [ArtSpeech](https://dl.acm.org/doi/abs/10.1145/3664647.3681097), Zero-shot (✓), Controllability (Timbre), RNN + CNN, [HiFI-GAN](https://github.com/jik876/hifi-gan), MelS, 2024.07, [Demo](https://zhongxu-wang.github.io/artspeeech.demopage/), [Code](https://github.com/Zhongxu-Wang/ArtSpeech)
- [CCSP](https://dl.acm.org/doi/abs/10.1145/3664647.3681348), Zero-shot (✓), Controllability (Timbre), Diffusion, [RVQ-based Codec]((https://dl.acm.org/doi/abs/10.1145/3664647.3681348)), Token, 2024.07, [Demo](https://ccsp2024.github.io/demo/)
- [SimpleSpeech 2](https://arxiv.org/abs/2408.13893), Zero-shot (✓), Controllability (Speed, Timbre), Flow-based DiT, [SQ Codec](https://arxiv.org/abs/2406.02328), Token, 2024.08, [Demo](https://dongchaoyang.top/SimpleSpeech2_demo/), [Code](https://github.com/yangdongchao/SimpleSpeech)
- [E1 TTS](https://arxiv.org/abs/2409.09351), Zero-shot (✓), Controllability (Timbre), DiT + Flow, [BigVGAN](https://github.com/NVIDIA/BigVGAN), Token + MelS, 2024.09, [Demo](https://e1tts.github.io/)
- [StyleTTS-ZS](https://arxiv.org/abs/2409.10058), Zero-shot (✓), Controllability (Timbre), Flow-based Diffusion + GAN, [Mel-based Decoder](https://arxiv.org/abs/2409.10058), MelS, 2024.09, [Demo](https://styletts-zs.github.io/)
- [NansyTTS](https://arxiv.org/abs/2409.17452), Zero-shot (✓), Controllability (Pitch, Speed, Prosody, Timbre, Description), Transformer, [NANSY++](https://arxiv.org/abs/2409.17452), MelS, 2024.09, [Demo](https://r9y9.github.io/projects/nansyttspp/)
- [NanoVoice](https://arxiv.org/abs/2409.15760), Zero-shot (✓), Controllability (Timbre), Diffusion, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.09
- [MS$^{2}$KU-VTTS](https://arxiv.org/abs/2410.14101), Zero-shot (✗), Controllability (Environment, Description), Diffusion, [BigvGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.10
- [EmoSphere++](https://arxiv.org/abs/2411.02625), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion), Transformer + Flow, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.11, [Demo](https://choddeok.github.io/EmoSphere-Demo/), [Code](https://github.com/Choddeok/EmoSpherepp)
- [EmoDubber](https://arxiv.org/abs/2412.08988), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion), Transformer + Flow, [Flow-based Vocoder](https://arxiv.org/abs/2412.08988), MelS, 2024.12, [Demo](https://galaxycong.github.io/EmoDub/)
- [HED](https://arxiv.org/abs/2412.12498), Zero-shot (✓), Controllability (Emotion), Flow-based Diffusion, [Vocos](https://github.com/gemelo-ai/vocos), MelS, 2024.12, [Demo](https://shinshoji01.github.io/HED-Demo/)
- [DiffStyleTTS](https://arxiv.org/abs/2412.03388), Zero-shot (✗), Controllability (Pitch, Energy, Speed, Prosody, Timbre), Transformer + Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2025.01, [Demo](https://xuan3986.github.io/DiffStyleTTS/)
- [DrawSpeech](https://arxiv.org/abs/2501.04256), Zero-shot (✗), Controllability (Energy, Prosody), Diffusion, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2025.01, [Demo](https://happycolor.github.io/DrawSpeech/), [Code](https://github.com/HappyColor/DrawSpeech_PyTorch)
- [ProEmo](https://arxiv.org/abs/2501.06276), Zero-shot (✗), Controllability (Pitch, Energy, Emotion, Description), Transformer, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2025.01, [Code](https://github.com/ZhangShaozuo/FastSpeech2PromptGuidance)

## 🎞️ Autoregressive Controllable TTS

Below are representative non-autoregressive controllable TTS methods. Each entry follows this format: **method name, zero-shot capability, controllability, acoustic model, vocoder, acoustic feature, release date, and code/demo**.

> *NOTE*: MelS and LinS represent Mel Spectrogram and Linear Spectrogram, respectively. Among today’s TTS systems, MelS, latent features (from VAEs, diffusion models, and other flow-based methods), and various types of discrete tokens are the most commonly used acoustic representations.

- [Prosody-Tacotron](https://proceedings.mlr.press/v80/skerry-ryan18a.html), Zero-shot (✗), Controllability (Pitch, Prosody), RNN, [WaveNet](https://arxiv.org/abs/1609.03499), MelS, 2018.03, [Demo](https://google.github.io/tacotron/publications/end_to_end_prosody_transfer/)
- [GST-Tacotron](https://ieeexplore.ieee.org/abstract/document/8639682), Zero-shot (✗), Controllability (Pitch, Prosody), CNN + RNN, [Griffin-Lim](https://pytorch.org/audio/main/generated/torchaudio.transforms.GriffinLim.html), LinS, 2018.03, [Demo](https://google.github.io/tacotron/publications/global_style_tokens/), [Code (unofficial)](https://github.com/KinglittleQ/GST-Tacotron)
- [GMVAE-Tacotron](https://arxiv.org/abs/1810.07217), Zero-shot (✗), Controllability (Pitch, Speed, Prosody, Description), VAE, [WaveRNN](https://github.com/fatchord/WaveRNN), MelS, 2018.12, [Demo](https://google.github.io/tacotron/publications/gmvae_controllable_tts/), [Code (unofficial)](https://github.com/rishikksh20/gmvae_tacotron)
- [VAE-Tacotron](https://ieeexplore.ieee.org/abstract/document/8683623), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), VAE, [WaveNet](https://arxiv.org/abs/1609.03499), MelS, 2019.02, [Code (unoffcial 1)](https://github.com/yanggeng1995/vae_tacotron), [Code (unoffcial 2)](https://github.com/xcmyz/VAE-Tacotron)
- [DurIAN](https://arxiv.org/abs/1909.01700), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), CNN + RNN, [MB-WaveRNN]((https://arxiv.org/abs/1909.01700)), MelS, 2019.09, [Demo](https://tencent-ailab.github.io/durian/), [Code (unofficial)](https://github.com/ivanvovk/durian-pytorch)
- [Flowtron](https://arxiv.org/abs/2005.05957), Zero-shot (✗), Controllability (Pitch, Speed, Prosody), CNN + RNN, [WaveGlow](https://github.com/NVIDIA/waveglow), MelS, 2020.07, [Demo](https://nv-adlr.github.io/Flowtron), [Code](https://github.com/NVIDIA/flowtron)
- [MsEmoTTS](https://ieeexplore.ieee.org/abstract/document/9693186), Zero-shot (✓), Controllability (Pitch, Prosody, Emotion), CNN + RNN, [WaveRNN](https://arxiv.org/abs/1711.10433), MelS, 2022.01, [Demo](https://leiyi420.github.io/MsEmoTTS/)
- [VALL-E](https://arxiv.org/abs/2301.02111), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2023.01, [Demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e/), [Code (unofficial 1)](https://github.com/enhuiz/vall-e), [Code (unofficial 2)](https://github.com/lifeiteng/vall-e)
- [SpearTTS](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00618/118854), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [SoundStream](https://github.com/wesbz/SoundStream), Token, 2023.02, [Demo](https://google-research.github.io/seanet/speartts/examples/), [Code (unofficial)](https://github.com/lucidrains/spear-tts-pytorch)
- [VALL-E X](https://arxiv.org/abs/2303.03926), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2023.03, [Demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-x/), [Code (unofficial)](https://github.com/Plachtaa/VALL-E-X)
- [Make-a-voice](https://arxiv.org/abs/2305.19269), Zero-shot (✓), Controllability (Timbre), Encoder-decoder Transformer, [Unit-based Vocoder](https://arxiv.org/abs/2305.19269), Token, 2023.05, [Demo](https://make-a-voice.github.io/)
- [TorToise](https://arxiv.org/abs/2305.07243), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer + Diffusion, [UnivNet](https://arxiv.org/abs/2106.07889), MelS, 2023.05, [Code](https://github.com/neonbjb/tortoise-tts)
- [MegaTTS](https://arxiv.org/abs/2306.03509), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer + GAN, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2023.06, [Demo](https://mega-tts.github.io/demo-page/)
- [SC VALL-E](https://arxiv.org/abs/2307.10550), Zero-shot (✓), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2023.07, [Demo](https://0913ktg.github.io/), [Code](https://github.com/0913ktg/SC_VALL-E)
- [Salle](https://ieeexplore.ieee.org/abstract/document/10445879), Zero-shot (✗), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion, Description), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2023.08, [Demo](https://sall-e.github.io/)
- [UniAudio](https://arxiv.org/abs/2310.00704), Zero-shot (✓), Controllability (Pitch, Speed, Prosody, Timbre, Description), Decoder-only Transformer, [UniAudio Codec](https://github.com/yangdongchao/UniAudio/tree/main/codec), Token, 2023.10, [Demo](https://dongchaoyang.top/UniAudio_demo/), [Code](https://github.com/yangdongchao/UniAudio)
- [ELLA-V](https://arxiv.org/abs/2401.07333), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2024.01, [Demo](https://ereboas.github.io/ELLAV/)
- [BaseTTS](https://arxiv.org/abs/2402.08093), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [Speechcode Decoder]((https://arxiv.org/abs/2402.08093)), Token, 2024.02, [Demo](https://www.amazon.science/base-tts-samples/)
- [Seed-TTS](https://arxiv.org/abs/2406.02430), Zero-shot (✓), Controllability (Timbre, Emotion), Decoder-only Transformer + DiT, *Unknown Vocoder*, Latent Feature, 2024.06, [Demo](https://bytedancespeech.github.io/seedtts_tech_report/)
- [VoiceCraft](https://arxiv.org/abs/2403.16973), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [HiFi-GAN](https://github.com/jik876/hifi-gan), Token, 2024.06, [Code](https://github.com/jasonppy/VoiceCraft)
- [XTTS](https://arxiv.org/abs/2406.04904), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer + GAN, [HiFi-GAN-based Vococder](https://arxiv.org/abs/2406.04904), Token + MelS, 2024.06, [Demo](https://edresson.github.io/XTTS/), [Code](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md)
- [CosyVoice](https://arxiv.org/abs/2407.05407), Zero-shot (✓), Controllability (Pitch, Speed, Prosody, Timbre, Emotion, Description), Decoder-only Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), Token, 2024.07, [Demo](https://fun-audio-llm.github.io/), [Code](https://github.com/FunAudioLLM/CosyVoice)
- [MELLE](https://arxiv.org/abs/2407.08551), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [HiFi-GAN](https://github.com/jik876/hifi-gan), MelS, 2024.07. [Demo](https://www.microsoft.com/en-us/research/project/vall-e-x/melle/)
- [VoxInstruct](https://dl.acm.org/doi/abs/10.1145/3664647.3681680), Zero-shot (✓), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion, Description), Decoder-only Transformer, [Vocos](https://github.com/gemelo-ai/vocos), Token, 2024.08, [Demo](https://voxinstruct.github.io/VoxInstruct/), [Code](https://github.com/thuhcsi/VoxInstruct)
- [Emo-DPO](https://arxiv.org/abs/2409.10157), Zero-shot (✗), Controllability (Emotion), LLM, [HiFi-GAN](https://github.com/jik876/hifi-gan), Token + MelS, 2024.09, [Demo](https://xiaoxue1117.github.io/Emo-tts-dpo/)
- [FireRedTTS](https://arxiv.org/abs/2409.03283), Zero-shot (✓), Controllability (Prosody, Timbre), Decoder-only Transformer + Flow, [BigVGAN](https://github.com/NVIDIA/BigVGAN)-v2, Token + MelS, 2024.09, [Demo](https://fireredteam.github.io/demos/firered_tts/), [Code](https://github.com/FireRedTeam/FireRedTTS)
- [CoFi-Speech](https://arxiv.org/abs/2409.11630), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [BigVGAN](https://github.com/NVIDIA/BigVGAN), Token + MelS, 2024.09, [Demo](https://hhguo.github.io/DemoCoFiSpeech/)
- [Takin](https://arxiv.org/abs/2409.12139), Zero-shot (✓), Controllability (Pitch, Speed, Prosody, Timbre, Emotion, Description), Decoder-only Transformer + Flow, [HiFi-GAN](https://github.com/jik876/hifi-gan), Token + MelS, 2024.09, [Demo](https://everest-ai.github.io/takinaudiollm/)
- [HALL-E](https://arxiv.org/abs/2410.04380), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [EnCodec](https://github.com/facebookresearch/encodec), Token, 2024.10
- [KALL-E](https://arxiv.org/abs/2412.16846), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion), Decoder-only Transformer, [WaveVAE]((https://arxiv.org/abs/2412.16846)), Latent Feature, 2024.12, [Demo](https://zxf-icpc.github.io/kalle/)
- [FleSpeech](https://arxiv.org/abs/2501.04644), Zero-shot (✓), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion, Description), Flow-based DiT, WaveGAN, Latent Feature, 2025.01, [Demo](https://kkksuper.github.io/FleSpeech/)
- [Step-Audio](https://arxiv.org/abs/2502.11946), Zero-shot (✓), Controllability (Prosody, Timbre, Emotion, Description), Decoder-only Transformer, [Flow-based Vocoder], Token, 2025.02, [Code](https://github.com/stepfun-ai/Step-Audio)
- [Vevo](https://openreview.net/forum?id=anQDiQZhDP), Zero-shot (✓), Controllability (Pitch, Energy, Speed, Prosody, Timbre, Emotion), Decoder-only Transformer, [BigVGAN](https://github.com/NVIDIA/BigVGAN), Token + MelS, 2025.02, [Demo](https://versavoice.github.io/), [Code](https://github.com/open-mmlab/Amphion/tree/main/models/vc/vevo)
- [CLaM-TTS](https://arxiv.org/abs/2404.02781), Zero-shot (✓), Controllability (Timbre), Encoder-decoder Transformer, [BigVGAN](https://github.com/NVIDIA/BigVGAN), Token + MelS, 2024.04, [Demo](https://clam-tts.github.io/)
- [RALL-E](https://arxiv.org/abs/2404.03204), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [SoundStream](https://github.com/wesbz/SoundStream), Token, 2024.05, [Demo](https://ralle-demo.github.io/RALL-E/)
- [ARDiT](https://arxiv.org/abs/2406.05551), Zero-shot (✓), Controllability (Speed, Timbre), Decoder-only DiT, [BigVGAN](https://github.com/NVIDIA/BigVGAN), MelS, 2024.06, [Demo](https://zjlww.github.io/ardit-web/)
- [VALL-E R](https://arxiv.org/abs/2406.07855), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [Vocos](https://github.com/gemelo-ai/vocos), Token, 2024.06, [Demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-r/)
- [VALL-E 2](https://arxiv.org/abs/2406.05370), Zero-shot (✓), Controllability (Timbre), Decoder-only Transformer, [Vocos](https://github.com/gemelo-ai/vocos), Token, 2024.06, [Demo](https://www.microsoft.com/en-us/research/project/vall-e-x/vall-e-2/), [Code (unofficial 1)](https://github.com/open-mmlab/Amphion/tree/main/egs/tts/VALLE_V2), [Code (unofficial 2)](https://github.com/ex3ndr/supervoice-vall-e-2)

## 💾 Datsets

A summary of open-source datasets for controllable TTS:

|Dataset|Hours|#Speakers|Labels||||||||||||Lang|Release<br>Time|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
||||Pit.|Ene.|Spe.|Age|Gen.|Emo.|Emp.|Acc.|Top.|Des.|Env.|Dia.|||
|[Taskmaster-1](https://arxiv.org/abs/1909.05358)|/|/||||||||||||✓|en|2019.09|
|[Libri-light](https://ieeexplore.ieee.org/abstract/document/9052942)|60,000|9,722|||||||||✓||||en|2019.12|
|[AISHELL-3](https://arxiv.org/abs/2010.11567)|85|218||||✓|✓|||✓|||||zh|2020.10|
|[ESD](https://www.sciencedirect.com/science/article/pii/S0167639321001308)|29|10||||||✓|||||||en,zh|2021.05|
|[GigaSpeech](https://arxiv.org/abs/2106.06909)|10,000|/|||||||||✓||||en|2021.06|
|[WenetSpeech](https://ieeexplore.ieee.org/abstract/document/9746682)|10,000|/|||||||||✓||||zh|2021.07|
|[PromptSpeech](https://ieeexplore.ieee.org/abstract/document/10096285)|/|/|✓|✓|✓|||✓||||✓|||en|2022.11|
|[DailyTalk](https://ieeexplore.ieee.org/abstract/document/10095751)|20|2||||||✓|||✓|||✓|en|2023.05|
|[TextrolSpeech](https://ieeexplore.ieee.org/abstract/document/10445879)|330|1,324|✓|✓|✓||✓|✓||||✓|||en|2023.08|
|[VoiceLDM](https://ieeexplore.ieee.org/abstract/document/10448268)|/|/|✓||||✓|✓||||✓|✓||en|2023.09|
|[VccmDataset](https://arxiv.org/abs/2406.01205)|330|1,324|✓|✓|✓||✓|✓||||✓|||en|2024.06|
|[MSceneSpeech](https://arxiv.org/abs/2407.14006)|13|13|||||||||✓||||zh|2024.07|
|[SpeechCraft](https://dl.acm.org/doi/abs/10.1145/3664647.3681674)|2,391|3,200|✓|✓|✓|✓|✓|✓|✓||✓|✓|||en,zh|2024.08|

*Abbreviations*: Pit(ch), Ene(rgy)=volume, Spe(ed), Gen(der), Emo(tion), Emp(hasis), Acc(ent), Dia(logue), Env(ironment), Des(cription).

## 📏 Evaluation Metrics

| Metric | Type | Eval Target | GT Required |
|:---:|:---:|:---:|:---:|
| Mel-Cepstral Distortion (MCD) $\downarrow$ | Objective | Acoustic similarity | ✓ |
| Frequency Domain Score Difference (FDSD) $\downarrow$ | Objective | Acoustic similarity | ✓ |
| Word Error Rate (WER) $\downarrow$ | Objective | Intelligibility | ✓ |
| Cosine Similarity $\downarrow$ | Objective | Speaker similarity | ✓ |
| Perceptual Evaluation of Speech Quality (PESQ) $\uparrow$ | Objective | Perceptual quality | ✓ |
| Signal-to-Noise Ratio (SNR) $\uparrow$ | Objective | Perceptual quality | ✓ |
| Mean Opinion Score (MOS) $\uparrow$ | Subjective | Preference | |
| Comparison Mean Opinion Score (CMOS) $\uparrow$ | Subjective | Preference | |
| AB Test | Subjective | Preference | |
| ABX Test | Subjective | Perceptual similarity | ✓ |

GT: Ground truth, $\downarrow$: Lower is better, $\uparrow$: Higher is better.
