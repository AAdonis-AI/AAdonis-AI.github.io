---
title: "espeak-ng"
date: 2025-10-01
draft: false
tags: ["espeak", "phonemes", "alignment"]
summary: "Installing espeak-ng with and without sudo — including a no-admin local build for HPC clusters."
---

`espeak-ng` is a text-to-phoneme engine required by several audio tools (VoiceCraft, phonemizer, etc.). Installing it is straightforward with sudo, but on shared HPC clusters you often need a local build.

## Case 1: Standard Install (with sudo)

```bash
sudo apt-get install espeak-ng
```

## Case 2: No-sudo Install (HPC / restricted Linux)

**Step 1 — Clone and build**

```bash
cd ~
git clone https://github.com/espeak-ng/espeak-ng.git
cd espeak-ng

./autogen.sh
./configure --prefix=$HOME/.local
make -j$(nproc)
make install
```

**Step 2 — Add to PATH and library path**

```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Step 3 — Verify**

```bash
which espeak-ng
# Expected: /home/<username>/.local/bin/espeak-ng

espeak-ng "Hello world"
```

**Step 4 — Clean removal (optional)**

```bash
rm -rf ~/espeak-ng ~/.local/bin/espeak-ng ~/.local/lib/libespeak-ng*
sed -i '/.local\/bin/d' ~/.bashrc
sed -i '/.local\/lib/d' ~/.bashrc
```
