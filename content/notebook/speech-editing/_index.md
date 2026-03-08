---
title: "Speech Editing"
description: "Modifying spoken audio at the word level — insertions, deletions, and substitutions while preserving speaker identity."
---

Speech editing aims to modify specific words or phrases in an existing recording without re-recording the full utterance. The key challenge is making the edited region perceptually seamless: matching the speaker's prosody, timbre, and rhythm, and transitioning naturally at the edit boundaries.

Modern approaches typically use forced alignment (to identify exact word boundaries), a codec to tokenize the audio, and a language model to infill the masked region autoregressively conditioned on the surrounding context and the target transcript.
