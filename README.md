# MeiGen's MultiTalk: Let them talk - Audio-driven multi-person conversational video generation

[![Replicate](https://replicate.com/zsxkib/multitalk/badge)](https://replicate.com/zsxkib/multitalk)

This repository contains a Cog implementation of **MultiTalk**, MeiGen's audio-driven multi-person conversational video generation system. This isn't just another talking head generator—MultiTalk creates realistic multi-person conversations, complete with synchronized lip movements, natural interactions, and even supports singing and cartoon characters. It's like having a virtual film studio that can bring any conversation to life.

MultiTalk takes multi-stream audio inputs, a reference image, and a text prompt, then generates videos where people actually interact with each other following the conversation flow, with precise lip synchronization that puts traditional dubbing to shame.

What makes MultiTalk special:
- 🎭 Multi-person conversations: Generate realistic conversations between multiple people, not just single talking heads
- 🎤 Perfect lip sync: Audio-driven generation with accurate lip synchronization 
- 👥 Interactive control: Direct virtual humans through natural language prompts
- 🎨 Versatile characters: Works with real people, cartoon characters, and even singing performances
- 📺 High quality output: 480p and 720p generation at arbitrary aspect ratios
- ⏱️ Long-form content: Generate videos up to 15 seconds with consistent quality

Model links and information:
*   Original Project: [MeiGen-AI/MultiTalk](https://github.com/MeiGen-AI/MultiTalk)
*   Research Paper: [Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation](https://arxiv.org/abs/2505.22647)
*   Project Website: [meigen-ai.github.io/multi-talk](https://meigen-ai.github.io/multi-talk/)
*   Model Weights: [MeiGen-AI/MeiGen-MultiTalk](https://huggingface.co/MeiGen-AI/MeiGen-MultiTalk)
*   Base Model: [Wan-AI/Wan2.1-I2V-14B-480P](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)
*   This Cog packaging by: [zsxkib on GitHub](https://github.com/zsxkib) / [@zsakib_ on Twitter](https://twitter.com/zsakib_)

## Prerequisites

*   Docker: You'll need Docker to build and run the Cog container. [Install Docker](https://docs.docker.com/get-docker/).
*   Cog: Cog is required to build and run this model locally. [Install Cog](https://github.com/replicate/cog#install).
*   NVIDIA GPU: You'll need a NVIDIA GPU with at least 24GB of memory (A100, H100, or RTX 4090+ recommended) for the best performance.

## Run locally: Conversations come to life

Running MultiTalk with Cog is straightforward. The system automatically handles model downloads, audio processing, and video generation—just provide your audio files and reference image, and watch as realistic conversations unfold before your eyes.

1.  Clone this repository:
    ```bash
    git clone https://github.com/zsxkib/cog-MultiTalk.git
    cd cog-MultiTalk
    ```

2.  Run the model:
    The first time you run any command, Cog will download the model weights (~30GB total), but after that initial setup, generation is fast.

    Single-person talking video:
    ```bash
    # Generate a single person speaking
    cog predict \
      -i image=@person.jpg \
      -i first_audio=@speech.wav \
      -i prompt="A professional speaker giving a presentation"
    
    # Create a singing performance
    cog predict \
      -i image=@singer.jpg \
      -i first_audio=@song.wav \
      -i prompt="A talented singer performing an emotional ballad" \
      -i num_frames=161
    
    # Cartoon character speaking
    cog predict \
      -i image=@cartoon.jpg \
      -i first_audio=@dialogue.wav \
      -i prompt="An animated character telling an exciting story"
    ```

    Multi-person conversations (This is where MultiTalk truly shines):
    ```bash
    # Two people having a conversation
    cog predict \
      -i image=@two_people.jpg \
      -i first_audio=@person1_speech.wav \
      -i second_audio=@person2_speech.wav \
      -i prompt="Two friends having an animated discussion about their favorite movies"
    
    # Podcast-style conversation
    cog predict \
      -i image=@podcast_setup.jpg \
      -i first_audio=@host_audio.wav \
      -i second_audio=@guest_audio.wav \
      -i prompt="A smiling man and woman wearing headphones sit in front of microphones, appearing to host a podcast" \
      -i num_frames=181
    
    # Interview scenario
    cog predict \
      -i image=@interview.jpg \
      -i first_audio=@interviewer.wav \
      -i second_audio=@interviewee.wav \
      -i prompt="A professional interview taking place in a modern office setting"
    ```

    **Explicit speaker + bboxes**

    Bounding boxes use the **same indexing as `wan/multitalk.py`**, not common “`[x1, y1, x2, y2]` pixel corners” order.

    - **Format:** `[row_min, col_min, row_max, col_max]` (four numbers in that order).
    - **Meaning:** On the conditioning image, **before** the model resizes it, PIL reports `width` × `height`. Masks are built with shape `[height, width]`. The pipeline does `human_mask[row_min:row_max, col_min:col_max] = 1`: first pair is the **vertical** (row) span, second pair is the **horizontal** (column) span.
    - **This is not** `[x1, y1, x2, y2]` where people usually put **x = horizontal** first. If you pass corner-style coordinates, **swap** so rows come first and columns second.

    **Example (640×480 image in PIL: `width=640`, `height=480`):** left half of the frame is columns `0–320`, full height rows `0–480` → `[0, 0, 480, 320]`. Right half → `[0, 320, 480, 640]`.

    `inactive_speaker_mode` defaults to `auto` (`second_audio` when `second_audio` is set, otherwise `none`).

    ```bash
    # Single speaker (unchanged): one person slot, no bboxes
    cog predict \
      -i image=@person.jpg \
      -i first_audio=@speech.wav \
      -i prompt="A person speaking to camera"

    # Two speakers: same as before — first_audio → person1, second_audio → person2 (default audio_type add)
    cog predict \
      -i image=@two_people.jpg \
      -i first_audio=@left.wav \
      -i second_audio=@right.wav \
      -i prompt="Two people in conversation"

    # Two people in frame, only person2 speaks: bboxes [row_min,col_min,row_max,col_max]; 640x480 PIL image (640 wide), left/right halves
    cog predict \
      -i image=@duo.jpg \
      -i first_audio=@only_right_speaks.wav \
      -i prompt="Two people seated; only one is talking" \
      -i person1_bbox='[0,0,480,320]' \
      -i person2_bbox='[0,320,480,640]' \
      -i active_speaker=person2 \
      -i inactive_speaker_mode=none
    ```

    Advanced generation control:
    ```bash
    # High-quality long-form generation
    cog predict \
      -i image=@speakers.jpg \
      -i first_audio=@long_speech.wav \
      -i prompt="A confident speaker delivering an important presentation" \
      -i num_frames=201 \
      -i sampling_steps=50 \
      -i turbo=false \
      -i seed=42
    
    # Fast generation with turbo mode
    cog predict \
      -i image=@quick_demo.jpg \
      -i first_audio=@short_audio.wav \
      -i prompt="A person giving a quick demo" \
      -i sampling_steps=20 \
      -i turbo=true
    
    # Reproducible results with fixed seed
    cog predict \
      -i image=@test_subject.jpg \
      -i first_audio=@test_audio.wav \
      -i prompt="A test subject for video generation experiments" \
      -i seed=123456 \
      -i num_frames=81
    ```

## How it works

This Cog implementation faithfully reproduces the original MultiTalk research pipeline with several optimizations for production use. Here's what happens under the hood:

*   `setup()` method: When the container starts up:
    1.  Downloads the complete MultiTalk model stack from Replicate's CDN (~30GB total):
        - Wan2.1-I2V-14B-480P: The 14 billion parameter base video generation model
        - chinese-wav2vec2-base: Audio encoder for speech feature extraction
        - MeiGen-MultiTalk: Custom audio conditioning weights trained for conversational video
    2.  Sets up GPU optimizations based on available memory (A100/H100 get the best performance settings)
    3.  Sets up the audio processing pipeline
    4.  Sets up the video generation pipeline

*   `predict()` method: Here's what happens:
    1.  Audio processing: Extracts audio from video files if needed, normalizes loudness, and handles both single and multi-person scenarios
    2.  Feature extraction: Uses the audio encoder to convert speech into data that captures timing and emotional content
    3.  Multi-person coordination: For conversations, combines multiple audio streams while keeping them aligned
    4.  Video generation: The 14 billion parameter model generates frames based on both the reference image and audio data
    5.  Sampling: Uses acceleration techniques for quality/speed balance
    6.  Post-processing: Combines generated video with original audio for synchronization

MultiTalk's key innovation is its ability to understand conversational dynamics—it doesn't just make mouths move, it generates natural interactions between people that follow the flow and emotional content of the conversation.

## Why MultiTalk is different

Traditional talking head generators can only animate single speakers with basic lip movements. MultiTalk changes this by:

- Understanding conversations: It grasps the back-and-forth nature of human dialogue and generates appropriate visual responses
- Multi-person awareness: Handles complex scenarios where multiple people interact naturally
- Audio-visual coherence: Creates synchronization not just of lip movements, but of facial expressions and body language that match the audio's emotional content
- Versatility: Works across different types of content—serious conversations, casual chats, singing, even cartoon characters

The research shows that MultiTalk can generate up to 15-second videos with consistent character appearance and natural interaction patterns that would previously require expensive motion capture and professional video production.

## Performance optimizations

This Cog implementation includes several performance optimizations:

- Automatic memory detection: Optimizes settings based on your GPU's memory capacity
- Turbo mode: Faster generation with optimized sampling parameters
- Acceleration: Speeds up inference by 2-3x with minimal quality loss
- Smart frame adjustment: Automatically corrects frame counts to valid values (4n+1 format)
- GPU memory management: Efficient cleanup between runs for consistent performance

## Deploy to Replicate

Want to share MultiTalk with the world? Push it to Replicate:

```bash
cog login
cog push r8.im/your-username/multitalk
```

## License

This implementation follows the original MultiTalk project's Apache 2.0 license. The MultiTalk model and research are from MeiGen-AI.

---

⭐ Star this on [GitHub](https://github.com/zsxkib/cog-MultiTalk)!

👋 Follow `zsakib_` on [Twitter/X](https://twitter.com/zsakib_)

**Enjoying MultiTalk?** Check out the original project and give the MeiGen team some love: [github.com/MeiGen-AI/MultiTalk](https://github.com/MeiGen-AI/MultiTalk)
