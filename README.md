# 🜏 Personal Scripts

A repository of scripts that solve my problems. There are no guarantees. If you choose to use them, the outcome is your concern, not mine.


## Using `docker.py`

`docker.py` runs the other scripts inside a Docker container, automatically setting up the environment based on the script’s header.
Simply mount your data, specify needed options, and execute.

**Basic Usage:**
```bash
./docker.py [options] <script.py> -- [script_args]
```

**Examples:**

- **Mount Data (`-d`):**
  ```bash
  ./docker.py -d /path/to/audio 'Audio Recognition/speech_to_text.py' -v /data/file.mp3
  ```
  Mounts `/path/to/audio` to `/data` in the container.

- **Enable GPU Support (`-G`):**
  ```bash
  ./docker.py -G 'Image Manipulation/upscale_image.py'
  ```
  Enables GPU access for GPU-accelerated tasks.

- **Set Environment Variables (`-e`):**
  ```bash
  ./docker.py -e OPENAI_API_KEY=your_api_key 'Audio Recognition/speech_to_text.py' --provider openai
  ```
  Passes `OPENAI_API_KEY` into the container.

- **Rebuild Without Cache (`-N`):**
  ```bash
  ./docker.py -N 'System Analysis/detect_hardware.py'
  ```
  Forces Docker to rebuild the image without using cache.
