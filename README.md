# 🜏 Personal Scripts

A repository of scripts that solve my problems. There are no guarantees. If you choose to use them, the outcome is your concern, not mine.


## Using `q.py`

`q.py` can be used to download and run scripts from this repository.

**Setup:**
```bash
wget https://raw.githubusercontent.com/zit-hb/Personal-Scripts/refs/heads/master/Tools/q.py -O ~/.local/bin/q
chmod +x ~/.local/bin/q
q --install
```

**Basic Usage:**

To run a script (through `docker.py` or natively), simply use the following command:
```bash
q [q_options] s [s_options] -- [docker_options] <script.py> [script_args]
```

**Aliases:**

You can add aliases to run scripts with even fewer characters:
```bash
q s --alias hw 'System Analysis/detect_hardware.py'
q s -- hw -h
```

You can use `--disable-docker/-D` to run scripts natively:
```bash
q s --alias dn 'Network Analysis/diagnose_network.py'
q s -D dn co si
```

## Using `docker.py`

`docker.py` runs the other scripts inside a Docker container, automatically setting up the environment based on the script’s header.
Simply mount your data, specify needed options, and execute.

**Basic Usage:**
```bash
./docker.py [options] -- <script.py> [script_args]
```

**Examples:**

- **Mount Data (`-d`):**
  ```bash
  ./docker.py -d /path/to/audio 'Audio Recognition/speech_to_text.py' -v /data/file.mp3
  ```
  Mounts `/path/to/audio` to `/data` in the container.

- **Open Ports (`-p`):**
  ```bash
  ./docker.py -p 80:80 -p 8080:8080 'Network Analysis/tcp_honeypot.py' -p 80 -p 8080 -o
  ```
  Forwards ports `80` and `8080` to the host machine.

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

- **Enable Privileged Mode (`-P`):**
  ```bash
  ./docker.py -P 'System Analysis/detect_hardware.py'
  ```
  Enables privileged mode for tasks that require it.

- **Enable GPU Support (`-G`):**
  ```bash
  ./docker.py -G 'Image Manipulation/upscale_image.py'
  ```
  Enables GPU access for GPU-accelerated tasks.
